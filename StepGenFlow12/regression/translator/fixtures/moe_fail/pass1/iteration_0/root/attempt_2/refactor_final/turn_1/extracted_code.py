# MoE (Routed) implementation using only the Step DSL.
# --------------------------------------------------------------
# 1. Load the token matrix `x` as a **single‑row** tile (tile_row=1,
#    tile_col=D).  The stream shape is (B,) – one stream element per token.
# 2. Load the integer routing tensors (`expert_multihot`,
#    `expert_weights`) as tiled streams that line up with the token stream.
# 3. Use the MultiHot selector from `expert_multihot` to partition the
#    token stream (and the per‑token scalar weight stream) per expert
#    with `flat_partition`.  The resulting per‑expert streams each have a
#    dynamic outer dimension (the number of tokens routed to that expert)
#    and a tile shape (1, D) for the tokens and (1, 1) for the scalars.
# 4. Load the three weight matrices once per expert using
#    `offchip_load_ref`.  The gate/up matrices are broadcast as a single
#    tile (1024×256); the down matrix is streamed over the F‑tile dimension
#    (8×256×1024) so that it can be multiplied after the projection is
#    repeated across those F‑tiles.
# 5. For each expert:
#       – Matmul token → gate weight, apply SiLU.
#       – Matmul token → up weight.
#       – Elementwise multiply the two (projected).
#       – Repeat the projection over the 8 F‑tiles (`repeat_static`).
#       – Matmul with the down‑weight slice, then sum over the F‑tile
#         dimension (`accum_add`).
#       – Multiply by the per‑token expert scalar weight.
#    The result of each expert is a stream with the same per‑token shape.
# 6. Re‑assemble the per‑expert streams back into token order with
#    `flat_reassemble` using the original MultiHot selector, then sum the
#    two active‑expert contributions per token (`accum_add`).
# 7. Store the final tiled stream off‑chip.
# --------------------------------------------------------------

def tiled_reference(dims, tensors):
    # ------------------------------------------------------------------
    # Shapes and tiling parameters
    # ------------------------------------------------------------------
    B = dims["B"]                     # batch / token count
    D = dims["D"]                     # model dimension
    F = dims["F"]                     # hidden dimension
    n_experts = dims["n_experts"]     # total experts
    n_active = dims["n_active"]       # top‑k
    tile_f = dims["tile_f"]           # 256
    F_tiles = F // tile_f              # 8 (2048 / 256)

    # ------------------------------------------------------------------
    # Load the input activation matrix X (B tokens, each a 1×D tile)
    # ------------------------------------------------------------------
    X = offchip_load(
        tensors["x"],
        stride=[1],                     # advance across the batch dimension
        out_shape_tiled=[B],           # stream has B elements
        tile_row=1,
        tile_col=D,
    )

    # ------------------------------------------------------------------
    # MultiHot selector (B × n_experts) – one row per token.
    # ------------------------------------------------------------------
    selector = select_gen(
        tensors["expert_multihot"],
        is_multihot=True,
        n=n_experts,
    )

    # ------------------------------------------------------------------
    # Partition tokens and per‑token scalar weights per expert.
    # ------------------------------------------------------------------
    token_per_expert = flat_partition(X, selector, n_experts)

    # expert_weights has shape (B, n_active); treat each scalar as a tile.
    weight_raw = offchip_load(
        tensors["expert_weights"],
        stride=[2, 1],                  # (row_stride, col_stride) over (B, n_active)
        out_shape_tiled=[B, n_active],
        tile_row=1,
        tile_col=1,
    )
    weight_per_expert = flat_partition(weight_raw, selector, n_experts)

    # ------------------------------------------------------------------
    # Load the three weight matrices (broadcast across the token stream).
    # ------------------------------------------------------------------
    gate_w_static = offchip_load_ref(
        X,
        tensors["gate_weights"],
        stride=[0],                     # broadcast same tile for all tokens
        out_shape_tiled=[1],
        tile_row=D,
        tile_col=tile_f,
    )
    gate_w_per_expert = flat_partition(gate_w_static, selector, n_experts)

    up_w_static = offchip_load_ref(
        X,
        tensors["up_weights"],
        stride=[0],
        out_shape_tiled=[1],
        tile_row=D,
        tile_col=tile_f,
    )
    up_w_per_expert = flat_partition(up_w_static, selector, n_experts)

    # Down weights are tiled over the F dimension (8 tiles of 256×D).
    down_w_static = offchip_load_ref(
        X,
        tensors["down_weights"],
        stride=[1],                     # advance across the F‑tile dimension
        out_shape_tiled=[F_tiles],
        tile_row=tile_f,
        tile_col=D,
    )
    down_w_per_expert = flat_partition(down_w_static, selector, n_experts)

    # ------------------------------------------------------------------
    # Per‑expert computation
    # ------------------------------------------------------------------
    expert_outputs = []
    for i in range(n_experts):
        # Token subset for this expert (tile shape (1, D))
        x_i = token_per_expert[i]

        # Corresponding scalar weight for each token (tile shape (1, 1))
        w_i = weight_per_expert[i]

        # ---- Gate and Up projections ----
        gate_i = binary_matmul(x_i, gate_w_per_expert[i])
        gate_i = unary_silu(gate_i)                     # SiLU activation
        up_i   = binary_matmul(x_i, up_w_per_expert[i])

        # Elementwise product (projected representation, tile (1, tile_f))
        proj_i = binary_mul(gate_i, up_i)

        # Repeat the projected tile over the 8 F‑tiles
        proj_rep = repeat_static(proj_i, factor=F_tiles)   # stream (…, 8) × tile (1,256)

        # ---- Down projection and reduction over F‑tiles ----
        down_mat = binary_matmul(proj_rep, down_w_per_expert[i])
        down_red = accum_add(down_mat, rank=1)            # sum over the F‑tile stream

        # Apply the per‑token scalar expert weight
        weighted = binary_mul(down_red, w_i)

        expert_outputs.append(weighted)

    # ------------------------------------------------------------------
    # Re‑assemble per‑expert streams back into token order and sum the
    # two active‑expert contributions per token.
    # ------------------------------------------------------------------
    merged = flat_reassemble(expert_outputs, selector)   # adds a dynamic dim for active experts
    final = accum_add(merged, rank=1)                    # sum over that dynamic dim

    # ------------------------------------------------------------------
    # Write the result off‑chip.
    # ------------------------------------------------------------------
    return offchip_store(final)