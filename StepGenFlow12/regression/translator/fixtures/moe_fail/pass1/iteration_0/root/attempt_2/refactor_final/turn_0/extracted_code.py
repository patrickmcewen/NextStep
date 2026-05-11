# MoE implementation using the STeP DSL.
# --------------------------------------------------------------
# 1) Load the token matrix `x` as a tiled stream (tile_n=1, tile_f=256).
#    This yields a stream shape (B, D//tile_f) where each tile holds
#    a 1×256 slice of the feature dimension.
#
# 2) Build a MultiHot selector from `expert_multihot` (shape B×n_experts).
#    The selector is repeated over the feature‑tile dimension so that it
#    has the same number of stream elements as `x`.  This repeated selector
#    is used for both routing the token tiles and for re‑assembling the
#    per‑expert results.
#
# 3) Partition `x` into per‑expert token streams using the repeated
#    selector.  Tokens that belong to an expert appear in that expert’s
#    stream (tokens may appear in two streams because top‑k=2).
#
# 4) Load the scalar expert weights (shape B×n_active) and partition them
#    with an Index selector derived from `expert_onehot`.  After partition
#    each per‑expert weight stream is repeated over the feature‑tile
#    dimension and flattened so that its stream shape matches the token
#    stream of the same expert.
#
# 5) For each expert:
#       – Load the three weight matrices (gate, up, down) once, using
#         `offchip_load_ref` with a reference stream that matches `x`.
#       – Partition those matrices with the same MultiHot selector used for
#         the tokens, obtaining per‑expert weight streams that align with
#         the token streams.
#       – Compute `gate = silu(x @ gate_w)` and `up = x @ up_w`.
#       – Multiply to obtain the projected representation.
#       – Repeat the projection over the F‑tile dimension and multiply with
#         the down‑projection weight.  Reduce over the F‑tile dimension
#         with `accum_add` to obtain the expert’s contribution to the
#         output (still tiled over the D‑tile dimension).
#       – Multiply by the per‑token expert scalar weight.
#
# 6) Re‑assemble the per‑expert contributions with `flat_reassemble`
#    (using the same repeated MultiHot selector) which creates a new
#    dynamic dimension for the active experts per token.  Reduce over that
#    dimension with `accum_add` to sum the two expert contributions.
#
# 7) Store the final tiled stream off‑chip.
#
# The implementation follows the stream‑shape invariant: every intermediate
# tensor is a stream with at least one stream dimension followed by a
# (tile_row, tile_col) tile pair.  All arithmetic and shape manipulation are
# performed via the DSL functions; no raw tensor operations are used.
# --------------------------------------------------------------

def tiled_reference(dims, tensors):
    # ------------------------------------------------------------------
    # Basic dimensions
    # ------------------------------------------------------------------
    B = dims["B"]                     # batch / token count
    D = dims["D"]                     # model dimension
    F = dims["F"]                     # hidden dimension
    n_experts = dims["n_experts"]     # total number of experts
    n_active = dims["n_active"]       # top‑k (2)
    tile_n = dims["tile_n"]           # usually 1
    tile_f = dims["tile_f"]           # 256

    # Number of tiles along the feature dimensions
    K_tiles = D // tile_f              # D‑tiles (inner dim for gate/up)
    F_tiles = F // tile_f              # F‑tiles (outer dim for gate/up, inner for down)

    # ------------------------------------------------------------------
    # Load the input activation matrix X (B, D) as a tiled stream.
    # Each tile is (tile_n=1, tile_f=256); the stream shape is (B, K_tiles).
    # ------------------------------------------------------------------
    X = offchip_load(
        tensors["x"],
        stride=[K_tiles, 1],            # advance over feature tiles
        out_shape_tiled=[B, K_tiles],
        tile_row=tile_n,
        tile_col=tile_f,
    )

    # ------------------------------------------------------------------
    # Create a MultiHot selector that tells which experts each token
    # should be sent to.  The selector is repeated over the K‑tile dimension
    # so that it aligns with the stream shape of X.
    # ------------------------------------------------------------------
    selector_mh = select_gen(
        tensors["expert_multihot"],
        is_multihot=True,
        n=n_experts,
    )
    # broadcast across the K‑tile dimension
    selector_rep = repeat_static(selector_mh, factor=K_tiles)

    # ------------------------------------------------------------------
    # Partition the token stream into per‑expert streams.
    # ------------------------------------------------------------------
    token_per_expert = flat_partition(X, selector_rep, n_experts)

    # ------------------------------------------------------------------
    # Load the per‑token scalar weights (shape B × n_active) and route them
    # to experts using the Index selector derived from `expert_onehot`.
    # ------------------------------------------------------------------
    weight_raw = offchip_load(
        tensors["expert_weights"],
        stride=[n_active, 1],
        out_shape_tiled=[B, n_active],
        tile_row=1,
        tile_col=1,
    )
    selector_idx = select_gen(
        tensors["expert_onehot"],
        is_multihot=False,
        n=n_experts,
    )
    weight_per_expert = flat_partition(weight_raw, selector_idx, n_experts)

    # ------------------------------------------------------------------
    # Load the three expert weight matrices once (gate, up, down).  They are
    # broadcast to the full token stream `X` using offchip_load_ref; then they
    # are partitioned with the same MultiHot selector so that each expert
    # receives a stream that aligns with its token stream.
    # ------------------------------------------------------------------
    # Gate weight: (D, F)  -> tile (tile_f, tile_f)
    gate_w_static = offchip_load_ref(
        X,
        tensors["gate_weights"],
        stride=[0, F_tiles],
        out_shape_tiled=[B, K_tiles],
        tile_row=tile_f,
        tile_col=tile_f,
    )
    gate_w_per_expert = flat_partition(gate_w_static, selector_rep, n_experts)

    # Up weight: (D, F)  -> same tiling as gate
    up_w_static = offchip_load_ref(
        X,
        tensors["up_weights"],
        stride=[0, F_tiles],
        out_shape_tiled=[B, K_tiles],
        tile_row=tile_f,
        tile_col=tile_f,
    )
    up_w_per_expert = flat_partition(up_w_static, selector_rep, n_experts)

    # Down weight: (F, D)  grid (F_tiles, K_tiles)
    # out_shape includes the extra F_tiles dimension so that later we can
    # repeat the projection over that dimension.
    down_w_static = offchip_load_ref(
        X,
        tensors["down_weights"],
        stride=[0, 1, K_tiles],
        out_shape_tiled=[B, K_tiles, F_tiles],
        tile_row=tile_f,
        tile_col=tile_f,
    )
    down_w_per_expert = flat_partition(down_w_static, selector_rep, n_experts)

    # ------------------------------------------------------------------
    # Process each expert independently.
    # ------------------------------------------------------------------
    expert_outputs = []
    for i in range(n_experts):
        # Token tiles for this expert (stream: dynamic, tile 1×256)
        x_i = token_per_expert[i]

        # Corresponding scalar weight stream (tile 1×1).  It is repeated over
        # the K‑tile dimension and flattened so that its stream shape matches
        # the token stream.
        w_i = weight_per_expert[i]                       # (dyn, 1,1)
        w_i_rep = repeat_static(w_i, factor=K_tiles)    # (dyn, K_tiles, 1,1)
        w_i_flat = flatten(w_i_rep, min_rank=0, max_rank=1)  # (dyn*K_tiles)

        # ---------- Gate and Up projections ----------
        gate_i = binary_matmul(x_i, gate_w_per_expert[i])
        gate_i = unary_silu(gate_i)
        up_i = binary_matmul(x_i, up_w_per_expert[i])
        proj_i = binary_mul(gate_i, up_i)               # (dyn, 1,256)

        # ---------- Down projection ----------
        # Repeat the projection over the F‑tile dimension so that we can
        # multiply with the down‑weight slice for each F‑tile.
        proj_rep = repeat_static(proj_i, factor=F_tiles)   # (dyn, F_tiles, 1,256)
        down_mat = binary_matmul(proj_rep, down_w_per_expert[i])
        # Reduce over the F‑tile dimension (rank=1)
        down_red = accum_add(down_mat, rank=1)

        # ---------- Apply the scalar expert weight ----------
        weighted = binary_mul(down_red, w_i_flat)

        expert_outputs.append(weighted)

    # ------------------------------------------------------------------
    # Re‑assemble the per‑expert contributions using the same repeated
    # MultiHot selector.  This yields a new dynamic dimension (size = n_active)
    # that we then sum over to obtain the final MoE output.
    # ------------------------------------------------------------------
    merged = flat_reassemble(expert_outputs, selector_rep)
    final = accum_add(merged, rank=1)

    # ------------------------------------------------------------------
    # Write the tiled result off‑chip.
    # ------------------------------------------------------------------
    return offchip_store(final)