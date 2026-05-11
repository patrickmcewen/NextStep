# MoE forward pass expressed purely with DSL ops.
# ------------------------------------------------
# 1. Load the token matrix `x` as a stream of B tiles (tile_row=1, tile_col=D).
# 2. Load the per‑token top‑k expert weights (`expert_weights`) as a Tile stream
#    with shape (1, B, n_active, 1, 1); this stream also serves as the reference
#    for broadcasting the token stream to the n_active dimension.
# 3. Promote `x` to add a singleton stream dimension next to the tile pair,
#    then expand it using the weight‑stream reference so that each token appears
#    once for each of its active experts, giving a token stream of shape
#    (1, B, n_active, 1, D).
# 4. Load the selector (MultiHot) that encodes which experts each token‑slot
#    uses.  This selector is used with `flat_partition` / `flat_reassemble`
#    to route tokens to/from per‑expert streams.
# 5. Partition both the token stream and the scalar‑weight stream per expert
#    via `flat_partition`.  Each partition yields a dynamic‑size stream of
#    tokens (or weights) for that expert.
# 6. For each expert:
#       – Load its three weight matrices (gate, up, down) as single‑tile streams.
#       – Broadcast each matrix to the expert‑specific token stream using
#         `expand_ref` (replace the two leading singleton stream dims).
#       – Perform the expert forward pass:
#           gate_out = token @ gate
#           up_out   = token @ up
#           proj     = silu(gate_out) * up_out
#           down_out = proj @ down
#       – Scale the result by the expert’s scalar weight.
# 7. Re‑assemble the per‑expert contributions back into the original token order
#    with `flat_reassemble`.
# 8. Reduce the two newly‑added stream dimensions (the ragged count dimension
#    and the n_active dimension) with `accum_add(rank=2)`.
# 9. Write the final (B, D) tensor off‑chip via `offchip_store`.
#
# All tensor arithmetic is performed through DSL calls; no raw torch ops,
# indexing, or new tensors are created.

def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # 1) Token stream: one tile per token, full feature vector.
    x = offchip_load(
        tensors["x"],
        stride=[1],
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )  # → stream(1, B)×tile(1, D)

    # 2) Load per‑position scalar expert weights; also serves as broadcast ref.
    weight_stream = offchip_load(
        tensors["expert_weights"],
        stride=[n_active, 1],
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )  # → stream(1, B, n_active)×tile(1, 1)

    # 3) Expand token stream to have an `n_active` dimension.
    x_promoted = promote(x, rank=0)                     # → stream(1, B, 1)×tile(1, D)
    token_stream = expand_ref(x_promoted, weight_stream, expand_rank=1)  # → stream(1, B, n_active)×tile(1, D)

    # 4) Multi‑hot selector for top‑k experts.
    control = select_gen(
        tensors["expert_onehot"],
        is_multihot=True,
        n=n_experts,
    )  # → stream(1, B, n_active)×tile(2, 8) (MultiHot)

    # 5) Partition tokens & scalar weights per expert.
    token_parts = flat_partition(token_stream, control, n_experts)   # list[StepTensor]
    weight_parts = flat_partition(weight_stream, control, n_experts)  # list[StepTensor]

    expert_outputs = []
    for i in range(n_experts):
        # Token and weight streams for expert i.
        tok = token_parts[i]      # stream(dynamic)×tile(1, D)
        wgt = weight_parts[i]     # stream(dynamic)×tile(1, 1)

        # ---- Load and broadcast expert weight matrices ----
        # Gate weight matrix (D × F)
        gate_raw = offchip_load(
            tensors["gate_weights"][i],
            stride=[1, 1],
            out_shape_tiled=(1, 1),
            tile_row=D,
            tile_col=F,
        )
        gate = expand_ref(gate_raw, tok, expand_rank=2)   # → stream(dynamic)×tile(D, F)

        # Up weight matrix (D × F)
        up_raw = offchip_load(
            tensors["up_weights"][i],
            stride=[1, 1],
            out_shape_tiled=(1, 1),
            tile_row=D,
            tile_col=F,
        )
        up = expand_ref(up_raw, tok, expand_rank=2)       # → stream(dynamic)×tile(D, F)

        # Down weight matrix (F × D)
        down_raw = offchip_load(
            tensors["down_weights"][i],
            stride=[1, 1],
            out_shape_tiled=(1, 1),
            tile_row=F,
            tile_col=D,
        )
        down = expand_ref(down_raw, tok, expand_rank=2)   # → stream(dynamic)×tile(F, D)

        # ---- Expert forward computation ----
        gate_out = binary_matmul(tok, gate)               # → stream(dynamic)×tile(1, F)
        up_out   = binary_matmul(tok, up)                 # → stream(dynamic)×tile(1, F)

        proj = binary_mul(unary_silu(gate_out), up_out)   # → stream(dynamic)×tile(1, F)

        down_out = binary_matmul(proj, down)              # → stream(dynamic)×tile(1, D)

        # Scale by the per‑expert scalar weight.
        weighted = binary_mul(down_out, wgt)              # → stream(dynamic)×tile(1, D)

        expert_outputs.append(weighted)

    # 7) Re‑assemble per‑expert contributions in the original token order.
    merged = flat_reassemble(expert_outputs, control)

    # 8) Collapse the ragged dimension and the n_active dimension.
    y = accum_add(merged, rank=2)  # → stream(1, B, 1)×tile(1, D)

    # 9) Write the result off‑chip.
    return offchip_store(y)