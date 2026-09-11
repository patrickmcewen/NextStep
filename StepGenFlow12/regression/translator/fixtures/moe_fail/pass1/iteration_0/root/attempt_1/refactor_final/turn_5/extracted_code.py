# MoE forward pass using only DSL operations.
# -------------------------------------------------
# 1. Load the token matrix `x`.
# 2. Load per‑token scalar expert weights (`expert_weights`).  This stream also
#    serves as the broadcast reference for expanding the token stream to the
#    active‑expert dimension.
# 3. Expand the token stream so each token appears once per its selected expert.
# 4. Load the top‑k selector (`expert_onehot`) as a MultiHot stream.
# 5. Partition the expanded token stream and the scalar‑weight stream per expert.
# 6. For each expert:
#       – Broadcast the three weight matrices (gate, up, down) onto the expert‑
#         specific token stream with `offchip_load_ref`.
#       – Compute the expert forward pass:
#             gate_out = token @ gate
#             up_out   = token @ up
#             proj     = silu(gate_out) * up_out
#             down_out = proj @ down
#       – Scale the result by the per‑expert scalar weight.
# 7. Re‑assemble the per‑expert contributions in the original token order.
# 8. Reduce over the ragged dimension and the static n_active dimension.
# 9. Store the final (B, D) result off‑chip.
#
# All arithmetic is expressed via DSL calls; no raw PyTorch ops or indexing
# are used.

def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # -----------------------------------------------------------------
    # 1) Load token matrix x (one tile per token, full feature vector).
    x = offchip_load(
        tensors["x"],
        stride=[1],
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )  # → stream(1, B) × tile(1, D)

    # -----------------------------------------------------------------
    # 2) Load per‑token scalar expert weights; also acts as broadcast reference.
    weight_stream = offchip_load(
        tensors["expert_weights"],
        stride=[n_active, 1],
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )  # → stream(1, B, n_active) × tile(1, 1)

    # -----------------------------------------------------------------
    # 3) Broadcast tokens to the n_active dimension.
    x_promoted = promote(x, rank=0)                     # → stream(1, B, 1) × tile(1, D)
    token_stream = expand_ref(
        x_promoted, weight_stream, expand_rank=1
    )  # → stream(1, B, n_active) × tile(1, D)

    # -----------------------------------------------------------------
    # 4) Multi‑hot selector for top‑k experts.
    control = select_gen(
        tensors["expert_onehot"],
        is_multihot=True,
        n=n_experts,
    )  # → stream(1, B) × tile(2, 8)  (MultiHot)

    # -----------------------------------------------------------------
    # 5) Partition tokens & scalar weights per expert.
    token_parts = flat_partition(token_stream, control, n_experts)   # list[StepTensor]
    weight_parts = flat_partition(weight_stream, control, n_experts)  # list[StepTensor]

    expert_outputs = []
    for i in range(n_experts):
        # Token and scalar‑weight streams for expert i.
        tok = token_parts[i]      # stream(dynamic) × tile(1, D)
        wgt = weight_parts[i]     # stream(dynamic) × tile(1, 1)

        # -----------------------------------------------------------------
        # 6) Load and broadcast expert weight matrices onto the token stream.
        #    A singleton stream dimension (out_shape_tiled=(1,)) is used so that
        #    offchip_load_ref does not encounter an empty meshgrid.
        gate = offchip_load_ref(
            tok,
            tensors["gate_weights"][i],
            stride=[0],
            out_shape_tiled=(1,),
            tile_row=D,
            tile_col=F,
        )  # → stream(dynamic) × tile(D, F)

        up = offchip_load_ref(
            tok,
            tensors["up_weights"][i],
            stride=[0],
            out_shape_tiled=(1,),
            tile_row=D,
            tile_col=F,
        )  # → stream(dynamic) × tile(D, F)

        down = offchip_load_ref(
            tok,
            tensors["down_weights"][i],
            stride=[0],
            out_shape_tiled=(1,),
            tile_row=F,
            tile_col=D,
        )  # → stream(dynamic) × tile(F, D)

        # -----------------------------------------------------------------
        # 7) Expert forward computation.
        gate_out = binary_matmul(tok, gate)               # → stream(dynamic) × tile(1, F)
        up_out   = binary_matmul(tok, up)                 # → stream(dynamic) × tile(1, F)

        proj = binary_mul(unary_silu(gate_out), up_out)   # → stream(dynamic) × tile(1, F)

        down_out = binary_matmul(proj, down)              # → stream(dynamic) × tile(1, D)

        # Scale by the per‑expert scalar weight.
        weighted = binary_mul(down_out, wgt)              # → stream(dynamic) × tile(1, D)

        expert_outputs.append(weighted)

    # -----------------------------------------------------------------
    # 8) Re‑assemble per‑expert contributions in the original token order.
    merged = flat_reassemble(expert_outputs, control)

    # -----------------------------------------------------------------
    # 9) Collapse the ragged dimension and the static n_active dimension.
    y = accum_add(merged, rank=2)  # → stream(1, B, 1) × tile(1, D)

    # -----------------------------------------------------------------
    # 10) Write the result off‑chip.
    return offchip_store(y)