# MoE forward pass expressed solely with DSL primitives.
# -------------------------------------------------
# 1. Load the input token matrix `x`.
# 2. Load per‑token scalar expert‑weight values (`expert_weights`);
#    this stream also serves as the reference for broadcasting the token stream
#    to the active‑expert dimension.
# 3. Expand the token stream so each token appears once per its selected expert.
# 4. Load the top‑k selector (`expert_onehot`) as a MultiHot stream.
# 5. Partition both the token stream and the scalar‑weight stream per expert.
# 6. For each expert:
#        – Load the three weight matrices (gate, up, down) as singleton streams.
#        – Broadcast each matrix to the expert‑specific token stream.
#        – Compute the expert forward pass:
#              gate_out = token @ gate
#              up_out   = token @ up
#              proj     = silu(gate_out) * up_out
#              down_out = proj @ down
#        – Scale the result by the expert‑specific scalar weight.
# 7. Re‑assemble the per‑expert contributions into the original token order.
# 8. Reduce over the `n_active` dimension (rank=1) to obtain the final output.
# 9. Store the result off‑chip.
#
# All tensor arithmetic is performed via DSL calls; no raw PyTorch ops,
# indexing, or tensor creation (apart from the allowed off‑chip loads) are used.

def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # -----------------------------------------------------------------
    # 1) Load token matrix x (one tile per token).
    x = offchip_load(
        tensors["x"],
        stride=[1],
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )  # → stream(1, B) × tile(1, D)

    # -----------------------------------------------------------------
    # 2) Load per‑token scalar expert weights (also the broadcast reference).
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
    )  # → stream(1, B, n_active, 1) × tile(1, D)

    # -----------------------------------------------------------------
    # 4) Multi‑hot selector for top‑k experts.
    control = select_gen(
        tensors["expert_onehot"],
        is_multihot=True,
        n=n_experts,
    )  # → stream(1, B) × tile(2, 8) (MultiHot)

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
        # 6) Load and broadcast expert weight matrices (gate, up, down).
        #   Each matrix is loaded as a (1,1) stream and then expanded to match
        #   the expert‑specific token stream.
        gate_raw = offchip_load(
            tensors["gate_weights"][i],
            stride=[0],
            out_shape_tiled=(1,),
            tile_row=D,
            tile_col=F,
        )
        gate = expand_ref(gate_raw, tok, expand_rank=2)   # → stream(dynamic) × tile(D, F)

        up_raw = offchip_load(
            tensors["up_weights"][i],
            stride=[0],
            out_shape_tiled=(1,),
            tile_row=D,
            tile_col=F,
        )
        up = expand_ref(up_raw, tok, expand_rank=2)       # → stream(dynamic) × tile(D, F)

        down_raw = offchip_load(
            tensors["down_weights"][i],
            stride=[0],
            out_shape_tiled=(1,),
            tile_row=F,
            tile_col=D,
        )
        down = expand_ref(down_raw, tok, expand_rank=2)   # → stream(dynamic) × tile(F, D)

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
    # 9) Collapse the n_active dimension (rank = 1) – keep the batch dimension.
    y = accum_add(merged, rank=1)  # → stream(1, B, 1) × tile(1, D)

    # -----------------------------------------------------------------
    # 10) Write the result off‑chip.
    return offchip_store(y)