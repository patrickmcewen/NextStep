# Implementation reasoning:
# 1️⃣ Stream the input activations.
# 2️⃣ Partition tokens per expert with a MultiHot selector.
# 3️⃣ Load each expert’s three weight matrices as singleton streams.
# 4️⃣ For each expert:
#    – Promote the token stream so it has three stream dimensions (1, N_i, 1),
#      matching the singleton weight streams.
#    – Expand the singleton weight streams to the token‑stream shape with
#      `expand_ref`.
#    – Run the expert forward pass (gate → up → silu → down) using matmul
#      and elementwise ops.
#    – Collapse the leading outer singleton with the token‑count dimension
#      using `flatten`, yielding a stream of shape (N_i, 1, D).
# 5️⃣ Re‑assemble the per‑expert streams back to token order with an Index
#    control (`expert_onehot`) via `flat_reassemble`.  This introduces a new
#    dynamic stream dimension that holds the ragged token counts.
# 6️⃣ Load the per‑token expert scalar weights, promote them to have an extra
#    singleton, and expand that singleton into the dynamic dimension created
#    by the re‑assemble step using `expand_ref`.
# 7️⃣ Multiply the re‑assembled contributions by the scalar weights, then
#    reduce BOTH the dynamic token‑count dimension and the static
#    top‑k‑expert dimension in one `accum_add(rank=2)` call.
# 8️⃣ Write the final (B, D) matrix back off‑chip with `offchip_store`.

def tiled_reference(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # ------------------------------------------------------------------
    # 1. Stream the input activations (B × D) as tiles (1, D)
    # ------------------------------------------------------------------
    x_stream = offchip_load(
        tensors["x"],
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
    )

    # ------------------------------------------------------------------
    # 2. Partition tokens per expert (MultiHot selector)
    # ------------------------------------------------------------------
    partition_ctrl = select_gen(
        tensors["expert_multihot"], is_multihot=True, n=n_experts
    )
    token_streams = flat_partition(x_stream, partition_ctrl, n=n_experts)

    # ------------------------------------------------------------------
    # 3. Load per‑expert weight matrices as singleton streams
    # ------------------------------------------------------------------
    gate_streams = [
        offchip_load(
            tensors["gate_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=D,
            tile_col=F,
        )
        for i in range(n_experts)
    ]
    up_streams = [
        offchip_load(
            tensors["up_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=D,
            tile_col=F,
        )
        for i in range(n_experts)
    ]
    down_streams = [
        offchip_load(
            tensors["down_weights"][i],
            stride=(1,),
            out_shape_tiled=(1,),
            tile_row=F,
            tile_col=D,
        )
        for i in range(n_experts)
    ]

    # ------------------------------------------------------------------
    # 4. Compute each expert's (un‑scaled) contribution
    # ------------------------------------------------------------------
    expert_outputs = []
    for i in range(n_experts):
        # Token stream for expert i: (N_i,)×tile(1, D)
        token_i = token_streams[i]

        # Promote to (1, N_i, 1) stream shape
        token_i_inner = promote(token_i, rank=0)          # (N_i, 1)×tile(1, D)
        token_i_full = promote_outer(token_i_inner)       # (1, N_i, 1)×tile(1, D)

        # Broadcast weight matrices to match (1, N_i, 1) stream shape
        gate_i_pre = promote(gate_streams[i], rank=0)     # (1, 1, 1)×tile(D, F)
        gate_i = expand_ref(gate_i_pre, token_i_full, expand_rank=2)

        up_i_pre = promote(up_streams[i], rank=0)         # (1, 1, 1)×tile(D, F)
        up_i = expand_ref(up_i_pre, token_i_full, expand_rank=2)

        # Expert forward pass
        gate_out = binary_matmul(token_i_full, gate_i)                # (1, N_i, 1)×tile(1, F)
        up_out   = binary_matmul(token_i_full, up_i)                  # (1, N_i, 1)×tile(1, F)
        proj     = binary_mul(unary_silu(gate_out), up_out)           # (1, N_i, 1)×tile(1, F)

        # Broadcast down matrix and finish matmul
        down_i_pre = promote(down_streams[i], rank=0)                  # (1, 1, 1)×tile(F, D)
        down_i = expand_ref(down_i_pre, proj, expand_rank=2)
        down_out = binary_matmul(proj, down_i)                         # (1, N_i, 1)×tile(1, D)

        # Collapse the leading outer singleton with the token count
        down_flat = flatten(down_out, min_rank=1, max_rank=2)          # (N_i, 1)×tile(1, D)

        expert_outputs.append(down_flat)

    # ------------------------------------------------------------------
    # 5. Re‑assemble tokens in original order (Index selector)
    # ------------------------------------------------------------------
    reassemble_ctrl = select_gen(
        tensors["expert_onehot"], is_multihot=False, n=n_experts
    )
    merged = flat_reassemble(expert_outputs, reassemble_ctrl)   # (1, B, n_active, dyn)×tile(1, D)

    # ------------------------------------------------------------------
    # 6. Load per‑token expert scalar weights and broadcast to the dynamic dim
    # ------------------------------------------------------------------
    weight_stream = offchip_load(
        tensors["expert_weights"],
        stride=(n_active, 1),
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )                                                          # (1, B, n_active)×tile(1,1)

    # Add an inner singleton so we can replace it with the ragged dim
    weight_stream_full = promote(weight_stream, rank=0)        # (1, B, n_active, 1)×tile(1,1)

    # Expand the trailing singleton into the dynamic dimension of `merged`
    weight_exp = expand_ref(weight_stream_full, merged, expand_rank=1)

    # ------------------------------------------------------------------
    # 7. Apply scalar weights and sum over both ragged and top‑k dimensions
    # ------------------------------------------------------------------
    weighted = binary_mul(merged, weight_exp)                 # (1, B, n_active, dyn)×tile(1, D)
    result = accum_add(weighted, rank=2)                      # (1, B)×tile(1, D)

    # ------------------------------------------------------------------
    # 8. Write result back off‑chip
    # ------------------------------------------------------------------
    return offchip_store(result)