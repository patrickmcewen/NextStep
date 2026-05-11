# Implementation reasoning:
# 1️⃣ Stream the input activations.
# 2️⃣ Partition tokens per expert with a MultiHot selector.
# 3️⃣ Load each expert’s three weight matrices as singleton streams.
# 4️⃣ For each expert:
#    – Promote the token stream so it has three stream dims (1, N_i, 1).
#    – Promote the singleton weight streams to three dims and expand them to
#      (1, N_i, 1) with `expand_ref`.
#    – Perform gate → up → silu → down using `binary_matmul`,
#      `binary_mul`, and `unary_silu`.
#    – Collapse the leading outer singleton with the token‐count dimension
#      using `flatten`, yielding a stream of shape (N_i, 1, D).
# 5️⃣ Re‑assemble the per‑expert streams back to token order with an Index
#    control (`expert_onehot`) via `flat_reassemble`.
# 6️⃣ Load the per‑token expert scalar weights and promote them so their
#    stream shape matches the re‑assembled tensor.
# 7️⃣ Multiply the re‑assembled contributions by the scalar weights,
#    then sum over the top‑k expert dimension with `accum_add(rank=2)`.
# 8️⃣ Write the final (B, D) matrix back off‑chip.

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
        # token stream for expert i: (N_i,)×tile(1, D)
        token_i = token_streams[i]

        # Add an inner singleton then an outer singleton → (1, N_i, 1)
        token_i_inner = promote(token_i, rank=0)        # (N_i, 1)×tile(1, D)
        token_i_full = promote_outer(token_i_inner)    # (1, N_i, 1)×tile(1, D)

        # ---- Broadcast weight matrices to the token‑stream shape ----
        gate_i_pre = promote(gate_streams[i], rank=0)   # (1, 1, 1)×tile(D, F)
        gate_i = expand_ref(gate_i_pre, token_i_full, expand_rank=2)

        up_i_pre = promote(up_streams[i], rank=0)       # (1, 1, 1)×tile(D, F)
        up_i = expand_ref(up_i_pre, token_i_full, expand_rank=2)

        # ---- Expert forward pass ----
        gate_out = binary_matmul(token_i_full, gate_i)          # (1, N_i, 1)×tile(1, F)
        up_out   = binary_matmul(token_i_full, up_i)            # (1, N_i, 1)×tile(1, F)
        proj     = binary_mul(unary_silu(gate_out), up_out)      # (1, N_i, 1)×tile(1, F)

        # ---- Down projection ----
        down_i_pre = promote(down_streams[i], rank=0)            # (1, 1, 1)×tile(F, D)
        down_i = expand_ref(down_i_pre, proj, expand_rank=2)
        down_out = binary_matmul(proj, down_i)                   # (1, N_i, 1)×tile(1, D)

        # Collapse the leading outer singleton with the token‑count dimension
        down_flat = flatten(down_out, min_rank=1, max_rank=2)    # (N_i, 1)×tile(1, D)

        expert_outputs.append(down_flat)

    # ------------------------------------------------------------------
    # 5. Re‑assemble tokens in original order (Index selector)
    # ------------------------------------------------------------------
    reassemble_ctrl = select_gen(
        tensors["expert_onehot"], is_multihot=False, n=n_experts
    )
    merged = flat_reassemble(expert_outputs, reassemble_ctrl)   # (1, B, 2, 1, 1, D)

    # ------------------------------------------------------------------
    # 6. Load per‑token expert scalar weights and promote to match `merged`
    # ------------------------------------------------------------------
    weight_stream = offchip_load(
        tensors["expert_weights"],
        stride=(n_active, 1),
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )                                                          # (1, B, 2)×tile(1,1)
    weight_stream_full = promote(weight_stream, rank=0)        # (1, B, 2, 1)×tile(1,1)

    # ------------------------------------------------------------------
    # 7. Apply scalar weights and sum over the top‑k expert dimension
    # ------------------------------------------------------------------
    weighted = binary_mul(merged, weight_stream_full)          # (1, B, 2, 1, 1, D)
    result = accum_add(weighted, rank=2)                       # (1, B, 1, D)

    # ------------------------------------------------------------------
    # 8. Write the final (B, D) matrix back off‑chip
    # ------------------------------------------------------------------
    return offchip_store(result)