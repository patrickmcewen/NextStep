# Implementation reasoning:
# • Stream the input matrix `x` as a (1‑D)‑by‑D tile stream.
# • Use `expert_multihot` (MultiHot) to partition the token stream per expert
#   with `flat_partition`.
# • Load each expert’s three weight matrices as singleton streams.
# • For each expert, promote the token stream to have two stream dimensions
#   (outer token count N_i and an inner singleton) and then add an outer
#   singleton so its stream shape becomes (1, N_i, 1).  This matches the two
#   singleton stream dimensions of the weight matrices.
# • Expand the weight matrices with `expand_ref` (replace both singleton
#   dimensions) to obtain streams of shape (1, N_i, 1) that line‑up with the
#   token stream.
# • Perform the expert forward pass (`gate → up → silu → down`) with
#   `binary_matmul` and elementwise ops.
# • After the down‑projection, `flatten` collapses the leading outer singleton
#   with the token‑count dimension, yielding a stream of shape (N_i, 1, D)
#   suitable for `flat_reassemble`.
# • `flat_reassemble` is driven by `expert_onehot` (an Index control) so that
#   the reassembled stream preserves the original top‑k ordering.
# • Load the per‑token expert weights (`expert_weights`) as a scalar tile
#   stream and multiply the reassembled result.
# • Finally, sum the two expert contributions per token with `accum_add` and
#   write the dense result back off‑chip with `offchip_store`.

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
    # 2. Partition tokens per expert (order‑agnostic)
    # ------------------------------------------------------------------
    control_partition = select_gen(
        tensors["expert_multihot"], is_multihot=True, n=n_experts
    )
    token_streams = flat_partition(x_stream, control_partition, n=n_experts)

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
    # 4. Compute each expert's (un‑weighted) contribution
    # ------------------------------------------------------------------
    expert_outputs = []
    for i in range(n_experts):
        # Token stream for this expert: (N_i, 1, D)
        token_i = token_streams[i]

        # Add an inner singleton dim → (N_i, 1, 1, D)
        token_i_inner = promote(token_i, rank=0)

        # Add an outer singleton → (1, N_i, 1, 1, D)
        token_i_full = promote_outer(token_i_inner)

        # Broadcast the weight matrices to match (1, N_i, 1) stream shape
        gate_i = expand_ref(gate_streams[i], token_i_full, expand_rank=2)
        up_i   = expand_ref(up_streams[i],   token_i_full, expand_rank=2)

        # Expert forward pass
        gate_out = binary_matmul(token_i_full, gate_i)                # (1, N_i, 1, F)
        up_out   = binary_matmul(token_i_full, up_i)                  # (1, N_i, 1, F)
        proj     = binary_mul(unary_silu(gate_out), up_out)           # (1, N_i, 1, F)

        # Broadcast down matrix and finish matmul
        down_i   = expand_ref(down_streams[i], proj, expand_rank=2)
        down_out = binary_matmul(proj, down_i)                         # (1, N_i, 1, D)

        # Collapse the leading outer singleton so shape[0] = token count (N_i)
        down_out_flat = flatten(down_out, min_rank=1, max_rank=2)     # (N_i, 1, 1, D)

        expert_outputs.append(down_out_flat)

    # ------------------------------------------------------------------
    # 5. Re‑assemble tokens preserving top‑k ordering
    # ------------------------------------------------------------------
    control_reassemble = select_gen(
        tensors["expert_onehot"], is_multihot=False, n=n_experts
    )
    merged = flat_reassemble(expert_outputs, control_reassemble)  # (1, B, n_active, 1, D)

    # ------------------------------------------------------------------
    # 6. Apply per‑token expert weights and sum across the active dimension
    # ------------------------------------------------------------------
    weight_stream = offchip_load(
        tensors["expert_weights"],
        stride=(n_active, 1),
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
    )
    weighted = binary_mul(merged, weight_stream)                 # (1, B, n_active, 1, D)
    result = accum_add(weighted, rank=1)                         # (1, B, 1, D)

    # ------------------------------------------------------------------
    # 7. Write result back off‑chip
    # ------------------------------------------------------------------
    return offchip_store(result)