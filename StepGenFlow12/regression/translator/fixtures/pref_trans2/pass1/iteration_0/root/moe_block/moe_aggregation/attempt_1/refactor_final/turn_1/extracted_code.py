# This implementation follows the MoE aggregation algorithm using the DSL.
# It loads the raw weight tensors, duplicates each token for the two
# activated experts, partitions tokens (and their routing weights) per
# expert, chunks each expert's token stream into fixed‑size blocks of 8
# (the size expected by the `expert_contribute` blackbox), calls the
# blackbox on each block, merges the block results back into a per‑expert
# stream, and finally re‑assembles and sums the contributions to produce
# the required output stream shape (1, 64, 1, 512).
def moe_aggregation(normed_2, w_gate, w_up, w_down,
                    expert_weights, expert_onehot, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 1) Load the three expert weight tensors from off‑chip memory.
    # ------------------------------------------------------------------
    w_gate_stream = offchip_load(
        w_gate, stride=(1,), out_shape_tiled=(8,), tile_row=512, tile_col=1792
    )
    w_up_stream = offchip_load(
        w_up, stride=(1,), out_shape_tiled=(8,), tile_row=512, tile_col=1792
    )
    w_down_stream = offchip_load(
        w_down, stride=(1,), out_shape_tiled=(8,), tile_row=1792, tile_col=512
    )

    # Collapse the leading singleton, yielding pure per‑expert streams.
    w_gate_flat = flatten(w_gate_stream, min_rank=0, max_rank=1)   # (8,512,1792)
    w_up_flat   = flatten(w_up_stream,   min_rank=0, max_rank=1)   # (8,512,1792)
    w_down_flat = flatten(w_down_stream, min_rank=0, max_rank=1)   # (8,1792,512)

    # Split each weight tensor into a list of per‑expert tensors.
    w_gate_list = parallelize(w_gate_flat, n=8)   # each (1,512,1792)
    w_up_list   = parallelize(w_up_flat,   n=8)   # each (1,512,1792)
    w_down_list = parallelize(w_down_flat, n=8)   # each (1,1792,512)

    # ------------------------------------------------------------------
    # 2) Duplicate every token for the two activated experts.
    # ------------------------------------------------------------------
    normed_rep = repeat_static(normed_2, factor=2)                     # (1,64,2,1,512)
    normed_flat = flatten(normed_rep, min_rank=0, max_rank=2)          # (128,1,512)
    token_stream = reshape_stream(normed_flat, chunk_size=2, rank=0)   # (64,2,1,512)

    # ------------------------------------------------------------------
    # 3) Produce the (multihot) routing mask as a DSL stream.
    # ------------------------------------------------------------------
    control_onehot = select_gen(expert_onehot, is_multihot=True, n=8)   # (1,64,2,8)

    # ------------------------------------------------------------------
    # 4) Partition tokens and routing weights per expert according to the mask.
    # ------------------------------------------------------------------
    token_parts = flat_partition(token_stream, control_onehot, n=8)     # list of 8 tensors, (k_i,1,512)
    routing_stream = metadata_gen(expert_weights)                       # (1,64,2,1,1)
    routing_parts = flat_partition(routing_stream, control_onehot, n=8) # list of 8 tensors, (k_i,1,1)

    # ------------------------------------------------------------------
    # 5) For each expert, further chunk its token stream into blocks of 8,
    #    invoke the blackbox on each block, and merge the block results.
    # ------------------------------------------------------------------
    contribs = []
    for i in range(8):
        # Per‑expert token and routing streams.
        tok_i = token_parts[i]          # (k_i,1,512)
        rout_i = routing_parts[i]       # (k_i,1,1)

        # Chunk into groups of 8 (padding if necessary).
        tok_i_chunks = reshape_stream(tok_i, chunk_size=8, rank=0)   # (C,8,1,512)
        rout_i_chunks = reshape_stream(rout_i, chunk_size=8, rank=0) # (C,8,1,1)

        n_chunks = tok_i_chunks.shape[0]

        if n_chunks == 0:
            # No tokens for this expert – contribution is an empty tensor.
            merged_flat = tok_i
        else:
            # Split the chunked tensors into a list (one per chunk).
            tok_chunks = parallelize(tok_i_chunks, n=n_chunks)      # each (1,8,1,512)
            rout_chunks = parallelize(rout_i_chunks, n=n_chunks)    # each (1,8,1,1)

            # Call the blackbox on every chunk.
            chunk_contribs = []
            for j in range(n_chunks):
                c = expert_contribute(
                    tok_chunks[j],
                    w_gate_list[i],
                    w_up_list[i],
                    w_down_list[i],
                    rout_chunks[j],
                    out_shapes=(tok_chunks[j].shape,),
                    out_perms=(None,),
                )
                chunk_contribs.append(c)

            # Merge the per‑chunk contributions back into a single stream.
            merged, _ = eager_merge(chunk_contribs)                 # (C,8,1,512)

            # Collapse the two stream dimensions (C and 8) into one.
            merged_flat = flatten(merged, min_rank=0, max_rank=1)    # (C*8,1,512)

        contribs.append(merged_flat)

    # ------------------------------------------------------------------
    # 6) Re‑assemble per‑expert contributions to the original token order
    #    and sum over the two active‑expert slots.
    # ------------------------------------------------------------------
    reassembled = flat_reassemble(contribs, control_onehot)   # (1,64,2,n_active,1,512)
    final = accum_add(reassembled, rank=2)                    # (1,64,1,512)

    return final