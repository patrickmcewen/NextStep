# The MoE aggregation works by (1) loading the per‑expert weight matrices from
# off‑chip memory, (2) duplicating each token representation for the two
# activation slots, (3) using the one‑hot routing mask to partition the token
# stream (and the corresponding routing‑weight stream) per expert via
# `flat_partition`, (4) invoking the `expert_contribute` blackbox for each
# expert with its own token slice, weight slice (obtained by parallelising the
# loaded weight streams), and routing‑weight slice, (5) re‑assembling the
# per‑expert results back into the original token‑slot ordering with
# `flat_reassemble`, and finally (6) summing over the “active‑expert” dimension
# and the activation‑slot dimension using `accum_add`.  The result has the
# required tile‑stream shape `(1, 64, 1, 512)`.
def moe_aggregation(normed_2, w_gate, w_up, w_down, expert_weights,
                    expert_onehot, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 1) Load the three expert weight tensors from off‑chip memory.
    #    Each weight has shape (expert, hidden, inter) or (expert, inter, hidden).
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

    # Collapse the leading singleton so the expert dimension becomes a pure stream
    w_gate_flat = flatten(w_gate_stream, min_rank=0, max_rank=1)   # (8,512,1792)
    w_up_flat   = flatten(w_up_stream,   min_rank=0, max_rank=1)
    w_down_flat = flatten(w_down_stream, min_rank=0, max_rank=1)

    # Split each weight matrix into a list of per‑expert tensors
    w_gate_list = parallelize(w_gate_flat, n=8)   # each: (1,512,1792)
    w_up_list   = parallelize(w_up_flat,   n=8)   # each: (1,512,1792)
    w_down_list = parallelize(w_down_flat, n=8)   # each: (1,1792,512)

    # ------------------------------------------------------------------
    # 2) Duplicate every token for the two activated experts (n_activated=2)
    # ------------------------------------------------------------------
    normed_rep = repeat_static(normed_2, factor=2)                # (1,2,64,1,512)
    normed_flat = flatten(normed_rep, min_rank=0, max_rank=2)    # (128,1,512)
    token_stream = reshape_stream(normed_flat, chunk_size=2, rank=0)  # (64,2,1,512)

    # ------------------------------------------------------------------
    # 3) Prepare the one‑hot routing mask as a DSL stream (no extra tile dims)
    # ------------------------------------------------------------------
    control_onehot = select_gen(expert_onehot, is_multihot=True, n=8)  # (1,64,2,8)

    # ------------------------------------------------------------------
    # 4) Partition tokens and routing weights per expert according to the mask
    # ------------------------------------------------------------------
    token_parts = flat_partition(token_stream, control_onehot, n=8)   # list of 8, each (k_i,1,512)
    routing_stream = metadata_gen(expert_weights)                     # (1,64,2,1,1)
    routing_parts = flat_partition(routing_stream, control_onehot, n=8)  # list of 8, each (k_i,1,1)

    # ------------------------------------------------------------------
    # 5) For each expert compute its contribution via the blackbox
    # ------------------------------------------------------------------
    contribs = []
    for i in range(8):
        contrib_i = expert_contribute(
            token_parts[i],
            w_gate_list[i],
            w_up_list[i],
            w_down_list[i],
            routing_parts[i],
            out_shapes=(token_parts[i].shape,),
            out_perms=(None,),
        )
        contribs.append(contrib_i)

    # ------------------------------------------------------------------
    # 6) Re‑assemble per‑expert contributions back to the original (token,slot)
    #    ordering and sum over the active‑expert and slot dimensions.
    # ------------------------------------------------------------------
    reassembled = flat_reassemble(contribs, control_onehot)   # (1,64,2,n_active,1,512)
    final = accum_add(reassembled, rank=2)                   # (1,64,1,512)

    return final