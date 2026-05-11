# MoE aggregation.
#   • For each routed expert we locate the (token, top‑k) pairs,
#     enforce a fixed capacity per expert (8 tokens), call the child
#     `expert_compute`, weight the result with the routing weight,
#     and finally re‑assemble all contributions.
#   • All on‑chip shape manipulations use DSL ops; off‑chip tensors are
#     only indexed on the host side before any DSL consumer.
def moe_compute(
    normed_2,
    w_gate,
    w_up,
    w_down,
    expert_weights,
    expert_onehot,
    *,
    out_shapes,
    out_perms=None,
):
    # -----------------------------------------------------------------
    # Derive the planner‑imposed capacity (tokens per routed expert).
    # For the given configuration 64 seq_len / 8 routed_experts = 8.
    # -----------------------------------------------------------------
    seq_len = normed_2.shape[0]          # 64
    top_k   = expert_onehot.shape[1]    # 2
    n_experts = w_gate.shape[0]         # 8
    capacity = seq_len // n_experts      # 8 (fixed per‑expert capacity)

    per_expert_outs = []   # weighted expert outputs (streams)
    per_expert_masks = []  # control masks for flat_reassemble

    for e_idx in range(n_experts):
        # -----------------------------------------------------------------
        # Tokens routed to this expert (and the slot position within top‑k).
        # -----------------------------------------------------------------
        mask = expert_onehot[:, :, e_idx]               # (seq_len, top_k) int64
        tok_all, top_all = torch.where(mask == 1)       # vanilla 1‑D tensors

        if tok_all.numel() == 0:
            # No tokens for this expert – skip it.
            continue

        # -----------------------------------------------------------------
        # Enforce the fixed capacity: truncate if there are too many tokens,
        # pad with dummy indices (and later zero‑out their weight) if there are
        # too few.
        # -----------------------------------------------------------------
        selected = min(tok_all.numel(), capacity)
        tok = tok_all[:selected]
        top = top_all[:selected]

        pad_len = capacity - selected
        if pad_len > 0:
            # Pad with zeros (any valid index works; weight will be zeroed).
            tok = torch.cat(
                [tok, torch.zeros(pad_len, dtype=tok.dtype, device=tok.device)]
            )
            top = torch.cat(
                [top, torch.zeros(pad_len, dtype=top.dtype, device=top.device)]
            )

        # -----------------------------------------------------------------
        # Call the child blackbox.  Its output stream must have exactly
        # `capacity` rows (the planner’s fixed size).
        # -----------------------------------------------------------------
        out_shape = ((capacity, 1, 512),)               # (N,1,dim) stream shape
        down_out = expert_compute(
            normed_2,
            tok,
            w_gate[e_idx],
            w_up[e_idx],
            w_down[e_idx],
            out_shapes=out_shape,
            out_perms=(None,),
        )                                                # (capacity,1,512)

        # -----------------------------------------------------------------
        # Routing weight for each token; zero out the padded entries.
        # -----------------------------------------------------------------
        weight = expert_weights[tok, top]               # (capacity,)
        if pad_len > 0:
            weight[-pad_len:] = 0.0

        # Broadcast the weight to match `down_out`'s stream shape.
        weight_stream = metadata_gen(weight)            # (1, capacity, 1, 1)
        weight_stream = flatten(
            weight_stream, min_rank=0, max_rank=1
        )                                                # (capacity,1,1)

        # Apply the per‑token weight.
        weighted = binary_mul(down_out, weight_stream)  # (capacity,1,512)
        per_expert_outs.append(weighted)

        # Control mask for this expert (float version of the one‑hot slice).
        per_expert_masks.append(mask.float())

    # -----------------------------------------------------------------
    # If no expert emitted any tokens, return a zero tensor of the required
    # shape (seq_len, 1, dim).
    # -----------------------------------------------------------------
    if not per_expert_outs:
        return unary_to_const_int(normed_2, 0.0)

    # -----------------------------------------------------------------
    # Re‑assemble the per‑expert streams according to the routing mask.
    # control shape: (seq_len, top_k, n_active_experts)
    # -----------------------------------------------------------------
    control = torch.stack(per_expert_masks, dim=-1)
    merged = flat_reassemble(per_expert_outs, control)

    # -----------------------------------------------------------------
    # Sum over the two stream dimensions (active‑expert and top‑k),
    # producing shape (1, seq_len, 1, dim) (with a leading singleton).
    # -----------------------------------------------------------------
    summed = accum_add(merged, rank=2)          # (1, seq_len, 1, dim)

    # -----------------------------------------------------------------
    # Remove the leading singleton stream dimension.
    # -----------------------------------------------------------------
    out = flatten(summed, min_rank=0, max_rank=1)   # (seq_len, 1, dim)
    return out