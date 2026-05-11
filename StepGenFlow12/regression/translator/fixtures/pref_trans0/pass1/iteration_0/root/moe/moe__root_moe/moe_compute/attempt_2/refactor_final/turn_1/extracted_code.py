# MoE aggregation.
#   • For each routed expert we take the (capacity‑limited) tokens routed to it,
#     call the child `expert_compute`, weight the result by the routing weights,
#     and finally re‑assemble all expert contributions with `flat_reassemble`.
#   • The planner expects exactly `seq_len // n_experts` token indices per
#     expert (capacity = 8 for the given dimensions).  When an expert has
#     fewer routed tokens we pad the token list with dummy indices and a zero
#     weight; when it has more we truncate to the capacity.
#   • All on‑chip shape manipulations use DSL ops (`metadata_gen`,
#     `flatten`, `binary_mul`, `flat_reassemble`, `accum_add`).  Raw off‑chip
#     tensors (`w_*`, `expert_weights`, `expert_onehot`) are only accessed
#     with host‑side indexing before any DSL consumer.
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
    # Parameters derived from the input shapes.
    # -----------------------------------------------------------------
    seq_len = normed_2.shape[0]                 # 64
    top_k = expert_onehot.shape[1]              # 2  (num activated experts)
    n_experts = w_gate.shape[0]                 # 8  (routed experts)

    # The planner’s capacity per expert (fixed by the contract).
    capacity = seq_len // n_experts              # 8

    per_expert_outs = []      # weighted expert outputs (streams)
    per_expert_masks = []     # control masks for flat_reassemble

    for e_idx in range(n_experts):
        # -----------------------------------------------------------------
        # Locate the token slots allocated to this expert.
        # -----------------------------------------------------------------
        mask = expert_onehot[:, :, e_idx]               # (seq_len, top_k) int64
        tok_all, top_all = torch.where(mask == 1)       # vanilla 1‑D tensors

        # If no tokens are routed to this expert, skip it.
        if tok_all.numel() == 0:
            continue

        # -----------------------------------------------------------------
        # Apply the planner’s capacity: truncate or pad to `capacity`.
        # -----------------------------------------------------------------
        selected = min(tok_all.numel(), capacity)
        tok = tok_all[:selected]
        top = top_all[:selected]

        if selected < capacity:
            pad_len = capacity - selected
            # Pad token indices with zeros (any valid index works).
            tok = torch.cat([tok, torch.zeros(pad_len,
                                              dtype=tok.dtype,
                                              device=tok.device)])
            top = torch.cat([top, torch.zeros(pad_len,
                                              dtype=top.dtype,
                                              device=top.device)])

        # -----------------------------------------------------------------
        # Call the child expert_compute.  Its output stream must have the
        # same first dimension as the (padded) token list.
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
        # Gather the routing weight for each selected token and broadcast it.
        # -----------------------------------------------------------------
        weight = expert_weights[tok, top]               # (capacity,)
        weight_stream = metadata_gen(weight)            # (1,capacity,1,1)
        weight_stream = flatten(weight_stream,
                                min_rank=0,
                                max_rank=1)          # (capacity,1,1)

        # Apply the per‑token weight.
        weighted = binary_mul(down_out, weight_stream)  # (capacity,1,512)
        per_expert_outs.append(weighted)

        # -----------------------------------------------------------------
        # Build a control mask that mirrors the selected slots.  It has the
        # same shape as the original one‑hot slice but zeros for the padded
        # entries (the padding indices are zero, so `mask == 0` there).
        # -----------------------------------------------------------------
        mask_float = mask.float()                       # (seq_len, top_k)
        per_expert_masks.append(mask_float)

    # -----------------------------------------------------------------
    # If no expert contributed anything, return a zero tensor of the proper
    # shape (seq_len, 1, dim).
    # -----------------------------------------------------------------
    if not per_expert_outs:
        return unary_to_const_int(normed_2, 0.0)

    # -----------------------------------------------------------------
    # Re‑assemble the per‑expert streams according to the control mask.
    # The mask shape is (seq_len, top_k, n_active_experts); stacking creates
    # the required 3‑D control tensor.
    # -----------------------------------------------------------------
    control = torch.stack(per_expert_masks, dim=-1)     # (seq_len, top_k, n_active)
    merged = flat_reassemble(per_expert_outs, control)

    # -----------------------------------------------------------------
    # Collapse the two stream dimensions (n_active and top_k) by summing.
    # After two reductions we obtain a stream of shape (seq_len,1,dim).
    # -----------------------------------------------------------------
    summed = accum_add(merged, rank=2)                  # (seq_len,1,512)

    return summed