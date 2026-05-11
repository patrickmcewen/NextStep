# MoE aggregation with per‑expert capacity (seq_len // n_experts = 8).
#   • For each routed expert we select at most `capacity` tokens,
#     picking the highest routing weights (so we keep the most important
#     tokens).  The selected token indices and their top‑k slot indices
#     are used to call the child `expert_compute`, which expects exactly
#     `capacity` tokens (the call is padded with zeros and a zero weight
#     for any missing slots).
#   • The expert's raw output is multiplied by the per‑token gating weight.
#   • A per‑expert binary mask (1 → contribution, 0 → no contribution) is
#     built that reflects only the selected token/slot pairs.  The masks
#     are stacked and fed to `flat_reassemble`, which gathers the per‑expert
#     streams into a token‑major stream.
#   • Finally we sum over the two stream dimensions (top‑k and active‑expert)
#     with `accum_add` and collapse the leading singleton with `flatten`,
#     yielding a stream of shape (seq_len, 1, dim) as required.
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
    seq_len = normed_2.shape[0]          # 64
    top_k   = expert_onehot.shape[1]    # 2
    n_experts = w_gate.shape[0]         # 8
    capacity = seq_len // n_experts      # 8 tokens per expert (fixed)

    per_expert_outs = []    # weighted expert outputs (streams)
    per_expert_masks = []   # binary masks for flat_reassemble

    for e_idx in range(n_experts):
        # -----------------------------------------------------------------
        # Binary mask for this expert: shape (seq_len, top_k), int64.
        # -----------------------------------------------------------------
        mask = expert_onehot[:, :, e_idx]               # (seq_len, top_k)
        tok_all, top_all = torch.where(mask == 1)       # vanilla 1‑D tensors

        if tok_all.numel() == 0:
            # No tokens routed to this expert – skip it entirely.
            continue

        # -----------------------------------------------------------------
        # Compute routing weights for all candidate (token, top) pairs.
        # -----------------------------------------------------------------
        all_weights = expert_weights[tok_all, top_all]   # (num_candidates,)

        # Keep the highest‑weight `capacity` pairs, preserving the original
        # order (torch.where gives lexicographic order).  This matches the
        # typical MoE capacity policy.
        if tok_all.numel() > capacity:
            _, topk_idx = torch.topk(all_weights, capacity, largest=True, sorted=False)
            selected_mask = torch.zeros_like(all_weights, dtype=torch.bool)
            selected_mask[topk_idx] = True
            tok = tok_all[selected_mask]
            top = top_all[selected_mask]
            weight = all_weights[selected_mask]
        else:
            tok = tok_all
            top = top_all
            weight = all_weights

        selected = tok.shape[0]               # number of real tokens kept
        pad_len = capacity - selected

        if pad_len > 0:
            # Pad token / top indices with zeros (any valid index works).
            tok = torch.cat([tok, torch.zeros(pad_len, dtype=tok.dtype, device=tok.device)])
            top = torch.cat([top, torch.zeros(pad_len, dtype=top.dtype, device=top.device)])
            weight = torch.cat([weight, torch.zeros(pad_len, dtype=weight.dtype, device=weight.device)])

        # -----------------------------------------------------------------
        # Call the child expert_compute (expects exactly `capacity` tokens).
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
        # Broadcast the per‑token routing weight and apply it.
        # -----------------------------------------------------------------
        weight_stream = metadata_gen(weight)            # (1, capacity, 1, 1)
        weight_stream = flatten(
            weight_stream, min_rank=0, max_rank=1
        )                                                # (capacity, 1, 1)

        weighted = binary_mul(down_out, weight_stream)  # (capacity,1,512)
        per_expert_outs.append(weighted)

        # -----------------------------------------------------------------
        # Build a float mask that has 1.0 exactly at the selected positions.
        # (Unselected positions stay 0.0; padded slots are also 0.0.)
        # -----------------------------------------------------------------
        mask_float = torch.zeros_like(mask, dtype=torch.float32)
        if selected > 0:
            mask_float[tok[:selected], top[:selected]] = 1.0
        per_expert_masks.append(mask_float)

    # -----------------------------------------------------------------
    # If no expert contributed anything, return a zero tensor of the
    # required shape (seq_len, 1, dim).
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
    # yielding a stream of shape (1, seq_len)×tile(1,512).
    # -----------------------------------------------------------------
    summed = accum_add(merged, rank=2)          # (1, seq_len)×tile(1,512)

    # -----------------------------------------------------------------
    # Remove the leading singleton stream dimension to match the
    # contract shape (seq_len, 1, dim).
    # -----------------------------------------------------------------
    out = flatten(summed, min_rank=0, max_rank=1)   # (seq_len, 1, dim)
    return out