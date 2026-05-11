# MoE aggregation.
#   • For every routed expert we locate the (token, top‑k‑slot) pairs,
#     call the child `expert_compute`, weight the result with the routing
#     weight, and finally re‑assemble all expert contributions.
#   • All shape manipulations that touch on‑chip tensors use DSL ops
#     (`metadata_gen`, `flatten`, `binary_mul`, `flat_reassemble`,
#     `accum_add`).  Off‑chip tensors are only indexed on the host side
#     before any DSL consumer.
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

    per_expert_outs = []   # weighted expert outputs (streams)
    per_expert_masks = []  # control masks for flat_reassemble

    for e_idx in range(n_experts):
        # -----------------------------------------------------------------
        # Tokens routed to this expert (and the slot position within top‑k)
        # -----------------------------------------------------------------
        mask = expert_onehot[:, :, e_idx]            # (seq_len, top_k) int64
        tok, top = torch.where(mask == 1)            # vanilla 1‑D tensors

        if tok.numel() == 0:
            # No tokens for this expert – skip it entirely.
            continue

        # -----------------------------------------------------------------
        # Call the child blackbox.  Its output stream must have a first
        # dimension equal to the number of selected tokens.
        # -----------------------------------------------------------------
        out_shape = ((tok.numel(), 1, 512),)          # (N,1,dim) stream shape
        down_out = expert_compute(
            normed_2,
            tok,
            w_gate[e_idx],
            w_up[e_idx],
            w_down[e_idx],
            out_shapes=out_shape,
            out_perms=(None,),
        )                                               # (N,1,512)

        # -----------------------------------------------------------------
        # Gather the routing weight for each selected token and broadcast it.
        # -----------------------------------------------------------------
        weight = expert_weights[tok, top]              # (N,)
        weight_stream = metadata_gen(weight)           # (1, N, 1, 1)
        weight_stream = flatten(
            weight_stream, min_rank=0, max_rank=1
        )                                               # (N, 1, 1)

        # Apply the per‑token weight.
        weighted = binary_mul(down_out, weight_stream)  # (N,1,512)
        per_expert_outs.append(weighted)

        # Control mask for this expert (float version of the one‑hot slice).
        per_expert_masks.append(mask.float())

    # -----------------------------------------------------------------
    # If no expert contributed any tokens, return a zero tensor of the
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
    # yielding shape (seq_len, 1, dim) with an extra leading singleton.
    # -----------------------------------------------------------------
    summed = accum_add(merged, rank=2)          # (1, seq_len, 1, dim)

    # -----------------------------------------------------------------
    # Remove the leading singleton stream dimension.
    # -----------------------------------------------------------------
    out = flatten(summed, min_rank=0, max_rank=1)  # (seq_len, 1, dim)
    return out