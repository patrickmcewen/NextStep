# MoE aggregation using DSL ops.
#   - Token indices and routing weights are extracted from the raw
#     `expert_onehot` and `expert_weights` tensors with plain torch
#     operations (allowed on off‑chip inputs before any DSL consumer).
#   - Each active expert is processed with the child `expert_compute`
#     blackbox, which expects vanilla‑shaped arguments.
#   - The per‑token routing weight is broadcast via `metadata_gen` and
#     multiplied with the expert output (`binary_mul`).
#   - Contributions from all experts are reassembled according to the
#     routing mask using `flat_reassemble`, summed across the top‑k
#     dimension and the singleton "active‑expert" dimension (`accum_add`),
#     and finally the two stream dimensions are merged into the required
#     `(seq_len, 1, dim)` shape with `flatten`.
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
    # Collect per‑expert weighted outputs and the corresponding control
    # masks.  Only experts that actually receive tokens are kept.
    # -----------------------------------------------------------------
    per_expert_outs = []
    per_expert_masks = []

    n_experts = w_gate.shape[0]  # number of routed experts (8)

    for e_idx in range(n_experts):
        # routing mask for this expert: shape (seq_len, top_k)
        mask = expert_onehot[:, :, e_idx]          # int64 tensor

        # token indices (tok) and the position within the top‑k slot (top_pos)
        tok, top_pos = torch.where(mask == 1)
        if tok.numel() == 0:
            # this expert receives no tokens – skip it completely
            continue

        # -----------------------------------------------------------------
        # Call the child expert_compute on the selected tokens.
        # The child expects vanilla shapes; we provide the appropriate
        # out_shapes tuple that matches the number of tokens.
        # -----------------------------------------------------------------
        exp_out_shape = ((tok.numel(), 1, 512),)   # (N, 1, dim) stream shape
        down_out = expert_compute(
            normed_2,
            tok,
            w_gate[e_idx],
            w_up[e_idx],
            w_down[e_idx],
            out_shapes=exp_out_shape,
            out_perms=(None,),
        )  # stream of shape (N, 1, 512)

        # -----------------------------------------------------------------
        # Apply the per‑token routing weight: expert_weights[tok, top_pos]
        # -----------------------------------------------------------------
        weight = expert_weights[tok, top_pos]      # vanilla shape (N,)
        # Convert the 1‑D weight vector to a stream (N, 1, 1)
        weight_stream = metadata_gen(weight)       # (1, N, 1, 1)
        # Merge the leading singleton stream dimension with the token
        # dimension so that it matches `down_out`'s stream shape.
        weight_stream = flatten(weight_stream, min_rank=0, max_rank=1)  # (N, 1, 1)

        # Broadcast‑multiply the expert output by the routing weight.
        weighted = binary_mul(down_out, weight_stream)  # (N, 1, 512)

        per_expert_outs.append(weighted)
        per_expert_masks.append(mask)

    # If no expert was active, the output is simply a zero tensor.
    if not per_expert_outs:
        return unary_to_const_int(normed_2, 0.0)

    # -----------------------------------------------------------------
    # Build the control tensor for flat_reassemble.
    # It has shape (seq_len, top_k, num_active_experts) and is
    # interpreted as a mask (>0) that selects rows from the expert
    # output streams.
    # -----------------------------------------------------------------
    control = torch.stack(per_expert_masks, dim=-1).float()  # (S, K, A)

    # Reassemble the per‑expert streams into a token‑major stream.
    # Result shape: (seq_len, top_k, 1, 1, dim)
    merged = flat_reassemble(per_expert_outs, control)

    # Sum over the singleton "active‑expert" dimension and over top_k.
    # After two reductions we get shape (seq_len, 1, dim).
    summed = accum_add(merged, rank=2)

    # Collapse the remaining two stream dimensions (seq_len, 1) into a
    # single stream dimension, yielding the required output shape.
    out = flatten(summed, min_rank=0, max_rank=1)  # (seq_len, 1, dim)

    return out