# The pre_attention node simply composes the two child blackboxes.
# 1. proj_and_norm computes Q, K, V from the raw inputs.
# 2. rope applies rotary positional embeddings to Q and K.
# Both children handle any necessary off‑chip loads internally, so we
# just forward the raw tensors unchanged.  We slice the `out_shapes`
# and `out_perms` tuples to match each child’s output count.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # proj_and_norm produces three outputs: Q, K, V
    Q, K, V = proj_and_norm(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    # rope produces two outputs: updated Q and K.
    # Use the first two shapes (and perms, if provided) from the node's contract.
    rope_out_shapes = (out_shapes[0], out_shapes[1])
    rope_out_perms = None
    if out_perms is not None:
        rope_out_perms = (out_perms[0], out_perms[1])

    Q, K = rope(
        Q,
        K,
        cos,
        sin,
        out_shapes=rope_out_shapes,
        out_perms=rope_out_perms,
    )

    return Q, K, V