# The pre_attention node performs two logical stages:
#   1. Project the input tensor into Q, K, V and apply RMSNorm (handled by
#      the `proj_and_norm` child). This stage produces three tensors with
#      vanilla shapes (64,16,32), (64,4,32), (64,4,32).
#   2. Apply Rotary Positional Embedding (RoPE) to Q and K (handled by the
#      `rope` child). It consumes the Q and K from the previous stage and
#      the cos/sin tensors, returning the rotated Q and K while keeping V
#      unchanged.
#
# All raw off‑chip inputs can be passed directly to the children – the
# child stubs will internally load them as needed. We only need to forward
# the requested output shapes (and any output permutations) to each child.
# No on‑chip tensor arithmetic or reshaping is required here.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ---------------------------------------------------------------
    # Stage 1: QKV projection + RMSNorm via child `proj_and_norm`.
    # ---------------------------------------------------------------
    # `proj_and_norm` expects three output shapes (Q, K, V). We forward the
    # shapes requested for this node. If the caller supplied output
    # permutations, forward the corresponding entries; otherwise rely on the
    # child's default (identity).
    if out_perms is None:
        proj_out_perms = None
    else:
        # The first three entries correspond to Q, K, V.
        proj_out_perms = (out_perms[0], out_perms[1], out_perms[2])

    Q, K, V = proj_and_norm(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        out_shapes=(out_shapes[0], out_shapes[1], out_shapes[2]),
        out_perms=proj_out_perms,
    )

    # ---------------------------------------------------------------
    # Stage 2: RoPE on Q and K via child `rope`.
    # ---------------------------------------------------------------
    # `rope` returns the rotated Q and K. It needs two output shapes (for
    # Q and K). We reuse the first two permutations if any were supplied.
    if out_perms is None:
        rope_out_perms = None
    else:
        rope_out_perms = (out_perms[0], out_perms[1])

    Q_rot, K_rot = rope(
        Q,
        K,
        cos,
        sin,
        out_shapes=(out_shapes[0], out_shapes[1]),
        out_perms=rope_out_perms,
    )

    # Return the final Q, K (rotated) and V tensors.
    return Q_rot, K_rot, V