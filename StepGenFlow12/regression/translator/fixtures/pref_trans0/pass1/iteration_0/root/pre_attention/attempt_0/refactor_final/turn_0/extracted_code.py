# pre_attention forwards the raw inputs to the two child blackboxes.
# 1. proj_and_norm computes Q, K, V from the input tensors.
#    Its `out_shapes` and `out_perms` are exactly the ones requested for the
#    final outputs of this node, so we forward them unchanged.
# 2. rope applies rotary positional embeddings to Q and K.
#    It only needs the target shapes for Q and K, i.e. the first two entries
#    of `out_shapes`.  The corresponding permutations (if any) are sliced
#    similarly.
# No on-chip DSL operations are required here; the children handle all
# loading and computation internally.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # Stage 1: QKV projection and per‑head RMSNorm
    Q, K, V = proj_and_norm(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    # Stage 2: RoPE on Q and K
    rope_out_shapes = (out_shapes[0], out_shapes[1])
    rope_out_perms = None if out_perms is None else (out_perms[0], out_perms[1])

    Q, K = rope(
        Q,
        K,
        cos,
        sin,
        out_shapes=rope_out_shapes,
        out_perms=rope_out_perms,
    )

    return Q, K, V