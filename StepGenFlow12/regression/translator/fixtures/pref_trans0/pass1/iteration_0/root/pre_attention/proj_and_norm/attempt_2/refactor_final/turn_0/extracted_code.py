# Implementation reasoning:
# This node orchestrates two child blackboxes:
#   1) `pre_attention_and_qkv` produces Q, K, V from the raw inputs.
#   2) `per_head_norm` normalises Q and K, yielding Q_norm and K_norm.
# The raw tensors can be passed directly to the children (the children
# handle any necessary off‑chip loads).  We simply forward the requested
# output shapes (and permutations, when supplied) to the children,
# slice the permutations for the second child, and return the final
# tuple `(Q_norm, K_norm, V)`.  No tensor‑method calls are used.
def proj_and_norm(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    # ---------------------------------------------------------
    # 1️⃣ Pre‑attention QKV projection (child 1)
    # ---------------------------------------------------------
    if out_perms is None:
        Q, K, V = pre_attention_and_qkv(
            input_tensor,
            q_proj,
            k_proj,
            v_proj,
            out_shapes=out_shapes,
        )
    else:
        Q, K, V = pre_attention_and_qkv(
            input_tensor,
            q_proj,
            k_proj,
            v_proj,
            out_shapes=out_shapes,
            out_perms=out_perms,
        )

    # ---------------------------------------------------------
    # 2️⃣ Per‑head RMSNorm on Q and K (child 2)
    # ---------------------------------------------------------
    head_norm_out_shapes = (out_shapes[0], out_shapes[1])

    if out_perms is None:
        Q_norm, K_norm = per_head_norm(
            Q,
            K,
            out_shapes=head_norm_out_shapes,
        )
    else:
        head_norm_out_perms = (out_perms[0], out_perms[1])
        Q_norm, K_norm = per_head_norm(
            Q,
            K,
            out_shapes=head_norm_out_shapes,
            out_perms=head_norm_out_perms,
        )

    return Q_norm, K_norm, V