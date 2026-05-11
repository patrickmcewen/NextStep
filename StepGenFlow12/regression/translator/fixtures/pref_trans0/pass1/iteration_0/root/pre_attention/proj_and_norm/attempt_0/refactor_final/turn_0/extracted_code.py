# The node simply forwards its raw inputs to the two child blackboxes.
# 1) `pre_attention_and_qkv` computes Q, K, V from the raw tensors.  The
#    parent’s `out_shapes` already describe the desired stream shapes for
#    these three tensors, so we pass them unchanged.
# 2) `per_head_norm` applies RMSNorm to Q and K.  Its outputs have the
#    same shapes as the first two entries of `out_shapes`; we therefore
#    pass `out_shapes[:2]` (and the matching slice of `out_perms` if present).
# No tensor‑method calls or explicit loads are needed because the blackboxes
# accept raw off‑chip tensors directly and handle shape reconstruction
# internally.
def proj_and_norm(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    # Step 1: Q, K, V projection and RMSNorm (pre‑attention)
    Q, K, V = pre_attention_and_qkv(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    # Step 2: Per‑head RMSNorm on Q and K
    Q_norm, K_norm = per_head_norm(
        Q,
        K,
        out_shapes=out_shapes[:2],
        out_perms=out_perms[:2] if out_perms is not None else None,
    )

    # Return the normalized Q, normalized K, and untouched V
    return Q_norm, K_norm, V