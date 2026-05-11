# Implementation notes:
# This node simply forwards its inputs to the two child blackboxes that
# implement the full pre‑attention computation.  All inputs are RAW
# (off‑chip), but blackboxes are allowed to receive RAW tensors directly;
# they internally handle any required loading.  No further DSL operations
# are needed here, so we just call the children with the requested
# `out_shapes`/`out_perms` and return the final Q, K, V streams.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # First stage: RMSNorm + Q/K/V projection
    Q, K, V = pre_attn_norm_and_proj(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # Second stage: per‑head RMSNorm and RoPE
    Q, K, V = per_head_norm_and_rope(
        Q,
        K,
        V,
        cos,
        sin,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return Q, K, V