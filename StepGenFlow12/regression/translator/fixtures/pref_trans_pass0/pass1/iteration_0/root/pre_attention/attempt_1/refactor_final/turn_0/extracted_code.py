# The pre‑attention node is a thin orchestrator:
#   1. Apply RMSNorm and the Q/K/V projection (child `pre_attn_norm_and_proj`).
#   2. Apply per‑head RMSNorm and RoPE (child `per_head_norm_and_rope`).
# Both children already know how to load RAW off‑chip tensors, so we pass the
# inputs directly to them without any tensor‑method transforms.  The output
# shapes required by the parent are exactly the shapes produced by the
# children, so we simply forward the `out_shapes` / `out_perms` arguments.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # 1️⃣ RMSNorm + linear projection for Q, K, V.
    Q, K, V = pre_attn_norm_and_proj(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    # 2️⃣ Per‑head RMSNorm and RoPE (uses the same cos/sin as above).
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