# The pre_attention node simply threads the inputs through the two
# black‑box sub‑models that implement the RMSNorm + projection and the
# per‑head RMSNorm + RoPE steps.  All tensors are raw off‑chip values, and
# the black boxes are responsible for loading them as needed.  No DSL
# operations are required here; we just forward the `out_shapes` and
# `out_perms` arguments to each child so the final streams have the
# expected shapes.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # 1️⃣ RMSNorm + QKV projection
    Q0, K0, V0 = pre_attn_norm_and_proj(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # 2️⃣ Per‑head RMSNorm and RoPE
    Q, K, V = per_head_norm_and_rope(
        Q0,
        K0,
        V0,
        cos,
        sin,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return Q, K, V