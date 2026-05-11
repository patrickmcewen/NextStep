# The pre‑attention node simply wires together the two child blackboxes.
# All inputs are RAW (off‑chip), so we must **not** apply any DSL consumer
# (binary_*, unary_*, …) before the first child.  The blackbox stubs handle
# any necessary off‑chip loads internally, so we can pass the raw tensors
# directly.  The expected output shapes of both children are the same as the
# node’s own output shapes, therefore we forward `out_shapes` (and the
# optional `out_perms`) unchanged to each call.
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # 1) RMS‑norm and QKV projection.
    Q_pre, K_pre, V_pre = pre_attn_norm_and_proj(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    # 2) Per‑head RMS‑norm and RoPE.
    Q, K, V = per_head_norm_and_rope(
        Q_pre,
        K_pre,
        V_pre,
        cos,
        sin,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return Q, K, V