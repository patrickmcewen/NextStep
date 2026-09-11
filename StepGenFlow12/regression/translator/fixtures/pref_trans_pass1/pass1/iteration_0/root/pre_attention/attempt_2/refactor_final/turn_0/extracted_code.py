# Implementation Reasoning:
# The pre‑attention node consists of two sequential blackbox calls:
#   1. `pre_attn_norm_and_proj` performs RMSNorm on the input tensor and
#      projects it to Q, K, V. It already produces the correct stream shapes
#      ((64,16,32), (64,4,32), (64,4,32)).
#   2. `per_head_norm_and_rope` applies per‑head RMSNorm and RoPE to those
#      Q/K/V streams. Its output shapes match the inputs, so we can forward
#      the same `out_shapes`/`out_perms` we received from the parent.
#
# All inputs are RAW off‑chip tensors, but blackboxes may accept raw tensors
# directly (they internally handle any required `offchip_load`).  Therefore
# we simply pass the arguments through without any tensor‑method transformations,
# respecting the “no transform before blackbox” rule.
#
# The function forwards `out_shapes` (a 3‑tuple of stream shapes) and
# `out_perms` (a parallel tuple of optional permutations) to each child.
# The final Q, K, V streams are returned to the caller.

def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # First stage: RMSNorm + projection to Q, K, V.
    Q, K, V = pre_attn_norm_and_proj(
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    # Second stage: per‑head RMSNorm and RoPE.
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