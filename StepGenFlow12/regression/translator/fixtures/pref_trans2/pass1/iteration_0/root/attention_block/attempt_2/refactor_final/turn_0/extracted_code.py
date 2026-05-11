# The attention_block composes the two child blackboxes:
#   1. `qkv_preprocess` produces tiled streams for Q, K, V.
#   2. `attention_o_proj` consumes those streams plus the output weight
#      to produce the final hidden representation.
# Raw weight tensors are handed directly to the children (they perform
# their own off‑chip loads).  No additional DSL ops are needed because each
# child already returns streams in the expected layout.  We only need to
# supply explicit stream shapes for the Q, K, V tensors so that the next
# child can consume them.  Finally we forward the parent‑specified
# `out_shapes`/`out_perms` to `attention_o_proj` and return its result.
def attention_block(normed, q_proj, k_proj, v_proj, cos, sin, o_proj_weight, *, out_shapes, out_perms=None):
    # Stream shapes for the three outputs of qkv_preprocess.
    #   Q: (batch=64, q_heads=16, head_dim=32) -> stream (1,64,16) tile (1,32)
    #   K/V: (batch=64, kv_heads=4, head_dim=32) -> stream (1,64,4) tile (1,32)
    Q_shape = (1, 64, 16, 1, 32)
    K_shape = (1, 64, 4, 1, 32)
    V_shape = (1, 64, 4, 1, 32)

    # Run QKV preprocessing.
    Q, K, V = qkv_preprocess(
        normed,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=(Q_shape, K_shape, V_shape),
        out_perms=(None, None, None),
    )

    # Compute the final O projection.
    out = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return out