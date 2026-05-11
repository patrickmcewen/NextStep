# The attention block consists of two stages implemented as blackboxes:
# 1) `qkv_preprocess` creates the Q, K, V streams from the normalized
#    activation and the projection matrices.  We request streams that
#    expose the head dimension as a stream axis (batch=1, seq_len=64,
#    heads) and keep the head‑dim (32) as the tile width with a tile
#    height of 1.  This yields stream shapes:
#        Q: (1, 64, 16, 1, 32)
#        K: (1, 64,  4, 1, 32)
#        V: (1, 64,  4, 1, 32)
#    The child handles all off‑chip loads and linear algebra.
# 2) `attention_o_proj` consumes the Q, K, V streams together with the
#    output projection weight and produces the final output stream.
#    Its output shape is dictated by this node's contract
#    `(1, 64, 1, 512)`, which we forward via the `out_shapes` argument.
# No additional DSL ops are required; the children perform the necessary
# reshaping, matmuls, and any required off‑chip loads internally.
def attention_block(normed, q_proj, k_proj, v_proj, cos, sin, o_proj_weight, *, out_shapes, out_perms=None):
    Q, K, V = qkv_preprocess(
        normed,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        out_shapes=(
            (1, 64, 16, 1, 32),  # Q stream shape
            (1, 64, 4, 1, 32),   # K stream shape
            (1, 64, 4, 1, 32),   # V stream shape
        ),
    )
    out = attention_o_proj(
        Q,
        K,
        V,
        o_proj_weight,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return out