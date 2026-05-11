# The node computes attention and a final projection.
# Q, K, V arrive already tiled as on‑chip streams, so they can be passed
# directly to the `attention_core` blackbox.  The weight matrix `o_proj_weight`
# is RAW (off‑chip); it may be given straight to the `o_proj` blackbox because
# the stub will load it as needed.
# The intermediate attention result is kept as a tile‑stream with shape
# (1, 64, 16, 32) – a single stream dimension (batch) followed by the token
# dimension and the per‑head tile (16 heads × 32‑dim).  This stream is fed into
# `o_proj`, whose output shape is dictated by the caller via `out_shapes`
# (the required shape is (1, 64, 1, 512)).  No tensor‑method calls are used
# between sources and blackbox calls, satisfying the call‑site rule.
def attention_o_proj(Q, K, V, o_proj_weight, *, out_shapes, out_perms=None):
    # Compute the attention tensor (stream shape (1, 64, 16, 32))
    attn = attention_core(
        Q,
        K,
        V,
        out_shapes=((1, 64, 16, 32),),   # stream shape matching the parent tensor layout
        out_perms=None,
    )
    # Project the attention output to the hidden dimension (final shape given by out_shapes)
    out = o_proj(
        attn,
        o_proj_weight,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return out