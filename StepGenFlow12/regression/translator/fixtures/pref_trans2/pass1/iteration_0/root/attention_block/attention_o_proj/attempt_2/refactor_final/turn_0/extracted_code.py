# Implementation reasoning:
# This node composes the two child blackboxes `attention_core` and `o_proj`.
# The inputs `Q`, `K`, `V` are already on‑chip streams with shape (1,64,16,32).
# We first invoke `attention_core` to obtain the attention tensor, preserving the
# same stream shape as `Q`.  The child stub will handle the conversion from the
# tiled stream to the vanilla shape it expects.
# The resulting attention stream is then passed, together with the raw weight
# matrix `o_proj_weight`, to the `o_proj` blackbox.  The parent requires the final
# output shape (1,64,1,512); we simply forward the `out_shapes` and `out_perms`
# arguments we received so the child can emit exactly that shape (and identity
# permutation).  No off‑chip load is needed for `o_proj_weight` because it can be
# given directly to the child blackbox, which recovers its vanilla shape
# internally.
def attention_o_proj(Q, K, V, o_proj_weight, *, out_shapes, out_perms=None):
    # Compute attention output (stream shape matches Q).
    attn = attention_core(
        Q,
        K,
        V,
        out_shapes=((1, 64, 16, 32),),
        out_perms=(None,),
    )
    # Project attention to hidden dimension (final required shape).
    out = o_proj(
        attn,
        o_proj_weight,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return out