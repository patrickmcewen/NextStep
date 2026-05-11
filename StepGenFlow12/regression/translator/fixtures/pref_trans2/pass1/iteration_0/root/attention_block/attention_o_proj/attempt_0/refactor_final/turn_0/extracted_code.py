# The node simply composes the two provided blackboxes.
# Q, K, V are already on‑chip streams with shape (1,64,16,32) and can be fed
# straight into `attention_core`.  The child returns the attention tensor in the
# same stream shape, which we forward to `o_proj`.  The projection weight is a
# raw off‑chip tensor; it may be passed directly to the `o_proj` blackbox without
# an off‑chip load, because blackboxes handle the recovery of vanilla shape
# internally.  The required output shape for this node is supplied via
# `out_shapes` (e.g. (1, 64, 1, 512)) and is simply propagated to `o_proj`,
# which produces the correctly tiled result.  No DSL tensor methods are used
# between sources and blackboxes.

def attention_o_proj(Q, K, V, o_proj_weight, *, out_shapes, out_perms=None):
    # Shape of the attention output aligns with the shape of the query stream.
    attn_out_shape = Q.shape  # (1, 64, 16, 32)

    # Compute attention using the child blackbox.
    attn = attention_core(
        Q,
        K,
        V,
        out_shapes=(attn_out_shape,),
        out_perms=(None,),
    )

    # Project the attention result to the final hidden dimension.
    # `out_shapes` and `out_perms` are exactly the shapes requested for this node.
    out = o_proj(
        attn,
        o_proj_weight,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return out