# Directly delegate to the heavy‑attention blackbox – the child stub
# handles the necessary reshapes internally.  No explicit tensor‑ops are
# needed; we simply forward the on‑chip streams and request the contract’s
# output shape.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    return attention_compute__root_attention_attention_compute(
        Q, K, V,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )