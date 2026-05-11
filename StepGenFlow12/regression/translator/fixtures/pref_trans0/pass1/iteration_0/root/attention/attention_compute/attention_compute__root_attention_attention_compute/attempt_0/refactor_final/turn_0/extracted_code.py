# The attention kernel consists of two stages:
#   1. Compute the raw attention scores `e = Q·Kᵀ` via the `compute_e` child.
#   2. Apply the scores to the values `V` via the `compute_attn` child.
# All inputs are already on‑chip tiled streams, so they can be passed
# directly to the children.  The children need explicit `out_shapes`
# (and optionally `out_perms`) because they internally flatten the
# streams to vanilla tensors.  `compute_e` emits a `(4,4,64,64)` stream,
# and `compute_attn` produces the final `(4,4,64,32)` stream, which we
# return unchanged (the caller supplies the desired `out_shapes`/`out_perms`).
def attention_compute__root_attention_attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Stage 1: raw attention scores
    e = compute_e(Qh, Kh, out_shapes=((4, 4, 64, 64),), out_perms=(None,))
    # Stage 2: weighted sum of values
    attn = compute_attn(e, Vh, out_shapes=out_shapes, out_perms=out_perms)
    return attn