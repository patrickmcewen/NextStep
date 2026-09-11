# The attention node simply wires together the two child blackboxes.
# • `Q`, `K`, `V` are already on‑chip streams, so they can be handed directly
#   to `compute_qkv` without any loading or reshaping.
# • `compute_qkv` is asked to emit three streams with the shapes required by
#   the model: Q‑head `(4,4,64,32)` and K/V‑head `(4,1,64,32)`.  Those shapes
#   already satisfy the DSL invariant (≥ 3‑D, last two dimensions are the tile).
# • The resulting Qh, Kh, Vh are fed into `attention_compute`, which produces
#   the final attention tensor.  Its output shape is exactly the `out_shapes`
#   requested for this node, so we just forward the `out_shapes` and any
#   `out_perms` the caller supplied.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    Qh, Kh, Vh = compute_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),   # Q‑head stream shape
            (4, 1, 64, 32),   # K‑head stream shape
            (4, 1, 64, 32),   # V‑head stream shape
        ),
        out_perms=None,
    )
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn