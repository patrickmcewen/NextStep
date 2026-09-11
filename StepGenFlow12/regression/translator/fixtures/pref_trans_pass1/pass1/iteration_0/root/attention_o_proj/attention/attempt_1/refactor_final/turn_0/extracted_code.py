# The `attention` node simply wires together the two child blackboxes.
# `Q`, `K`, `V` are already on‑chip streams, so they can be passed directly.
# `compute_qkv` expects tile‑stream shapes that match its three vanilla
# outputs: Qh (4,4,64,32), Kh (4,1,64,32), Vh (4,1,64,32).  We provide those
# shapes explicitly.  The final `attention_compute` output shape is dictated by
# the caller via `out_shapes`/`out_perms`, which we forward unchanged.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    Qh, Kh, Vh = compute_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),   # Qh vanilla shape as a tile‑stream shape
            (4, 1, 64, 32),   # Kh vanilla shape as a tile‑stream shape
            (4, 1, 64, 32),   # Vh vanilla shape as a tile‑stream shape
        ),
        out_perms=None,
    )
    # Produce the final attention output, respecting the caller‑requested
    # stream shape and permutation.
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn