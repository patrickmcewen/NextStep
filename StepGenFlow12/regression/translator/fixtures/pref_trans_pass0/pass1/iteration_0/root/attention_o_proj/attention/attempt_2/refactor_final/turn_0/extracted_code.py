# The attention node simply wires together the two child blackboxes.
# Q, K, V are already on‑chip streams (shape (64,16,32), (64,4,32), (64,4,32)).
# 1) Call `compute_qkv` to obtain streamed heads Qh, Kh, Vh.  The child expects
#    stream shapes that match its vanilla output shapes; we provide them directly:
#      Qh : (4, 4, 64, 32)   (2 stream dims + tile dims)
#      Kh : (4, 1, 64, 32)
#      Vh : (4, 1, 64, 32)
#    No tensor‑method transformations are applied between the inputs and the
#    blackbox call.
# 2) Feed those three streams into `attention_compute`.  The required output shape
#    for this node is given by `out_shapes` (a 1‑tuple containing (64,16,32));
#    we forward that shape (and any requested permutation) to the child.
# The result of `attention_compute` is the final attention tensor stream,
# which we return directly.  All operations are pure blackbox calls; there is
# no off‑chip load/store or other DSL manipulation needed here.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    # Step 1: project Q, K, V into head representations.
    Qh, Kh, Vh = compute_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),   # Qh stream shape
            (4, 1, 64, 32),   # Kh stream shape
            (4, 1, 64, 32),   # Vh stream shape
        ),
        out_perms=(None, None, None),
    )

    # Step 2: perform attention on the head streams.
    attn = attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )

    return attn