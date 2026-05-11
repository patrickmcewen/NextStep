# The attention node simply wires together the three child blackboxes that
# implement the full attention computation.
#   1. `prepare_qkv` splits the on‑chip Q, K, V streams into per‑expert
#      sub‑streams (Qh, Kh, Vh).  Its outputs have the fixed tile‑stream
#      shapes (4,4,64,32), (4,1,64,32) and (4,1,64,32) respectively.
#   2. `attention_weights` computes the attention matrix from Qh and Kh,
#      yielding a stream of shape (4,4,64,64).
#   3. `apply_weights_and_reshape` applies the attention matrix to Vh and
#      reshapes the result back to the original layout.  The desired output
#      shape for this node is supplied via `out_shapes` (and optionally
#      `out_perms`), so we forward those directly.
# No tensor‑method transforms are performed between the calls; all shape
# handling is delegated to the child stubs via their `out_shapes` arguments.
def attention(Q, K, V, *, out_shapes, out_perms=None):
    # 1. Split Q, K, V into per‑expert streams.
    Qh, Kh, Vh = prepare_qkv(
        Q,
        K,
        V,
        out_shapes=(
            (4, 4, 64, 32),   # Qh
            (4, 1, 64, 32),   # Kh
            (4, 1, 64, 32),   # Vh
        ),
    )

    # 2. Compute attention weights.
    attn_weights = attention_weights(
        Qh,
        Kh,
        out_shapes=((4, 4, 64, 64),),
    )

    # 3. Apply weights to V and reshape to the requested output shape.
    attn = apply_weights_and_reshape(
        attn_weights,
        Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn