# The attention root node simply stitches together the two child
# blackboxes. Both Qh, Kh, Vh are already on‑chip streams, so they can be
# passed directly to the children.  The children are responsible for any
# internal reshaping; we just supply the expected stream shapes for their
# outputs and forward the final result.  The contract requires the root
# output to have shape (4, 4, 64, 32) and identity permutation, which matches
# the shape produced by `compute_attn`, so we return it unchanged.
def attention_compute__root_attention_attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Compute raw attention scores e = Qh @ Khᵀ
    e = compute_e(
        Qh,
        Kh,
        out_shapes=((4, 4, 64, 64),),   # child expects this stream shape
        out_perms=(None,),
    )
    # Apply attention weights to values Vh
    attn = compute_attn(
        e,
        Vh,
        out_shapes=((4, 4, 64, 32),),   # child returns the final stream shape
        out_perms=(None,),
    )
    # The caller’s out_shapes should be ((4, 4, 64, 32),); they match `attn`.
    return attn