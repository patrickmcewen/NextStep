# The attention_compute node must reshape the on‑chip inputs Q, K, V into the
# shapes expected by the heavy attention blackbox (Qh: (4,4,64,32), Kh/Vh:
# (4,1,64,32)).  This is done by a sequence of `reshape_stream` and
# `accum_retile_row` calls that move the sequence dimension from the stream
# into the tile‑row dimension and split the head dimension into the two GQA
# stream dimensions.  After the blackbox computes the attention we request its
# output be reshaped directly to the contract’s final shape `(64, 16, 32)`,
# avoiding an explicit inverse transformation.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # Q: (S=64, heads=16, D=32) -> Qh: (4, 4, 64, 32)
    qh = reshape_stream(Q, chunk_size=4, rank=0)          # (16, 4, 16, 32)
    qh = accum_retile_row(qh, rank=1)                    # (16, 64, 32)
    qh = reshape_stream(qh, chunk_size=4, rank=0)        # (4, 4, 64, 32)

    # K: (S=64, kv_heads=4, D=32) -> Kh: (4, 1, 64, 32)
    kh = reshape_stream(K, chunk_size=4, rank=0)          # (16, 4, 4, 32)
    kh = accum_retile_row(kh, rank=1)                    # (16, 16, 32)
    kh = reshape_stream(kh, chunk_size=4, rank=0)        # (4, 4, 16, 32)
    kh = accum_retile_row(kh, rank=1)                    # (4, 64, 32)
    kh = reshape_stream(kh, chunk_size=1, rank=0)        # (4, 1, 64, 32)

    # V: (S=64, kv_heads=4, D=32) -> Vh: (4, 1, 64, 32)
    vh = reshape_stream(V, chunk_size=4, rank=0)          # (16, 4, 4, 32)
    vh = accum_retile_row(vh, rank=1)                    # (16, 16, 32)
    vh = reshape_stream(vh, chunk_size=4, rank=0)        # (4, 4, 16, 32)
    vh = accum_retile_row(vh, rank=1)                    # (4, 64, 32)
    vh = reshape_stream(vh, chunk_size=1, rank=0)        # (4, 1, 64, 32)

    # Heavy attention computation; request the final output shape directly.
    attn = attention_compute__root_attention_attention_compute(
        qh, kh, vh, out_shapes=out_shapes, out_perms=out_perms
    )

    return attn