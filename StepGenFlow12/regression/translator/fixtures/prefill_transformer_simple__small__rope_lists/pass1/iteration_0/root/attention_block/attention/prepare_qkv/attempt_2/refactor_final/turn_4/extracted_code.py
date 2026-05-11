# QKV preparation using only DSL ops.
#   Q: (seq_len, heads, dim) → (kv, q_per_kv, seq_len, dim)
#   K/V: (seq_len, kv_heads, dim) → (kv_heads, 1, seq_len, dim)
# The plan:
#   1. Move the original tile‑row dimension (heads) into the stream with
#      `retile_streamify(chunk=1)`.  Tile rows become 1.
#   2. Split the combined stream dimension into separate stream axes:
#        – For Q: (seq_len, heads) → (seq_len, kv, q_per_kv)
#        – For K/V: (seq_len, kv)   → (seq_len, kv)
#   3. Use a Buffered + `streamify` pair to permute the stream axes so that
#      the sequence dimension becomes the innermost stream axis.
#      This is done by treating the stream axes as the buffer grid and
#      providing a stride that reproduces the original linear index.
#   4. Finally merge the innermost stream axis (the sequence) into the tile‑row
#      dimension with `accum_retile_row`.  For K/V we also add a singleton stream
#      dimension after the kv axis with `reshape_stream(chunk_size=1)`.
#
def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------
    # Q → (kv, q_per_kv, seq_len, dim)
    # 1. Move heads into the stream.
    q0 = retile_streamify(Q, chunk=1)                       # stream(1024) × tile(1,32)

    # 2. Split the combined stream into (seq_len, heads).
    q1 = reshape_stream(q0, chunk_size=16, rank=0)          # stream(64,16) × tile(1,32)

    # 3. Split heads (16) into (kv, q_per_kv) = (4,4).
    q2 = reshape_stream(q1, chunk_size=4, rank=0)           # stream(64,4,4) × tile(1,32)

    # 4. Bufferize the three stream dimensions so we can reorder them.
    q_buf = bufferize(q2, rank=3)                           # buffer grid = (64,4,4)

    # 5. Reorder to (kv, q_per_kv, seq_len) via streamify.
    #    Linear index = seq·16 + kv·4 + q  (original layout)
    #    For out shape (kv=4, q=4, seq=64) we need stride = [4, 1, 16].
    q3 = streamify(q_buf, stride=[4, 1, 16], out_shape_tiled=(4, 4, 64))
                                                             # stream(4,4,64) × tile(1,32)

    # 6. Merge the innermost stream (seq_len) into the tile‑row dimension.
    Qh = accum_retile_row(q3, rank=1)                       # stream(4,4) × tile(64,32)

    # ------------------------------
    # K → (kv, 1, seq_len, dim)
    # 1. Move kv heads into the stream.
    k0 = retile_streamify(K, chunk=1)                       # stream(256) × tile(1,32)

    # 2. Split into (seq_len, kv) = (64,4).
    k1 = reshape_stream(k0, chunk_size=4, rank=0)           # stream(64,4) × tile(1,32)

    # 3. Bufferize both stream axes.
    k_buf = bufferize(k1, rank=2)                           # buffer grid = (64,4)

    # 4. Reorder to (kv, seq_len).  Original linear index = seq·4 + kv.
    #    For out shape (kv=4, seq=64) stride = [1, 4].
    k2 = streamify(k_buf, stride=[1, 4], out_shape_tiled=(4, 64))
                                                             # stream(4,64) × tile(1,32)

    # 5. Merge seq_len into tile rows.
    k3 = accum_retile_row(k2, rank=1)                       # stream(4) × tile(64,32)

    # 6. Add a singleton stream dimension after kv.
    Kh = reshape_stream(k3, chunk_size=1, rank=0)           # stream(4,1) × tile(64,32)

    # ------------------------------
    # V → (kv, 1, seq_len, dim)  (identical to K)
    v0 = retile_streamify(V, chunk=1)                       # stream(256) × tile(1,32)
    v1 = reshape_stream(v0, chunk_size=4, rank=0)           # stream(64,4) × tile(1,32)
    v_buf = bufferize(v1, rank=2)                           # buffer grid = (64,4)
    v2 = streamify(v_buf, stride=[1, 4], out_shape_tiled=(4, 64))
                                                             # stream(4,64) × tile(1,32)
    v3 = accum_retile_row(v2, rank=1)                       # stream(4) × tile(64,32)
    Vh = reshape_stream(v3, chunk_size=1, rank=0)           # stream(4,1) × tile(64,32)

    return Qh, Kh, Vh