# QKV preparation using only DSL operations.
#   Q: (seq_len, heads, dim) → (kv, q_per_kv, seq_len, dim)
#   K/V: (seq_len, kv_heads, dim) → (kv, 1, seq_len, dim)
# The transformation is expressed as a series of stream‑shape manipulations:
#   * `retile_streamify` moves the original tile‑row dimension into the stream.
#   * `reshape_stream` splits a stream dimension into two (e.g., heads → kv×q_per_kv,
#     or combined head×seq → kv×seq).
#   * `accum_retile_row` merges the innermost stream dimension into the tile‑row axis,
#     turning the sequence length into the tile‑row dimension.
#   * A final `reshape_stream` introduces a singleton stream dimension for K and V.
#
def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # --------------------
    # Q: (64, 16, 32) → (4, 4, 64, 32)
    q0 = retile_streamify(Q, chunk=1)                     # → stream(1024) × tile(1,32)
    q1 = reshape_stream(q0, chunk_size=64, rank=0)        # → stream(16,64) × tile(1,32)
    q2 = reshape_stream(q1, chunk_size=4, rank=1)         # → stream(4,4,64) × tile(1,32)
    Qh = accum_retile_row(q2, rank=1)                     # → stream(4,4) × tile(64,32)

    # --------------------
    # K: (64, 4, 32) → (4, 1, 64, 32)
    k0 = retile_streamify(K, chunk=1)                     # → stream(256) × tile(1,32)
    k1 = reshape_stream(k0, chunk_size=64, rank=0)        # → stream(4,64) × tile(1,32)
    k2 = accum_retile_row(k1, rank=1)                     # → stream(4) × tile(64,32)
    Kh = reshape_stream(k2, chunk_size=1, rank=0)          # → stream(4,1) × tile(64,32)

    # --------------------
    # V: (64, 4, 32) → (4, 1, 64, 32) (identical to K)
    v0 = retile_streamify(V, chunk=1)                     # → stream(256) × tile(1,32)
    v1 = reshape_stream(v0, chunk_size=64, rank=0)        # → stream(4,64) × tile(1,32)
    v2 = accum_retile_row(v1, rank=1)                     # → stream(4) × tile(64,32)
    Vh = reshape_stream(v2, chunk_size=1, rank=0)          # → stream(4,1) × tile(64,32)

    return Qh, Kh, Vh