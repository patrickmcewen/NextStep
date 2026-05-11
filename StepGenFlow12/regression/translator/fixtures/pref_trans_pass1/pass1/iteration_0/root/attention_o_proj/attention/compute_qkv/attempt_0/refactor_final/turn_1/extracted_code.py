# Implementation reasoning:
# - Q, K, V arrive as on‑chip streams:
#   Q: stream(64,) tile(16,32)   (seq_len × num_heads × head_dim)
#   K, V: stream(64,) tile(4,32) (seq_len × num_kv_heads × head_dim)
# - The reference implementation reshapes Q so that the 16 heads are split into
#   (kv_heads=4, query_per_kvhead=4) and makes the sequence length the tile
#   dimension.  K and V move the 4‑head dimension into a stream slot and also
#   make the sequence length a tile dimension.
# - The DSL lacks a direct permute, but an arbitrary permutation can be built
#   with the pattern: promote → retile_streamify (move tile rows into the
#   stream) → bufferize (turn stream into a flat buffer) → streamify (read the
#   buffer back with a stride that implements the desired permutation) →
#   accum_retile_row (absorb the seq_len stream dimension back into the tile
#   rows).
# - For Q we need two new stream dimensions (kv_heads, query_per_kvhead) and
#   to keep seq_len as a tile row.  For K/V we need one new stream dimension
#   (kv_heads) plus a singleton dimension, then also absorb seq_len.
# - Stride vectors implement the index mapping:
#   * Q: linear_idx = kv*4 + q*1 + seq*16  → stride = (4, 1, 16)
#   * K/V: linear_idx = kv*1 + seq*4      → stride = (1, 0, 4) (the middle
#     dimension is size‑1, so its stride can be 0)
# - Finally, `accum_retile_row` merges the innermost stream (seq_len) into the
#   tile rows, yielding the exact shapes required by the contract.

def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Number of KV heads (the tile‑row size of K/V)
    kv_heads = int(K.shape[-2])  # == 4 for this problem

    # -----------------------------------------------------------------
    # Q: (seq_len, num_heads, head_dim) → (kv, q_per_kv, seq_len, head_dim)
    # -----------------------------------------------------------------
    q_tmp = promote(Q, rank=0)                     # stream (64, 1), tile (16, 32)
    q_tmp = retile_streamify(q_tmp, chunk=1)       # stream (64, 16), tile (1, 32)
    q_buf = bufferize(q_tmp, rank=2)               # Buffer(shape=(64, 16))
    q_tmp = streamify(
        q_buf,
        stride=(kv_heads, 1, 16),                  # (4, 1, 16)
        out_shape_tiled=(kv_heads, kv_heads, 64)   # (4, 4, 64)
    )
    Q_out = accum_retile_row(q_tmp, rank=1)        # absorb seq_len → tile rows=64

    # -----------------------------------------------------------------
    # K: (seq_len, kv_heads, head_dim) → (kv_heads, 1, seq_len, head_dim)
    # -----------------------------------------------------------------
    k_tmp = promote(K, rank=0)                     # stream (64, 1), tile (4, 32)
    k_tmp = retile_streamify(k_tmp, chunk=1)       # stream (64, 4), tile (1, 32)
    k_buf = bufferize(k_tmp, rank=2)               # Buffer(shape=(64, 4))
    k_tmp = streamify(
        k_buf,
        stride=(1, 0, kv_heads),                   # (1, 0, 4)
        out_shape_tiled=(kv_heads, 1, 64)          # (4, 1, 64)
    )
    K_out = accum_retile_row(k_tmp, rank=1)        # absorb seq_len → tile rows=64

    # -----------------------------------------------------------------
    # V: same transformation as K
    # -----------------------------------------------------------------
    v_tmp = promote(V, rank=0)                     # stream (64, 1), tile (4, 32)
    v_tmp = retile_streamify(v_tmp, chunk=1)       # stream (64, 4), tile (1, 32)
    v_buf = bufferize(v_tmp, rank=2)               # Buffer(shape=(64, 4))
    v_tmp = streamify(
        v_buf,
        stride=(1, 0, kv_heads),                   # (1, 0, 4)
        out_shape_tiled=(kv_heads, 1, 64)          # (4, 1, 64)
    )
    V_out = accum_retile_row(v_tmp, rank=1)        # absorb seq_len → tile rows=64

    return Q_out, K_out, V_out