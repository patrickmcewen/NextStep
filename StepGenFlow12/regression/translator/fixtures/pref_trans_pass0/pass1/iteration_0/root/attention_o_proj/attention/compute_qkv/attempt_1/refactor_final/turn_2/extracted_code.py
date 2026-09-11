# The reference implementation reshapes Q from (S, H, D) to (S, Hkv, Q_per_kv, D) and then
# permutes to (Hkv, Q_per_kv, S, D).  In the DSL we cannot call `.view` or `.permute`,
# so we achieve the same layout with a sequence of stream‑aware primitives:
#   1. `accum_retile_row` merges the sequence (stream) dimension into the tile‑row
#      dimension, turning (S, H, D) into a single large tile (S*H, D).
#   2. `promote(..., rank=0)` adds a leading stream dimension of size 1, making the
#      tensor a proper stream (required for the next step).
#   3. `retile_streamify` splits that huge tile‑row dimension into chunks of size S
#      (the sequence length).  This yields a stream of length H (the number of heads)
#      with tile rows =S.
#   4. `reshape_stream` finally splits the head stream into the two desired head‑split
#      dimensions (num_kv_heads and queries‑per‑kv‑head).
# The same steps are applied to K and V, with a chunk size of 1 for the head‑split
# dimension because K/V have only one query per KV‑head.
def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Q: (seq_len, num_heads, head_dim) -> (num_kv_heads, q_per_kv, seq_len, head_dim)
    Q_h = accum_retile_row(Q, rank=1)                 # (seq_len * num_heads, head_dim) -> (1024, 32)
    Q_h = promote(Q_h, rank=0)                        # (1, 1024, 32)
    Q_h = retile_streamify(Q_h, chunk=64, split_row=True)  # (16, 64, 32)
    Q_h = reshape_stream(Q_h, chunk_size=4, rank=0)  # (4, 4, 64, 32)

    # K: (seq_len, num_kv_heads, head_dim) -> (num_kv_heads, 1, seq_len, head_dim)
    K_h = accum_retile_row(K, rank=1)                 # (seq_len * num_kv_heads, head_dim) -> (256, 32)
    K_h = promote(K_h, rank=0)                        # (1, 256, 32)
    K_h = retile_streamify(K_h, chunk=64, split_row=True)  # (4, 64, 32)
    K_h = reshape_stream(K_h, chunk_size=1, rank=0)  # (4, 1, 64, 32)

    # V: same transformation as K
    V_h = accum_retile_row(V, rank=1)                 # (256, 32)
    V_h = promote(V_h, rank=0)                        # (1, 256, 32)
    V_h = retile_streamify(V_h, chunk=64, split_row=True)  # (4, 64, 32)
    V_h = reshape_stream(V_h, chunk_size=1, rank=0)  # (4, 1, 64, 32)

    return Q_h, K_h, V_h