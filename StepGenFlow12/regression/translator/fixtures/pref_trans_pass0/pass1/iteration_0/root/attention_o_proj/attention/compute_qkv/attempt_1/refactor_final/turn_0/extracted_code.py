# The original PyTorch code reshapes Q from (S, H, D) to (Hkv, Q_per_kv, S, D) and
# permutes K, V to (Hkv, 1, S, D).  In the DSL we cannot use .view/.permute, so we
# achieve the same layout with a sequence of stream‑aware primitives:
#   1. `accum_retile_row` merges the sequence stream into the tile‑row dimension,
#      turning (S, H, D) into a single large tile (S*H, D).
#   2. `promote` adds a leading stream dimension of size 1, making the tensor a
#      proper stream so that `retile_streamify` can operate.
#   3. `retile_streamify` splits the enlarged tile‑row into chunks of size
#      `seq_len` (= 64).  For Q this yields a stream of length 16 (= Hkv*Q_per_kv);
#      for K/V it yields a stream of length 4 (= Hkv).
#   4. Finally `reshape_stream` splits the remaining stream dimension into the
#      explicit KV‑head and query‑per‑KV‑head axes.  For K/V we also split a
#      dimension of size 1 to insert the required singleton stream dimension.
# This series of DSL calls produces exactly the shapes
#   Q → (4, 4, 64, 32)
#   K → (4, 1, 64, 32)
#   V → (4, 1, 64, 32)
# without any raw tensor arithmetic or indexing.
def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Q: (seq_len, num_heads, head_dim) -> (num_kv_heads, q_per_kv, seq_len, head_dim)
    Q_h = accum_retile_row(Q, rank=1)                # (seq_len * num_heads, head_dim) -> (1024, 32)
    Q_h = promote(Q_h, rank=1)                       # (1, 1024, 32)
    Q_h = retile_streamify(Q_h, chunk=64, split_row=True)  # (16, 64, 32)
    Q_h = reshape_stream(Q_h, chunk_size=4, rank=0) # (4, 4, 64, 32)

    # K: (seq_len, num_kv_heads, head_dim) -> (num_kv_heads, 1, seq_len, head_dim)
    K_h = accum_retile_row(K, rank=1)                # (256, 32)
    K_h = promote(K_h, rank=1)                       # (1, 256, 32)
    K_h = retile_streamify(K_h, chunk=64, split_row=True)  # (4, 64, 32)
    K_h = reshape_stream(K_h, chunk_size=1, rank=0) # (4, 1, 64, 32)

    # V: same transformation as K
    V_h = accum_retile_row(V, rank=1)                # (256, 32)
    V_h = promote(V_h, rank=1)                       # (1, 256, 32)
    V_h = retile_streamify(V_h, chunk=64, split_row=True)  # (4, 64, 32)
    V_h = reshape_stream(V_h, chunk_size=1, rank=0) # (4, 1, 64, 32)

    return Q_h, K_h, V_h