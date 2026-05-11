# The inputs Q, K, V are already on‑chip streams:
#   Q: (seq_len, num_heads, head_dim)   → stream(64,)×tile(16,32)
#   K, V: (seq_len, num_kv_heads, head_dim) → stream(64,)×tile(4,32)
#
# Required outputs:
#   Qh → (num_kv_heads, query_per_kvhead, seq_len, head_dim) = (4,4,64,32)
#   Kh, Vh → (num_kv_heads, 1, seq_len, head_dim) = (4,1,64,32)
#
# The transformation can be performed entirely with on‑chip DSL ops:
#   1. `accum_retile_row` merges the sequence‑length stream dimension into the
#      tile‑row dimension, yielding a 2‑D tensor of shape (heads*seq_len, dim).
#   2. `promote` adds a leading singleton stream dimension so that the tensor
#      is a valid stream (rank ≥ 1) for the next step.
#   3. `retile_streamify` splits the enlarged tile‑row dimension back into a
#      stream dimension of size `num_heads` (or `num_kv_heads`) and restores the
#      tile‑row size to `seq_len`.
#   4. `reshape_stream` finally splits the remaining stream dimension:
#        – for Q into (num_kv_heads, query_per_kvhead)
#        – for K/V into (num_kv_heads, 1)
#   All arithmetic is scalar Python int manipulation; no raw tensor ops are used.
def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # scalar dimensions
    seq_len = Q.shape[0]                     # S = 64
    num_heads = Q.shape[1]                   # H = 16
    num_kv_heads = K.shape[1]                # Hkv = 4
    query_per_kvhead = num_heads // num_kv_heads  # qpkv = 4

    # -------------------------------------------------
    # Q → (Hkv, qpkv, S, D)
    # -------------------------------------------------
    Q_merged = accum_retile_row(Q, rank=1)            # (H*S, D) = (1024, 32)
    Q_prom = promote(Q_merged, rank=0)                # (1, H*S, D) = (1,1024,32)
    Q_retiled = retile_streamify(Q_prom, chunk=seq_len)  # (H, S, D) = (16,64,32)
    Qh = reshape_stream(Q_retiled, chunk_size=query_per_kvhead, rank=0)  # (Hkv, qpkv, S, D) = (4,4,64,32)

    # -------------------------------------------------
    # K → (Hkv, 1, S, D)
    # -------------------------------------------------
    K_merged = accum_retile_row(K, rank=1)            # (Hkv*S, D) = (256, 32)
    K_prom = promote(K_merged, rank=0)                # (1,256,32)
    K_retiled = retile_streamify(K_prom, chunk=seq_len)  # (Hkv, S, D) = (4,64,32)
    Kh = reshape_stream(K_retiled, chunk_size=1, rank=0)   # (Hkv, 1, S, D) = (4,1,64,32)

    # -------------------------------------------------
    # V → (Hkv, 1, S, D)
    # -------------------------------------------------
    V_merged = accum_retile_row(V, rank=1)            # (Hkv*S, D) = (256, 32)
    V_prom = promote(V_merged, rank=0)                # (1,256,32)
    V_retiled = retile_streamify(V_prom, chunk=seq_len)  # (Hkv, S, D) = (4,64,32)
    Vh = reshape_stream(V_retiled, chunk_size=1, rank=0)   # (Hkv, 1, S, D) = (4,1,64,32)

    return Qh, Kh, Vh