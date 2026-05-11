# The input tensors Q, K, V are on‑chip streams with shapes
#   Q: (seq_len, num_heads, head_dim)
#   K, V: (seq_len, num_kv_heads, head_dim)
# We need to produce the multi‑query layout:
#   Qh → (num_kv_heads, query_per_kvhead, seq_len, head_dim)
#   Kh, Vh → (num_kv_heads, 1, seq_len, head_dim)
#
# The required shape changes are expressed entirely with DSL primitives:
#   1. `promote_outer` adds a leading stream dimension (needed because
#      `accum_retile_row` expects at least one stream rank to merge).
#   2. `accum_retile_row` merges the original sequence‑length stream dimension
#      into the tile‑row dimension, yielding a 2‑D tensor (seq_len * heads, head_dim)
#      while preserving a singleton stream rank.
#   3. `retile_streamify` splits the enlarged tile‑row dimension back into a
#      stream dimension (the number of heads) and restores the desired tile‑row
#      size (seq_len).  The `chunk` argument is set to `seq_len`.
#   4. `reshape_stream` finally splits the remaining stream dimension (the total
#      number of heads) into the two required stream axes:
#        – for Q: `query_per_kvhead` (e.g. 4) and `num_kv_heads` (e.g. 4);
#        – for K/V: a singleton dimension (chunk = 1) after the `num_kv_heads`
#          stream axis.
#
# All arithmetic is performed on scalar Python ints; no raw tensor ops are used.
def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # scalar dimensions
    seq_len = Q.shape[0]                      # S
    num_heads = Q.shape[1]                    # H
    num_kv_heads = K.shape[1]                 # Hkv
    query_per_kvhead = num_heads // num_kv_heads  # qpkv

    # ---- Q (full multi‑query) ------------------------------------------------
    # 1) add outer stream dim
    Q_prom = promote_outer(Q)                 # (1, S, H, D)
    # 2) merge the original stream (S) into tile rows (H) → (1, S*H, D)
    Q_merged = accum_retile_row(Q_prom, rank=1)
    # 3) split the enlarged tile rows (S*H) back into a stream of size H
    #    and restore tile rows = S
    Q_split = retile_streamify(Q_merged, chunk=seq_len, split_row=True)  # (H, S, D)
    # 4) split the remaining stream (H) into (Hkv, qpkv)
    Qh = reshape_stream(Q_split, chunk_size=query_per_kvhead, rank=0)

    # ---- K (single‑head per KV) ---------------------------------------------
    K_prom = promote_outer(K)                 # (1, S, Hkv, D)
    K_merged = accum_retile_row(K_prom, rank=1)  # (1, S*Hkv, D)
    K_split = retile_streamify(K_merged, chunk=seq_len, split_row=True)  # (Hkv, S, D)
    Kh = reshape_stream(K_split, chunk_size=1, rank=0)  # (Hkv, 1, S, D)

    # ---- V (same pattern as K) -----------------------------------------------
    V_prom = promote_outer(V)                 # (1, S, Hkv, D)
    V_merged = accum_retile_row(V_prom, rank=1)  # (1, S*Hkv, D)
    V_split = retile_streamify(V_merged, chunk=seq_len, split_row=True)  # (Hkv, S, D)
    Vh = reshape_stream(V_split, chunk_size=1, rank=0)  # (Hkv, 1, S, D)

    return Qh, Kh, Vh