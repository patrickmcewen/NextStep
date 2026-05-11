# The reference implementation reshapes Q from (S, H, D) to (Hkv, qpkv, S, D) and
# simply permutes K/V from (S, Hkv, D) to (Hkv, 1, S, D).
# In the DSL the tensor shapes are stream dimensions + two tile dimensions.
# Q is streamed over the sequence (stream=64, tile=(16, 32)).  We need to
#   * swap the stream dimension (seq_len) with the tile‑row dimension (heads),
#   * then split the former heads dimension (size 16) into the two required
#     stream dimensions (num_kv_heads=4, query_per_kvhead=4).
# This is done by:
#   1. permuting the tensor to make the heads axis a stream axis,
#   2. using `reshape_stream` to split that stream axis into the two
#      desired stream axes (chunk size = query_per_kvhead).
# For K and V we only need to swap the stream and tile‑row axes and then
# insert a singleton stream dimension.  `promote` adds that singleton.
#
# The operations used are all DSL calls (`reshape_stream`, `promote`);
# the only Python arithmetic is on dimension sizes (ints).

def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Compute the number of query heads per KV head (must be an integer).
    query_per_kvhead = Q.shape[1] // K.shape[1]  # 16 // 4 = 4

    # Q: (seq_len, heads, dim) -> (heads, seq_len, dim) then split heads.
    Q_perm = Q.permute(1, 0, 2)                     # (heads, seq_len, dim)
    Qh = reshape_stream(Q_perm, chunk_size=query_per_kvhead, rank=0)  # (4, 4, 64, 32)

    # K and V: (seq_len, kv_heads, dim) -> (kv_heads, seq_len, dim)
    # then insert a singleton stream dimension after kv_heads.
    K_perm = K.permute(1, 0, 2)                     # (kv_heads, seq_len, dim)
    V_perm = V.permute(1, 0, 2)                     # (kv_heads, seq_len, dim)

    Kh = promote(K_perm, rank=0)                   # (kv_heads, 1, seq_len, dim) -> (4, 1, 64, 32)
    Vh = promote(V_perm, rank=0)                   # (kv_heads, 1, seq_len, dim) -> (4, 1, 64, 32)

    return Qh, Kh, Vh