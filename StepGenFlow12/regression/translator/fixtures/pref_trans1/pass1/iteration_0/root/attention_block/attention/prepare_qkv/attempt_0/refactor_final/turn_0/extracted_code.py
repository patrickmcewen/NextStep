# The Q, K, V tensors are already on‑chip streams with shape (seq_len, heads, head_dim).
# We need to reshape them to the multi‑query layout expected by downstream
# attention kernels:
#   Q → (num_kv_heads, query_per_kvhead, seq_len, head_dim)
#   K,V → (num_kv_heads, 1, seq_len, head_dim)
# This is achieved by:
#   * Computing the derived dimensions (seq_len, head_dim, num_heads, num_kv_heads,
#     query_per_kvhead).
#   * Using `.view` to split the head dimension of Q into the two new stream axes,
#     then `.permute` to order them as required.
#   * For K and V we simply permute the (seq_len, num_kv_heads, head_dim) axes
#     and add a singleton stream dimension with `unsqueeze(1)`.
# The resulting tensors have the exact shapes required by the contract:
#   Qh: (4, 4, 64, 32)
#   Kh, Vh: (4, 1, 64, 32)
def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # Q: [S, H, D]   K,V: [S, Hkv, D]
    seq_len = Q.shape[0]          # S = 64
    head_dim = Q.shape[2]         # D = 32
    num_heads = Q.shape[1]        # H = 16
    num_kv_heads = K.shape[1]     # Hkv = 4
    query_per_kvhead = num_heads // num_kv_heads  # 16 // 4 = 4

    # Reshape Q to (S, Hkv, query_per_kvhead, D) then permute to (Hkv, query_per_kvhead, S, D)
    Qh = Q.view(seq_len, num_kv_heads, query_per_kvhead, head_dim).permute(1, 2, 0, 3)

    # K and V: (S, Hkv, D) → (Hkv, 1, S, D)
    Kh = K.permute(1, 0, 2).unsqueeze(1)
    Vh = V.permute(1, 0, 2).unsqueeze(1)

    return Qh, Kh, Vh