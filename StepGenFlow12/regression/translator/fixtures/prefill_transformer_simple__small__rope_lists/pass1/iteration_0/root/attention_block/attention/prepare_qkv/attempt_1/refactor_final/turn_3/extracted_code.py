def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Helper scalars (pure Python, no tensor creation)
    # ----------------------------------------------------------------------
    seq_len = Q.shape[0]                         # S = 64
    num_heads = Q.shape[1]                       # H = 16
    num_kv_heads = K.shape[1]                    # Hkv = 4
    query_per_kvhead = num_heads // num_kv_heads  # qpkv = 4

    # ----------------------------------------------------------------------
    # Q → Qh : (Hkv, qpkv, S, D)  → stream (4,4) × tile(64,32)
    # ----------------------------------------------------------------------
    # 1) Pull heads (tile‑rows) into the stream dimension.
    q0 = retile_streamify(Q, chunk=1, split_row=True)            # (S·H, 1, D)
    # 2) Split that combined stream into (S·Hkv, qpkv).
    q1 = reshape_stream(q0, chunk_size=query_per_kvhead, rank=0)  # (S·Hkv, qpkv, 1, D)
    # 3) Split the first stream dim into (Hkv, S).
    q2 = reshape_stream(q1, chunk_size=seq_len, rank=1)           # (Hkv, S, qpkv, 1, D)
    # 4) Merge the inner two stream dims (S and qpkv) → (Hkv, S·qpkv).
    q3 = flatten(q2, min_rank=0, max_rank=1)                      # (Hkv, S·qpkv, 1, D)
    # 5) Split that product back into (qpkv, S), giving order (Hkv, qpkv, S).
    q4 = reshape_stream(q3, chunk_size=seq_len, rank=0)           # (Hkv, qpkv, S, 1, D)
    # 6) Merge the innermost stream dim (S) into tile rows.
    Qh = accum_retile_row(q4, rank=1)                             # stream (Hkv, qpkv) × tile(S, D)

    # ----------------------------------------------------------------------
    # K → Kh : (Hkv, 1, S, D)  → stream (4,1) × tile(64,32)
    # ----------------------------------------------------------------------
    k0 = retile_streamify(K, chunk=1, split_row=True)            # (S·Hkv, 1, D)
    k1 = reshape_stream(k0, chunk_size=seq_len, rank=0)           # (Hkv, S, 1, D)
    k2 = accum_retile_row(k1, rank=1)                             # stream (Hkv) × tile(S, D)
    Kh = reshape_stream(k2, chunk_size=1, rank=0)                 # stream (Hkv, 1) × tile(S, D)

    # ----------------------------------------------------------------------
    # V → Vh : (Hkv, 1, S, D)  → stream (4,1) × tile(64,32)
    # ----------------------------------------------------------------------
    v0 = retile_streamify(V, chunk=1, split_row=True)            # (S·Hkv, 1, D)
    v1 = reshape_stream(v0, chunk_size=seq_len, rank=0)           # (Hkv, S, 1, D)
    v2 = accum_retile_row(v1, rank=1)                             # stream (Hkv) × tile(S, D)
    Vh = reshape_stream(v2, chunk_size=1, rank=0)                 # stream (Hkv, 1) × tile(S, D)

    return Qh, Kh, Vh