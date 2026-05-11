def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Helper scalars (pure Python, no tensor creation)
    # ----------------------------------------------------------------------
    seq_len = Q.shape[0]                    # S = 64
    num_heads = Q.shape[1]                  # H = 16
    num_kv_heads = K.shape[1]               # Hkv = 4
    query_per_kvhead = num_heads // num_kv_heads  # qpkv = 4

    # ----------------------------------------------------------------------
    # Q → Qh : (Hkv, qpkv, S, D)  → stream (4, 4) × tile(64, 32)
    # ----------------------------------------------------------------------
    # 1) Move the original tile‑row dimension (heads) into the stream.
    q0 = retile_streamify(Q, chunk=1, split_row=True)          # (S·H, 1, D)
    # 2) Split off the query‑per‑kv‑head axis.
    q1 = reshape_stream(q0, chunk_size=query_per_kvhead, rank=0)  # (S·Hkv, q, 1, D)
    # 3) Split the remaining outer dimension (S·Hkv) into (S, Hkv).
    q2 = reshape_stream(q1, chunk_size=num_kv_heads, rank=1)      # (S, Hkv, q, 1, D)
    # 4) Merge the first two stream dimensions (S and Hkv) -> (S·Hkv, q).
    q3 = flatten(q2, min_rank=1, max_rank=2)                     # (S·Hkv, q, 1, D)
    # 5) Reshape that merged dimension into (Hkv, S, q).
    q4 = reshape_stream(q3, chunk_size=seq_len, rank=1)          # (Hkv, S, q, 1, D)
    # 6) Merge the last two stream dimensions (S and q) -> (Hkv, S·q).
    q5 = flatten(q4, min_rank=0, max_rank=1)                     # (Hkv, S·q, 1, D)
    # 7) Split the merged dim into (q, S) → (Hkv, q, S, 1, D).
    q6 = reshape_stream(q5, chunk_size=seq_len, rank=0)          # (Hkv, q, S, 1, D)
    # 8) Fold the S stream dimension into the tile‑row dimension.
    Qh = accum_retile_row(q6, rank=1)                            # (Hkv, q) × tile(S, D)

    # ----------------------------------------------------------------------
    # K → Kh : (Hkv, 1, S, D)  → stream (4, 1) × tile(64, 32)
    # ----------------------------------------------------------------------
    k0 = retile_streamify(K, chunk=1, split_row=True)          # (S·Hkv, 1, D)
    k1 = reshape_stream(k0, chunk_size=num_kv_heads, rank=0)   # (S, Hkv, 1, D)
    k2 = flatten(k1, min_rank=0, max_rank=1)                   # (S·Hkv, 1, D)
    k3 = reshape_stream(k2, chunk_size=seq_len, rank=0)        # (Hkv, S, 1, D)
    k4 = accum_retile_row(k3, rank=1)                          # (Hkv, S, D)
    Kh = reshape_stream(k4, chunk_size=1, rank=0)              # (Hkv, 1, S, D)

    # ----------------------------------------------------------------------
    # V → Vh : (Hkv, 1, S, D)  → stream (4, 1) × tile(64, 32)
    # ----------------------------------------------------------------------
    v0 = retile_streamify(V, chunk=1, split_row=True)          # (S·Hkv, 1, D)
    v1 = reshape_stream(v0, chunk_size=num_kv_heads, rank=0)   # (S, Hkv, 1, D)
    v2 = flatten(v1, min_rank=0, max_rank=1)                   # (S·Hkv, 1, D)
    v3 = reshape_stream(v2, chunk_size=seq_len, rank=0)        # (Hkv, S, 1, D)
    v4 = accum_retile_row(v3, rank=1)                          # (Hkv, S, D)
    Vh = reshape_stream(v4, chunk_size=1, rank=0)              # (Hkv, 1, S, D)

    return Qh, Kh, Vh