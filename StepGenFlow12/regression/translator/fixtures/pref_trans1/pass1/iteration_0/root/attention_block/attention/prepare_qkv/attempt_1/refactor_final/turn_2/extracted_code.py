def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Helper scalars (pure Python, no tensor creation)
    # ----------------------------------------------------------------------
    seq_len = Q.shape[0]                    # 64
    num_heads = Q.shape[1]                  # 16
    num_kv_heads = K.shape[1]               # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ----------------------------------------------------------------------
    # Q → Qh : (Hkv, qpkv, S, D)
    # ----------------------------------------------------------------------
    # 1. Move the original tile‑rows (heads) into the stream.
    q0 = retile_streamify(Q, chunk=1, split_row=True)          # stream: S*H , tile rows: 1
    # 2. Split that stream into (Hkv·S) × qpkv.
    q1 = reshape_stream(q0, chunk_size=query_per_kvhead, rank=0)  # (S·Hkv, qpkv)
    # 3. Split the left‑most dimension into (Hkv, S).
    q2 = reshape_stream(q1, chunk_size=seq_len, rank=1)          # (Hkv, S, qpkv)
    # 4. Merge the middle (S) and innermost (qpkv) stream dims.
    q3 = flatten(q2, min_rank=0, max_rank=1)                     # (Hkv, S·qpkv)
    # 5. Split the merged dim back into (qpkv, S) – now ordering is (Hkv, qpkv, S).
    q4 = reshape_stream(q3, chunk_size=seq_len, rank=0)          # (Hkv, qpkv, S)
    # 6. Promote the token dimension into tile rows.
    Qh = accum_retile_row(q4, rank=1)                           # (Hkv, qpkv) × tile( S , D)

    # ----------------------------------------------------------------------
    # K → Kh : (Hkv, 1, S, D)
    # ----------------------------------------------------------------------
    k0 = retile_streamify(K, chunk=1, split_row=True)          # stream: S·Hkv , tile rows: 1
    k1 = reshape_stream(k0, chunk_size=seq_len, rank=1)          # (Hkv, S, 1)
    k2 = reshape_stream(k1, chunk_size=1, rank=0)                # (Hkv, S, 1, 1)
    k3 = accum_retile_row(k2, rank=2)                           # (Hkv) × tile( S , D)
    Kh = reshape_stream(k3, chunk_size=1, rank=0)                # (Hkv, 1) × tile( S , D)

    # ----------------------------------------------------------------------
    # V → Vh : (Hkv, 1, S, D)
    # ----------------------------------------------------------------------
    v0 = retile_streamify(V, chunk=1, split_row=True)          # stream: S·Hkv , tile rows: 1
    v1 = reshape_stream(v0, chunk_size=seq_len, rank=1)          # (Hkv, S, 1)
    v2 = reshape_stream(v1, chunk_size=1, rank=0)                # (Hkv, S, 1, 1)
    v3 = accum_retile_row(v2, rank=2)                           # (Hkv) × tile( S , D)
    Vh = reshape_stream(v3, chunk_size=1, rank=0)                # (Hkv, 1) × tile( S , D)

    return Qh, Kh, Vh