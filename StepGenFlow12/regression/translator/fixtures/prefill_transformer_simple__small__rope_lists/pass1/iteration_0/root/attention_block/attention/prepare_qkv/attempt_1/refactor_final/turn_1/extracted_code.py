def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Implementation reasoning:
    #   * Q, K, V are on‑chip streams:
    #       Q : stream(S) × tile(H, D)   where S=seq_len, H=num_heads, D=head_dim
    #       K, V : stream(S) × tile(Hkv, D) where Hkv=num_kv_heads
    #
    #   Goal (PyTorch reference):
    #       Qh = Q.view(S, Hkv, qpkv, D).permute(1, 2, 0, 3)   → (Hkv, qpkv, S, D)
    #       Kh = K.permute(1, 0, 2).unsqueeze(1)               → (Hkv, 1, S, D)
    #       Vh = V.permute(1, 0, 2).unsqueeze(1)               → (Hkv, 1, S, D)
    #
    #   The DSL only reshapes streams, never directly permutes tile axes.
    #   We achieve the required re‑ordering by:
    #     1. Moving tile‑row dimensions (heads) into the stream dimension
    #        with `retile_streamify(..., chunk=1)`.  After this the stream
    #        length becomes S * H (or S * Hkv) and tile rows are 1.
    #     2. Splitting that combined stream back into separate stream axes
    #        using `reshape_stream`.  The first split isolates the original
    #        sequence length, producing a stream of (H, S) for Q and (Hkv, S)
    #        for K/V.
    #     3. For Q we further split the head axis into (num_kv_heads, q_per_kv)
    #        with a second `reshape_stream`.  The ordering of the split
    #        (chunk = query_per_kvhead) yields the desired stream order
    #        (num_kv_heads, q_per_kv, S).
    #     4. Finally `accum_retile_row` merges the innermost stream axis
    #        (the sequence dimension) into the tile‑row dimension, producing
    #        tiles sized (S, D).  For K/V we only need a single extra stream
    #        dimension of size 1, which we add with a final `reshape_stream`.
    # ----------------------------------------------------------------------

    # scalar dimensions
    seq_len = Q.shape[0]                     # S
    num_kv_heads = K.shape[-2]               # Hkv
    query_per_kvhead = Q.shape[-2] // num_kv_heads  # q_per_kv = H / Hkv

    # -----------------------------
    # Q → Qh  (shape: (Hkv, qpkv, S, D))
    # -----------------------------
    # 1. Move head rows into the stream (tile rows become 1)
    q0 = retile_streamify(Q, chunk=1, split_row=True)
    # 2. Split combined stream (S * H) into (H, S)
    q1 = reshape_stream(q0, chunk_size=seq_len, rank=0)
    # 3. Split head dimension H into (Hkv, qpkv)
    q2 = reshape_stream(q1, chunk_size=query_per_kvhead, rank=1)
    # 4. Merge the sequence stream dimension into tile rows
    Qh = accum_retile_row(q2, rank=1)

    # -----------------------------
    # K → Kh  (shape: (Hkv, 1, S, D))
    # -----------------------------
    k0 = retile_streamify(K, chunk=1, split_row=True)
    k1 = reshape_stream(k0, chunk_size=seq_len, rank=0)   # (Hkv, S)
    k2 = accum_retile_row(k1, rank=1)                    # tile rows = S, stream = (Hkv,)
    Kh = reshape_stream(k2, chunk_size=1, rank=0)         # add singleton stream dim → (Hkv, 1)

    # -----------------------------
    # V → Vh  (shape: (Hkv, 1, S, D))
    # -----------------------------
    v0 = retile_streamify(V, chunk=1, split_row=True)
    v1 = reshape_stream(v0, chunk_size=seq_len, rank=0)   # (Hkv, S)
    v2 = accum_retile_row(v1, rank=1)                    # tile rows = S, stream = (Hkv,)
    Vh = reshape_stream(v2, chunk_size=1, rank=0)         # add singleton stream dim → (Hkv, 1)

    return Qh, Kh, Vh