def prepare_qkv(Q, K, V, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Reasoning:
    #   The PyTorch reference does:
    #       Qh = Q.view(S, Hkv, qpkv, D).permute(1, 2, 0, 3)   -> (Hkv, qpkv, S, D)
    #       Kh = K.permute(1, 0, 2).unsqueeze(1)                -> (Hkv, 1, S, D)
    #       Vh = V.permute(1, 0, 2).unsqueeze(1)                -> (Hkv, 1, S, D)
    #
    #   In the DSL a stream tensor has shape (*stream, tile_r, tile_c).
    #   Q:   (S=64, tile_r=16, tile_c=32)
    #   K/V: (S=64, tile_r= 4, tile_c=32)
    #
    #   We need to:
    #     * split the stream dimension (64) into (num_heads, query_per_kvhead)
    #       where num_heads = 16 and query_per_kvhead = 4.
    #     * further split num_heads into (num_kv_heads, query_per_kvhead)
    #       where num_kv_heads = 4.
    #     * merge the innermost stream dimension (query_per_kvhead) into the
    #       tile‑row dimension to obtain tile rows = S (=64).
    #   For K/V we also have to lift the original tile‑row dimension (4) into a
    #   stream dimension, then merge the two query_per_kvhead dimensions into the
    #   tile rows, and finally add a dummy stream dimension of size 1.
    #
    #   The DSL provides:
    #       reshape_stream   – split a stream dimension into two stream dims.
    #       accum_retile_row – merge the last stream dimension into tile rows.
    #   Using these primitives we can express the exact view+permute logic.
    # ----------------------------------------------------------------------

    # ---- compute scalar helper values ------------------------------------
    num_heads = Q.shape[1]                 # 16
    num_kv_heads = K.shape[1]              # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ---- Q -> Qh ---------------------------------------------------------
    # 1. split stream (64) into (num_heads, query_per_kvhead) -> (16,4,16,32)
    q_step1 = reshape_stream(Q, chunk_size=query_per_kvhead, rank=0)
    # 2. split the first stream dim (num_heads) into (num_kv_heads, query_per_kvhead)
    #    -> (4,4,4,16,32)
    q_step2 = reshape_stream(q_step1, chunk_size=num_kv_heads, rank=1)
    # 3. merge the innermost stream dim (query_per_kvhead) into tile rows
    #    -> (4,4,64,32)  which is the desired Qh
    Qh = accum_retile_row(q_step2, rank=1)

    # ---- K -> Kh ---------------------------------------------------------
    # 1. split stream (64) into (num_heads, query_per_kvhead) -> (16,4,4,32)
    k_step1 = reshape_stream(K, chunk_size=query_per_kvhead, rank=0)
    # 2. split the first stream dim (num_heads) into (num_kv_heads, query_per_kvhead)
    #    -> (4,4,4,4,32)
    k_step2 = reshape_stream(k_step1, chunk_size=num_kv_heads, rank=1)
    # 3. merge the two query_per_kvhead stream dims into tile rows
    #    (rank=2 performs both merges) -> (4,64,32)
    k_step3 = accum_retile_row(k_step2, rank=2)
    # 4. introduce the required singleton stream dim -> (4,1,64,32)
    Kh = reshape_stream(k_step3, chunk_size=1, rank=0)

    # ---- V -> Vh ---------------------------------------------------------
    v_step1 = reshape_stream(V, chunk_size=query_per_kvhead, rank=0)
    v_step2 = reshape_stream(v_step1, chunk_size=num_kv_heads, rank=1)
    v_step3 = accum_retile_row(v_step2, rank=2)
    Vh = reshape_stream(v_step3, chunk_size=1, rank=0)

    return Qh, Kh, Vh