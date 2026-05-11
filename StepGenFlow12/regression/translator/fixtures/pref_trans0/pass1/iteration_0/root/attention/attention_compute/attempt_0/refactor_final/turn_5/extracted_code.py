def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Extract scalar dimensions (pure Python)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32

    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # Q → Qh : (kv, q_per_kv) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    Qh = retile_streamify(Q, chunk=1, split_row=True)                     # → stream(64*16)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=0)                  # → stream(16,64)×tile(1,32)
    Qh = reshape_stream(Qh, chunk_size=query_per_kvhead, rank=1)         # → stream(4,4,64)×tile(1,32)
    Qh = accum_retile_row(Qh, rank=1)                                    # → stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # K → Kh : (kv, 1) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    Kh = retile_streamify(K, chunk=1, split_row=True)                     # → stream(64*4)×tile(1,32)
    Kh = reshape_stream(Kh, chunk_size=seq_len, rank=0)                  # → stream(4,64)×tile(1,32)
    Kh = accum_retile_row(Kh, rank=1)                                    # → stream(4)×tile(64,32)
    Kh = reshape_stream(Kh, chunk_size=1, rank=0)                         # → stream(4,1)×tile(64,32)

    # ------------------------------------------------------------------
    # V → Vh : same pattern as K
    # ------------------------------------------------------------------
    Vh = retile_streamify(V, chunk=1, split_row=True)                     # → stream(64*4)×tile(1,32)
    Vh = reshape_stream(Vh, chunk_size=seq_len, rank=0)                  # → stream(4,64)×tile(1,32)
    Vh = accum_retile_row(Vh, rank=1)                                    # → stream(4)×tile(64,32)
    Vh = reshape_stream(Vh, chunk_size=1, rank=0)                         # → stream(4,1)×tile(64,32)

    # ------------------------------------------------------------------
    # Heavy attention kernel (blackbox)
    # ------------------------------------------------------------------
    attn = attention_compute__root_attention_attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=(
            (num_kv_heads, query_per_kvhead, seq_len, head_dim),  # vanilla shape of the output
        ),
        out_perms=(None,),
    )   # → stream(kv, q_per_kv) × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # Convert back to the contract shape: (seq_len,) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    x = retile_streamify(attn, chunk=1, split_row=True)                 # → stream(kv, q_per_kv, seq_len)×tile(1,head_dim)
    x = flatten(x, min_rank=1, max_rank=2)                              # merge kv & q_per_kv → stream(num_heads, seq_len)×tile(1,head_dim)
    x = flatten(x, min_rank=0, max_rank=1)                              # merge both stream dims   → stream(num_heads*seq_len)×tile(1,head_dim)
    x = reshape_stream(x, chunk_size=num_heads, rank=0)                 # → stream(seq_len, num_heads)×tile(1,head_dim)
    out = accum_retile_row(x, rank=1)                                   # → stream(seq_len)×tile(num_heads, head_dim)

    return out