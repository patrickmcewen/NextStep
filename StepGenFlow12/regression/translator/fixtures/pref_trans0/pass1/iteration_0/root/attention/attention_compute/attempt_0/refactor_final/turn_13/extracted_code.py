def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalar dimensions (pure Python)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32

    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads   # 4

    # ------------------------------------------------------------------
    # Build Qh : (num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    # Split the tile‑row (heads) into the stream.
    Qh = retile_streamify(Q, chunk=1, split_row=True)                 # → stream(1024,)×tile(1,D)

    # Regroup into (head, seq) stream.
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=0)              # → stream(16,64)×tile(1,D)

    # Split the head stream into (kv, qp, seq).
    Qh = reshape_stream(Qh, chunk_size=query_per_kvhead, rank=1)     # → stream(4,4,64)×tile(1,D)

    # Move the sequence dimension into the tile rows.
    Qh = accum_retile_row(Qh, rank=1)                                 # → stream(4,4)×tile(64,D)

    # ------------------------------------------------------------------
    # Build Kh and Vh : (num_kv_heads, 1) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def make_kv(x):
        kv = retile_streamify(x, chunk=1, split_row=True)            # → stream(S*kv)×tile(1,D)
        kv = reshape_stream(kv, chunk_size=seq_len, rank=0)         # → stream(kv, S)×tile(1,D)
        kv = accum_retile_row(kv, rank=1)                           # → stream(kv,)×tile(S,D)
        kv = reshape_stream(kv, chunk_size=1, rank=0)               # → stream(kv,1)×tile(S,D)
        return kv

    Kh = make_kv(K)   # → stream(4,1)×tile(64,D)
    Vh = make_kv(V)   # → stream(4,1)×tile(64,D)

    # ------------------------------------------------------------------
    # Heavy attention kernel (blackbox)
    # ------------------------------------------------------------------
    attn = attention_compute__root_attention_attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=(
            (num_kv_heads, query_per_kvhead, seq_len, head_dim),   # vanilla shape expected by the child
        ),
        out_perms=(None,),
    )   # → stream(4,4)×tile(64,D)

    # ------------------------------------------------------------------
    # Inverse transformation: back to (seq_len,) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    # 1) Split the tile‑row (seq_len) into a stream dimension.
    a = retile_streamify(attn, chunk=1, split_row=True)              # → stream(4,256)×tile(1,D)

    # 2) Separate qp*seq into (qp, seq) stream dimensions.
    a = reshape_stream(a, chunk_size=seq_len, rank=0)                # → stream(4,4,64)×tile(1,D)

    # 3) Merge the kv and qp dimensions → (heads, seq).
    a = flatten(a, min_rank=1, max_rank=2)                           # → stream(16,64)×tile(1,D)

    # 4) Flatten heads and seq into a single stream.
    a = flatten(a, min_rank=0, max_rank=1)                           # → stream(1024,)×tile(1,D)

    # 5) Reshape to (seq_len, heads) stream ordering.
    a = reshape_stream(a, chunk_size=num_heads, rank=0)              # → stream(64,16)×tile(1,D)

    # 6) Move the heads dimension into the tile rows.
    out = accum_retile_row(a, rank=1)                                 # → stream(64,)×tile(16,32)

    return out