def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Problem dimensions (plain Python scalars)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32

    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads   # 4

    # ------------------------------------------------------------------
    # Build Qh = (num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    # 1) Split the tile‑row (heads) into the stream.
    Qh = retile_streamify(Q, chunk=1, split_row=True)                 # (S*H)×tile(1,D)

    # 2) Split the flat stream into (num_kv_heads*seq_len, query_per_kvhead)
    Qh = reshape_stream(Qh, chunk_size=query_per_kvhead, rank=0)      # (S*kv, qp)×tile(1,D)

    # 3) Split the first stream dimension into (num_kv_heads, seq_len)
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=1)               # (kv, S, qp)×tile(1,D)

    # 4) Merge the sequence and qp dimensions (produces kv × (S*qp))
    Qh = flatten(Qh, min_rank=0, max_rank=1)                         # (kv, S*qp)×tile(1,D)

    # 5) Split that combined dimension back into (query_per_kvhead, seq_len)
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=0)              # (kv, qp, S)×tile(1,D)

    # 6) Move the sequence dimension into the tile rows.
    Qh = accum_retile_row(Qh, rank=1)                                 # (kv, qp)×tile(S,D)

    # ------------------------------------------------------------------
    # Build Kh and Vh = (num_kv_heads, 1) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def make_kv(x):
        kv = retile_streamify(x, chunk=1, split_row=True)             # (S*kv)×tile(1,D)
        kv = reshape_stream(kv, chunk_size=seq_len, rank=0)          # (kv, S)×tile(1,D)
        kv = accum_retile_row(kv, rank=1)                            # (kv,)×tile(S,D)
        kv = reshape_stream(kv, chunk_size=1, rank=0)                # (kv,1)×tile(S,D)
        return kv

    Kh = make_kv(K)   # (kv,1)×tile(S,D)
    Vh = make_kv(V)   # (kv,1)×tile(S,D)

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
    )   # → stream(kv, qp) × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # Inverse transformation: back to (seq_len,) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    # 1) Pull the sequence dimension out of the tile rows.
    a = retile_streamify(attn, chunk=1, split_row=True)               # (kv, qp*seq)×tile(1,D)

    # 2) Separate the combined dimension into (qp, seq).
    a = reshape_stream(a, chunk_size=seq_len, rank=0)                  # (kv, qp, seq)×tile(1,D)

    # 3) Merge the kv and qp dimensions → (heads, seq) stream.
    a = flatten(a, min_rank=1, max_rank=2)                            # (heads, seq)×tile(1,D)

    # 4) Collapse both stream dimensions into a flat stream.
    a = flatten(a, min_rank=0, max_rank=1)                            # (heads*seq)×tile(1,D)

    # 5) Reshape into (seq_len, num_heads) stream.
    a = reshape_stream(a, chunk_size=num_heads, rank=0)               # (seq, heads)×tile(1,D)

    # 6) Move the head dimension into the tile rows.
    out = accum_retile_row(a, rank=1)                                 # (seq,)×tile(heads, D)

    return out