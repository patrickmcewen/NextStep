def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalar dimensions (pure Python)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                 # 64
    num_heads = Q.shape[1]               # 16
    head_dim = Q.shape[2]                # 32

    num_kv_heads = K.shape[1]            # 4
    query_per_kvhead = num_heads // num_kv_heads   # 4

    # ------------------------------------------------------------------
    # Build Qh = (num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    # 1) Split the tile‑row dimension (num_heads) into the stream.
    Q_flat = retile_streamify(Q, chunk=1, split_row=True)          # → stream(1024,)×tile(1,32)
    # 2) Regroup into (num_heads, seq_len) stream.
    Q_seq = reshape_stream(Q_flat, chunk_size=seq_len, rank=0)    # → stream(16,64)×tile(1,32)
    # 3) Split the head stream into (kv, qp, seq_len).
    Q_kvqp = reshape_stream(Q_seq, chunk_size=query_per_kvhead, rank=1)  # → stream(4,4,64)×tile(1,32)
    # 4) Move the seq_len dimension into the tile rows.
    Qh = accum_retile_row(Q_kvqp, rank=2)                        # → stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # Build Kh and Vh = (num_kv_heads, 1) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def make_kv(x):
        # Split the tile‑row (kv‑heads) into the stream.
        flat = retile_streamify(x, chunk=1, split_row=True)          # → stream(256,)×tile(1,32)
        # Regroup into (kv, seq_len) stream.
        seq = reshape_stream(flat, chunk_size=seq_len, rank=0)       # → stream(4,64)×tile(1,32)
        # Move seq_len into the tile rows.
        tiled = accum_retile_row(seq, rank=1)                        # → stream(4,)×tile(64,32)
        # Add a trailing singleton stream dim to match the child’s expected shape.
        return reshape_stream(tiled, chunk_size=1, rank=0)           # → stream(4,1)×tile(64,32)

    Kh = make_kv(K)   # → stream(4,1)×tile(64,32)
    Vh = make_kv(V)   # → stream(4,1)×tile(64,32)

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
    )   # → stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # Convert the child’s output back to (seq_len,) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    # 1) Pull the seq_len dimension out of the tile rows.
    attn_flat = retile_streamify(attn, chunk=1, split_row=True)   # → stream(4,4,64)×tile(1,32)

    # 2) Merge the KV‑head dimensions (kv, qp) → head.
    merged_head_seq = flatten(attn_flat, min_rank=1, max_rank=2)   # → stream(16,64)×tile(1,32)

    # 3) Collapse the remaining two stream dimensions (head, seq) into one.
    merged_all = flatten(merged_head_seq, min_rank=0, max_rank=1)  # → stream(1024,)×tile(1,32)

    # 4) Split the flat stream into (seq_len, num_heads).
    seq_head = reshape_stream(merged_all, chunk_size=num_heads, rank=0)  # → stream(64,16)×tile(1,32)

    # 5) Move the head dimension into the tile rows.
    out = accum_retile_row(seq_head, rank=0)                       # → stream(64,)×tile(16,32)

    return out