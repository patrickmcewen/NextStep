def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scaler‑free dimensions
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32

    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads   # 4

    # ------------------------------------------------------------------
    # Build Qh = (num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    # Split the head dimension (tile rows) into stream tokens.
    Qh = retile_streamify(Q, chunk=1, split_row=True)                 # → stream(1024,)×tile(1,32)
    # Reshape the flat stream into (head, seq) where head = num_heads.
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=0)              # → stream(16,64)×tile(1,32)
    # Split the head stream into (kv, q) dimensions.
    Qh = reshape_stream(Qh, chunk_size=query_per_kvhead, rank=1)     # → stream(4,4,64)×tile(1,32)
    # Move the seq dimension from the tile back onto the tile rows.
    Qh = accum_retile_row(Qh, rank=1)                                # → stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # Build Kh and Vh = (num_kv_heads, 1) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def make_kv(x):
        kv = retile_streamify(x, chunk=1, split_row=True)            # → stream(256,)×tile(1,32)
        kv = reshape_stream(kv, chunk_size=seq_len, rank=0)          # → stream(4,64)×tile(1,32)
        kv = accum_retile_row(kv, rank=1)                           # → stream(4,)×tile(64,32)
        kv = reshape_stream(kv, chunk_size=1, rank=0)                # → stream(4,1)×tile(64,32)
        return kv

    Kh = make_kv(K)    # stream(4,1)×tile(64,32)
    Vh = make_kv(V)    # stream(4,1)×tile(64,32)

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
    )   # → stream(num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # Inverse transformation: back to (seq_len,) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    # 1) Split the seq_len tile rows into the stream.
    x = retile_streamify(attn, chunk=1, split_row=True)                # → stream(kv, q*seq)×tile(1, head_dim)

    # 2) Separate the combined (q * seq) dimension into distinct q and seq axes.
    x = reshape_stream(x, chunk_size=seq_len, rank=0)                 # → stream(kv, q, seq)×tile(1, head_dim)

    # 3) Merge the kv and q axes to recover the original head stream.
    x = flatten(x, min_rank=1, max_rank=2)                            # → stream(num_heads, seq)×tile(1, head_dim)

    # 4) Merge the remaining two stream dimensions into a single flat stream.
    x = flatten(x, min_rank=0, max_rank=1)                            # → stream(num_heads * seq)×tile(1, head_dim)

    # 5) Reshape the flat stream back to (seq_len, num_heads) ordering.
    x = reshape_stream(x, chunk_size=num_heads, rank=0)               # → stream(seq_len, num_heads)×tile(1, head_dim)

    # 6) Pull the head dimension out of the stream and into the tile rows.
    out = accum_retile_row(x, rank=1)                                 # → stream(seq_len)×tile(num_heads, head_dim)

    return out