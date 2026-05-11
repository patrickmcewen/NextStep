def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Dimension bookkeeping (pure Python)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32

    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads   # 4

    # ------------------------------------------------------------------
    # Build Qh : (num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    # Split the tile‑row (head) dimension into the stream.
    Qh = retile_streamify(Q, chunk=1, split_row=True)                 # (S*H)×tile(1,D)

    # Regroup into (head, seq) stream, then split the head stream into
    # (kv, qp, seq) and finally move seq into the tile rows.
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=0)               # (H, S)×tile(1,D)
    Qh = reshape_stream(Qh, chunk_size=query_per_kvhead, rank=1)      # (kv, qp, S)×tile(1,D)
    Qh = flatten(Qh, min_rank=0, max_rank=1)                         # merge kv & qp → (heads, S)×tile(1,D)
    Qh = accum_retile_row(Qh, rank=1)                                # (kv, qp)×tile(S,D)

    # ------------------------------------------------------------------
    # Helper to produce Kh / Vh  : (num_kv_heads, 1) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def make_kv(x):
        # Split the tile‑row (kv) dimension into the stream.
        y = retile_streamify(x, chunk=1, split_row=True)              # (S*kv)×tile(1,D)

        # Regroup into (kv, seq) stream.
        y = reshape_stream(y, chunk_size=seq_len, rank=0)             # (kv, S)×tile(1,D)

        # Move the seq dimension into the tile rows.
        y = accum_retile_row(y, rank=1)                               # (kv,)×tile(S,D)

        # Add the required singleton stream dimension.
        y = reshape_stream(y, chunk_size=1, rank=0)                   # (kv,1)×tile(S,D)
        return y

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
    )   # → stream(kv, qp) × tile(seq, dim)

    # ------------------------------------------------------------------
    # Transpose (kv*qp) stream → seq stream, then merge heads into tile rows.
    # ------------------------------------------------------------------
    # 1) Split the seq‑dimension (tile rows) into the stream, producing a
    #    list where each element corresponds to one sequence position and
    #    contains all heads for that position.
    per_seq = parallelize(attn, n=seq_len)          # list of `seq_len` tensors,
                                                   # each shape: stream(num_heads)×tile(1,dim)

    # 2) For each sequence token, turn the head‑stream into tile rows.
    per_seq_tiles = [
        repeat_static(accum_retile_row(s, rank=1), factor=1)   # (1,)×tile(num_heads, dim)
        for s in per_seq
    ]

    # 3) Concatenate the per‑token tiles back into a single stream.
    out, _ = eager_merge(per_seq_tiles)            # shape: (seq_len,)×tile(num_heads, dim)

    return out