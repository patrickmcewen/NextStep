def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # --------------------------------------------------------------
    # Basic dimensions (plain Python scalars)
    # --------------------------------------------------------------
    seq_len          = Q.shape[0]               # 64
    num_heads        = Q.shape[1]               # 16
    head_dim         = Q.shape[2]               # 32

    num_kv_heads     = K.shape[1]               # 4
    query_per_kvhead = num_heads // num_kv_heads   # 4

    # --------------------------------------------------------------
    # Build Qh : (num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
    # --------------------------------------------------------------
    # 1) Turn the per‑token (head‑row) tile into a stream.
    Q_flat = retile_streamify(Q, chunk=1, split_row=True)          # → stream(seq_len* num_heads)×tile(1, head_dim)

    # 2) Split that flat stream back into individual heads (round‑robin).
    per_head = parallelize(Q_flat, n=num_heads)                    # list of `num_heads` tensors, each shape:
                                                                  #   stream(seq_len)×tile(1, head_dim)

    # 3) Concatenate the per‑head streams so that each head appears in a contiguous block.
    merged_q, _ = eager_merge(per_head)                             # → stream(num_heads * seq_len)×tile(1, head_dim)

    # 4) Reshape to (head, seq) stream.
    Q_hs = reshape_stream(merged_q, chunk_size=seq_len, rank=0)    # → stream(num_heads, seq_len)×tile(1, head_dim)

    # 5) Merge the inner stream (seq) into the tile rows → tile rows = seq_len.
    Q_tile = accum_retile_row(Q_hs, rank=1)                         # → stream(num_heads)×tile(seq_len, head_dim)

    # 6) Split the head stream into (num_kv_heads, query_per_kvhead).
    Qh = reshape_stream(Q_tile, chunk_size=query_per_kvhead, rank=0)  # → stream(num_kv_heads, query_per_kvhead)×tile(seq_len, head_dim)

    # --------------------------------------------------------------
    # Build Kh and Vh : (num_kv_heads, 1) × tile(seq_len, head_dim)
    # --------------------------------------------------------------
    def make_kv(x):
        # 1) Move the tile‑row dimension (kv‑heads) into the stream.
        flat = retile_streamify(x, chunk=1, split_row=True)       # → stream(seq_len * num_kv_heads)×tile(1, head_dim)

        # 2) Split the flat stream into per‑kv‑head streams.
        per_kv = parallelize(flat, n=num_kv_heads)                # list of `num_kv_heads` tensors,
                                                                  # each shape stream(seq_len)×tile(1, head_dim)

        # 3) Concatenate the per‑kv streams (kv‑major order).
        merged, _ = eager_merge(per_kv)                            # → stream(num_kv_heads * seq_len)×tile(1, head_dim)

        # 4) Reshape to (kv, seq) stream.
        kv_seq = reshape_stream(merged, chunk_size=seq_len, rank=0)  # → stream(num_kv_heads, seq_len)×tile(1, head_dim)

        # 5) Merge seq into tile rows.
        kv_tile = accum_retile_row(kv_seq, rank=1)                 # → stream(num_kv_heads)×tile(seq_len, head_dim)

        # 6) Add the required trailing singleton stream dimension.
        kv_tile = reshape_stream(kv_tile, chunk_size=1, rank=0)    # → stream(num_kv_heads, 1)×tile(seq_len, head_dim)
        return kv_tile

    Kh = make_kv(K)   # (num_kv_heads, 1)×tile(seq_len, head_dim)
    Vh = make_kv(V)   # (num_kv_heads, 1)×tile(seq_len, head_dim)

    # --------------------------------------------------------------
    # Heavy attention kernel (blackbox)
    # --------------------------------------------------------------
    attn = attention_compute__root_attention_attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=(
            (num_kv_heads, query_per_kvhead, seq_len, head_dim),   # vanilla shape expected by the child
        ),
        out_perms=(None,),
    )   # → stream(num_kv_heads, query_per_kvhead)×tile(seq_len, head_dim)

    # --------------------------------------------------------------
    # Convert the child's output back to the contract shape:
    #   stream(seq_len) × tile(num_heads, head_dim)
    # --------------------------------------------------------------
    # 1) Collapse the (kv, qp) stream dimensions into a single head stream.
    attn_heads = flatten(attn, min_rank=0, max_rank=1)              # → stream(num_heads)×tile(seq_len, head_dim)

    # 2) Split the tile‑row dimension (seq_len) into a stream dimension.
    split = retile_streamify(attn_heads, chunk=1, split_row=True)   # → stream(num_heads, seq_len)×tile(1, head_dim)

    # 3) Re‑group tokens so that each stream corresponds to a single sequence position
    #    (i.e. turn (head, seq) → list of `seq_len` streams each of length `num_heads`).
    per_seq = parallelize(split, n=seq_len)                         # list of `seq_len` tensors,
                                                                  # each shape stream(num_heads)×tile(1, head_dim)

    # 4) Concatenate the per‑sequence streams back into a flat stream.
    merged_attn, _ = eager_merge(per_seq)                           # → stream(seq_len * num_heads)×tile(1, head_dim)

    # 5) Reshape to (seq_len, num_heads) stream.
    seq_head = reshape_stream(merged_attn, chunk_size=num_heads, rank=0)  # → stream(seq_len, num_heads)×tile(1, head_dim)

    # 6) Move the head dimension into the tile rows.
    out = accum_retile_row(seq_head, rank=1)                       # → stream(seq_len)×tile(num_heads, head_dim)

    return out