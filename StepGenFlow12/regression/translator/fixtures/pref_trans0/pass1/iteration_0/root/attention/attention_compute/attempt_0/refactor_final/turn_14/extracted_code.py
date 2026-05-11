def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalar dimensions (pure Python)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32

    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads   # 4
    heads = num_heads                  # kv * query_per_kvhead

    # ------------------------------------------------------------------
    # Build Qh : (num_kv_heads, query_per_kvhead) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    # 1) Split the tile‑row dimension (heads) into a stream.
    Qh = retile_streamify(Q, chunk=1, split_row=True)                 # → stream(1024,)×tile(1,D)

    # 2) Separate that stream into (num_kv_heads, query_per_kvhead, seq_len)
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=0)              # → stream(16,64)×tile(1,D)
    Qh = reshape_stream(Qh, chunk_size=query_per_kvhead, rank=1)     # → stream(4,4,64)×tile(1,D)

    # 3) Merge the (kv, qp) axes → heads stream, keep seq as tile rows.
    Qh = flatten(Qh, min_rank=0, max_rank=1)                         # → stream(16,64)×tile(1,D)

    # 4) Move the sequence dimension into the tile rows.
    Qh = reshape_stream(Qh, chunk_size=seq_len, rank=0)              # → stream(64,16)×tile(1,D)
    Qh = accum_retile_row(Qh, rank=1)                                # → stream(4,4)×tile(64,D)

    # ------------------------------------------------------------------
    # Helper to produce Kh / Vh : (num_kv_heads, 1) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def make_kv(x):
        # Split tile‑row (heads) into a stream.
        y = retile_streamify(x, chunk=1, split_row=True)            # → stream(256,)×tile(1,D)

        # Separate into (kv, seq) streams.
        y = reshape_stream(y, chunk_size=seq_len, rank=0)           # → stream(4,64)×tile(1,D)

        # Move the sequence dimension into the tile rows.
        y = accum_retile_row(y, rank=1)                             # → stream(4,)×tile(64,D)

        # Add the required trailing singleton stream dimension.
        y = reshape_stream(y, chunk_size=1, rank=0)                 # → stream(4,1)×tile(64,D)
        return y

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
    # Convert the child's output back to the contract shape:
    #   stream(seq_len) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    # 1) Collapse the two head‑related stream dimensions into a single heads stream.
    attn_heads = flatten(attn, min_rank=0, max_rank=1)               # → stream(16,)×tile(64,D)

    # 2) Split the heads stream into a list of per‑head tensors.
    per_head = parallelize(attn_heads, n=heads)                     # list of `heads` tensors,
                                                                     # each shape (1,)×tile(64,D)

    # 3) For each head, split the tile‑row (seq_len) into the stream.
    per_head_rows = [
        retile_streamify(h, chunk=1, split_row=True)               # → stream(64,)×tile(1,D)
        for h in per_head
    ]

    # 4) Interleave the per‑head rows back into a single stream
    #    (round‑robin ordering yields (seq, head) layout).
    interleaved = static_reassemble(per_head_rows)                  # → stream(64*16,)×tile(1,D)

    # 5) Reshape to (seq_len, num_heads) stream.
    interleaved = reshape_stream(interleaved, chunk_size=heads, rank=0)  # → stream(64,16)×tile(1,D)

    # 6) Move the head dimension into the tile rows.
    out = accum_retile_row(interleaved, rank=1)                     # → stream(64,)×tile(16,32)

    return out