# Implementation notes:
# - The parent passes on‑chip streams Q, K, V with shapes:
#     Q : stream(seq_len) × tile(num_heads, head_dim)
#     K : stream(seq_len) × tile(num_kv_heads, head_dim)
#     V : stream(seq_len) × tile(num_kv_heads, head_dim)
# - The heavy‑weight attention blackbox expects the GQA‑reshaped tensors:
#     Qh : stream(num_kv_heads, query_per_kv) × tile(seq_len, head_dim)
#     Kh : stream(num_kv_heads, 1)            × tile(seq_len, head_dim)
#     Vh : stream(num_kv_heads, 1)            × tile(seq_len, head_dim)
#   The following DSL pipeline builds exactly those layouts.
#   * For Q we split each row into a separate token, parallelize across heads,
#     then re‑assemble each head’s rows into a single tile (seq_len rows).
#   * For K and V we parallelize across KV‑heads and re‑assemble similarly,
#     adding a singleton stream dimension afterwards.
# - After the blackbox produces the attention result (stream(kv, q_per_kv) ×
#   tile(seq_len, head_dim)), we invert the transformation to obtain the
#   contract‑shaped output (stream(seq_len) × tile(num_heads, head_dim)).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Basic dimensions (plain Python scalars)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]               # 64
    num_heads = Q.shape[1]             # 16
    head_dim = Q.shape[2]              # 32
    num_kv_heads = K.shape[1]          # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # Build Qh = (num_kv_heads, query_per_kv) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    # 1) Split each row (head) into its own token.
    Q_rows = retile_streamify(Q, chunk=1, split_row=True)          # stream(64*16)×tile(1,32)

    # 2) Parallelize across heads → list of 16 streams, each (seq_len,)×tile(1,32).
    Q_per_head = parallelize(Q_rows, n=num_heads)

    # 3) For each head, collapse its stream (seq_len) into the tile rows.
    Q_head_tiles = []
    for h in Q_per_head:
        # merge seq_len into tile rows → tile(seq_len, head_dim), no stream dims
        t = accum_retile_row(h, rank=1)
        # add a leading singleton stream dimension so static_reassemble can interleave
        t = promote(t, rank=1)            # stream(1,)×tile(seq_len, head_dim)
        Q_head_tiles.append(t)

    # 4) Interleave the 16 head tiles → stream(num_heads)×tile(seq_len, head_dim)
    Q_combined = static_reassemble(Q_head_tiles)                 # stream(16,)×tile(64,32)

    # 5) Split the head stream into (kv, query_per_kv) dimensions.
    Qh = reshape_stream(Q_combined, chunk_size=query_per_kvhead, rank=0)  # stream(4,4)×tile(64,32)

    # ------------------------------------------------------------------
    # Build Kh and Vh = (num_kv_heads, 1) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def make_kv_stream(x):
        # 1) Split each row into its own token.
        rows = retile_streamify(x, chunk=1, split_row=True)      # stream(seq_len*num_kv)×tile(1,32)

        # 2) Parallelize across KV heads.
        per_kv = parallelize(rows, n=num_kv_heads)               # list of kv tensors (seq_len,)×tile(1,32)

        kv_tiles = []
        for kv in per_kv:
            # merge seq_len into tile rows → tile(seq_len, head_dim)
            t = accum_retile_row(kv, rank=1)                     # tile(seq_len, head_dim)
            # add leading singleton stream dim
            t = promote(t, rank=1)                               # stream(1,)×tile(seq_len, head_dim)
            kv_tiles.append(t)

        # 3) Interleave the KV‑head tiles → stream(num_kv_heads,)×tile(seq_len, head_dim)
        combined = static_reassemble(kv_tiles)                    # stream(4,)×tile(64,32)

        # 4) Add the required singleton dimension after the KV axis.
        return reshape_stream(combined, chunk_size=1, rank=0)      # stream(4,1)×tile(64,32)

    Kh = make_kv_stream(K)       # stream(4,1)×tile(64,32)
    Vh = make_kv_stream(V)       # stream(4,1)×tile(64,32)

    # ------------------------------------------------------------------
    # Heavy attention compute (blackbox)
    # ------------------------------------------------------------------
    attn = attention_compute__root_attention_attention_compute(
        Qh,
        Kh,
        Vh,
        out_shapes=(
            (num_kv_heads, query_per_kvhead, seq_len, head_dim),  # vanilla shape of the output
        ),
        out_perms=(None,),
    )

    # ------------------------------------------------------------------
    # Convert the blackbox output back to contract shape:
    #   stream(seq_len) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    # 1) Split tile rows (seq_len) into a stream dimension.
    a1 = retile_streamify(attn, chunk=1, split_row=True)          # stream(4,256)×tile(1,32)

    # 2) Recover the (kv, q_per_kv, seq_len) stream layout.
    a2 = reshape_stream(a1, chunk_size=seq_len, rank=0)           # stream(4,4,64)×tile(1,32)

    # 3) Merge the first two stream dimensions (kv & q_per_kv) → heads.
    a3 = flatten(a2, min_rank=1, max_rank=2)                      # stream(16,64)×tile(1,32)

    # 4) Collapse both remaining stream dimensions into one.
    a4 = flatten(a3, min_rank=0, max_rank=1)                      # stream(1024,)×tile(1,32)

    # 5) Split the flat stream into (seq_len, num_heads) order.
    a5 = reshape_stream(a4, chunk_size=num_heads, rank=0)        # stream(64,16)×tile(1,32)

    # 6) Merge the head stream into the tile rows.
    out = accum_retile_row(a5, rank=1)                           # stream(64,)×tile(16,32)

    return out