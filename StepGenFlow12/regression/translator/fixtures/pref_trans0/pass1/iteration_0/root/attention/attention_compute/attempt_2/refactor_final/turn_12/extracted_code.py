# attention_compute ---------------------------------------------------------
# Build the GQA‑shaped tensors, invoke the heavy‑weight child, and then
# merge the KV‑head stream dimension into the tile‑row dimension so that the
# final layout matches the contract (seq, heads, dim).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0️⃣  Scalar constants (plain Python ints – allowed)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # Helper: reshape a (seq, heads, dim) stream into the GQA layout
    #   (num_kv_heads, query_per_kvhead) × tile(seq, dim)
    # ------------------------------------------------------------------
    def _to_gqa(tensor, q_per_kv):
        # 1) split the head dimension into chunks of size q_per_kv
        x = retile_streamify(tensor, chunk=q_per_kv, split_row=True)
        # 2) split the combined stream (seq * (heads/q_per_kv)) into
        #    (num_kv_heads, seq) while keeping tile rows = q_per_kv
        x = reshape_stream(x, chunk_size=seq_len, rank=0)
        # 3) merge the seq stream dimension into the tile rows
        x = accum_retile_row(x)
        # 4) split the enlarged tile rows back into seq‑sized chunks,
        #    promoting q_per_kv to a new stream dimension
        x = retile_streamify(x, chunk=seq_len, split_row=True)
        # 5) finally split the merged stream (num_kv_heads * q_per_kv) into the
        #    two desired stream dimensions
        x = reshape_stream(x, chunk_size=q_per_kv, rank=0)
        return x

    # ------------------------------------------------------------------
    # 1️⃣  Build Qh, Kh, Vh in the layout the child expects.
    # ------------------------------------------------------------------
    Qh = _to_gqa(Q, query_per_kvhead)   # (num_kv_heads, query_per_kv) × tile(seq, dim)
    Kh = _to_gqa(K, 1)                  # (num_kv_heads, 1)               × tile(seq, dim)
    Vh = _to_gqa(V, 1)                  # (num_kv_heads, 1)               × tile(seq, dim)

    # ------------------------------------------------------------------
    # 2️⃣  Heavy‑weight attention (the child).  We ask the child to emit a
    #     vanilla shape where the sequence dimension is the leading stream
    #     dimension: (seq, num_kv_heads, query_per_kvhead, head_dim).
    #     The child will recover this vanilla shape via flatten‑reshape,
    #     which exactly implements the required permutation.
    # ------------------------------------------------------------------
    child_out_shape = ((seq_len, num_kv_heads, query_per_kvhead, head_dim),)
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=child_out_shape, out_perms=None
    )   # stream(seq, num_kv_heads) × tile(query_per_kvhead, head_dim)

    # ------------------------------------------------------------------
    # 3️⃣  Merge the KV‑head stream dimension into the tile‑row dimension,
    #     yielding (seq) × tile(num_heads, head_dim) as required by the
    #     contract.
    # ------------------------------------------------------------------
    attn = accum_retile_row(attn)    # stream(seq) × tile(num_heads, head_dim)

    # ------------------------------------------------------------------
    # 4️⃣  Return the final stream tensor.
    # ------------------------------------------------------------------
    return attn