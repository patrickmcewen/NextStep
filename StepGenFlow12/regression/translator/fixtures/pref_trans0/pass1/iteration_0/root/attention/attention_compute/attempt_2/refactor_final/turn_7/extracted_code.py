# attention_compute ---------------------------------------------------------
# Transform the on‑chip inputs Q, K, V from (seq, heads, dim) into the
# GQA layout required by the heavy‑weight child, invoke the child, and then
# reshape the result back to (seq, heads, dim).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0️⃣  Scalars (pure Python – allowed)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # 1️⃣  Convert a (seq, heads, dim) stream into
    #      (kv, q_per_kv) × tile(seq, dim) – the exact layout the child
    #      expects.  The sequence of DSL ops mirrors the
    #      reference implementation:
    #        Q → Q.view(seq, kv, q_per_kv, dim).permute(1,2,0,3)
    # ------------------------------------------------------------------
    def _to_gqa(tensor, q_per_kv):
        # tensor: stream(seq_len,) × tile(num_heads, head_dim)
        #   1) split the tile‑row (heads) into chunks of size q_per_kv,
        #      moving the chunk count into the stream.
        x = retile_streamify(tensor, chunk=q_per_kv, split_row=True)
        #   2) split the combined stream (seq_len * (heads/q_per_kv))
        #      into (kv, seq_len) while restoring tile rows = q_per_kv.
        x = reshape_stream(x, chunk_size=seq_len, rank=0)
        #   3) merge the seq_len stream dimension into the tile rows.
        x = accum_retile_row(x)
        #   4) split that large tile‑row back into seq_len‑sized chunks,
        #      promoting the chunk index to the second stream dimension.
        x = retile_streamify(x, chunk=seq_len, split_row=True)
        #   5) finally split the merged stream (kv * q_per_kv) into the
        #      two desired stream dimensions.
        x = reshape_stream(x, chunk_size=q_per_kv, rank=0)
        return x

    # ------------------------------------------------------------------
    # 2️⃣  Build the tensors in the child’s expected layout.
    # ------------------------------------------------------------------
    Qh = _to_gqa(Q, query_per_kvhead)   # (kv, q_per_kv) × tile(seq, dim)
    Kh = _to_gqa(K, 1)                  # (kv, 1)       × tile(seq, dim)
    Vh = _to_gqa(V, 1)                  # (kv, 1)       × tile(seq, dim)

    # ------------------------------------------------------------------
    # 3️⃣  Heavy‑weight attention (the child).  Its output vanilla shape is
    #     (kv, q_per_kv, seq, dim); we request that shape via out_shapes.
    # ------------------------------------------------------------------
    child_out_shape = ((num_kv_heads, query_per_kvhead, seq_len, head_dim),)
    attn_raw = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=child_out_shape, out_perms=None
    )   # stream(kv, q_per_kv) × tile(seq, dim)

    # ------------------------------------------------------------------
    # 4️⃣  Collapse the two GQA stream dimensions back into a single
    #     “heads” stream dimension.
    # ------------------------------------------------------------------
    attn_flat = flatten(attn_raw, min_rank=0, max_rank=1)   # stream(num_heads) × tile(seq, dim)

    # ------------------------------------------------------------------
    # 5️⃣  Swap the stream (heads) with the tile‑row (seq) so that the
    #     final layout matches the contract: stream(seq) × tile(num_heads, dim).
    # ------------------------------------------------------------------
    #   - retile_streamify splits the tile‑row (currently seq) into chunks of
    #     size = num_heads, turning each chunk into a new stream element.
    #   - Because tile‑row = seq and chunk = num_heads, the resulting stream
    #     dimension length is seq, and the new tile‑row length becomes num_heads.
    attn_out = retile_streamify(attn_flat, chunk=num_heads, split_row=True)

    # ------------------------------------------------------------------
    # 6️⃣  Return the stream tensor that the root will store off‑chip.
    # ------------------------------------------------------------------
    return attn_out