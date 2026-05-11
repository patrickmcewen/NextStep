# The attention_compute node builds the GQA‑shaped tensors Qh, Kh, Vh using only
# DSL primitives (no .permute, .reshape, etc.).  It then calls the heavy‑weight
# child `attention_compute__root_attention_attention_compute`, flattens the
# two stream dimensions back into the “heads” dimension and finally swaps the
# stream dimension (heads) with the tile‑row dimension (seq_len) to obtain the
# required output shape (seq_len, num_heads, head_dim).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0️⃣  Extract scalar dimensions (plain Python ints)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # 1️⃣  Helper to convert (seq, heads, dim) → (kv, q_per_kv, seq, dim)
    #     using only DSL ops (no .permute).  The pattern is:
    #       • split the head dimension into chunks (retile_streamify)
    #       • split the resulting stream (reshape_stream) to expose
    #         (kv, seq, q_per_kv, dim)
    #       • swap the stream dimension `seq` with the tile‑row dimension
    #         `q_per_kv` via (accum_retile_row → retile_streamify →
    #         reshape_stream).
    # ------------------------------------------------------------------
    def to_gqa(tensor, q_per_kv):
        # tensor : (seq_len, heads, head_dim)  stream(seq_len)×tile(q_per_kv*kv, head_dim)
        # 1) split heads into chunks of size q_per_kv, moving the chunk count into the stream
        x = retile_streamify(tensor, chunk=q_per_kv, split_row=True)
        # 2) split the (now enlarged) stream into (kv, seq_len) while restoring tile rows = q_per_kv
        x = reshape_stream(x, chunk_size=seq_len, rank=0)
        # 3) swap the stream dim `seq_len` with the tile‑row dim `q_per_kv`
        x = accum_retile_row(x)                         # merge seq_len (last stream dim) into tile rows
        x = retile_streamify(x, chunk=seq_len, split_row=True)  # move seq_len back to stream, tile rows become q_per_kv
        x = reshape_stream(x, chunk_size=q_per_kv, rank=0)       # split merged stream back into (kv, q_per_kv)
        return x

    # ------------------------------------------------------------------
    # 2️⃣  Build Qh, Kh, Vh in the exact layout expected by the child
    # ------------------------------------------------------------------
    Qh = to_gqa(Q, query_per_kvhead)          # (num_kv_heads, query_per_kvhead, seq_len, head_dim)
    Kh = to_gqa(K, 1)                         # (num_kv_heads, 1, seq_len, head_dim)
    Vh = to_gqa(V, 1)                         # (num_kv_heads, 1, seq_len, head_dim)

    # ------------------------------------------------------------------
    # 3️⃣  Call the heavy attention child.  Its own contract expects a
    #     single output with vanilla shape (kv, q_per_kv, seq_len, head_dim).
    # ------------------------------------------------------------------
    child_out_shape = ((num_kv_heads, query_per_kvhead, seq_len, head_dim),)
    attn_raw = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=child_out_shape, out_perms=None
    )   # shape (kv, q_per_kv, seq_len, head_dim)

    # ------------------------------------------------------------------
    # 4️⃣  Collapse the (kv, q_per_kv) stream dimensions into the single
    #     “heads” stream dimension.
    # ------------------------------------------------------------------
    attn_flat = flatten(attn_raw, min_rank=0, max_rank=1)   # (num_heads, seq_len, head_dim)

    # ------------------------------------------------------------------
    # 5️⃣  Swap the stream (heads) with the tile‑row (seq_len) so that the
    #     final layout matches the parent’s expectation:
    #       stream(seq_len) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    attn_out = retile_streamify(attn_flat, chunk=num_heads, split_row=True)

    # ------------------------------------------------------------------
    # 6️⃣  Return the stream tensor; the root will store it off‑chip.
    # ------------------------------------------------------------------
    return attn_out