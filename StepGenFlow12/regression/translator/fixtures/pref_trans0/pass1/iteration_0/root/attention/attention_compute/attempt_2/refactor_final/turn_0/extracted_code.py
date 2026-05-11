# The implementation follows the reference PyTorch logic:
#   1. Reshape Q to Qh by splitting the head dimension into KV‑head groups
#      and queries‑per‑KV‑head, and moving the sequence dimension into the tile.
#   2. Reshape K and V similarly (KV‑heads become a stream dimension, sequence
#      becomes a tile dimension).
#   3. Call the provided child that implements the heavy attention compute.
#   4. Convert the result back to the original (seq, heads, dim) layout.
#   5. Return the stream tensor; the root node will store it off‑chip.
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # 1.  Constants (all are plain Python integers – allowed in the DSL)
    # ----------------------------------------------------------------------
    seq_len = Q.shape[0]                 # 64
    num_heads = Q.shape[1]               # 16
    head_dim = Q.shape[2]                # 32
    num_kv_heads = K.shape[1]            # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ----------------------------------------------------------------------
    # 2.  Transform Q → Qh  (shape: (num_kv_heads, query_per_kvhead, seq_len, head_dim))
    #     * First split the head dimension (tile‑rows) into groups of size
    #       `query_per_kvhead`.  `retile_streamify` moves those groups into the
    #       stream while leaving a tile of size `query_per_kvhead`.
    #     * Then split the combined stream dimension (seq_len × num_kv_heads)
    #       into the KV‑head stream and the sequence stream.
    # ----------------------------------------------------------------------
    Q_retiled = retile_streamify(Q, chunk=query_per_kvhead, split_row=True)
    # Q_retiled : (seq_len * num_kv_heads, query_per_kvhead, head_dim)

    Qh = reshape_stream(Q_retiled, chunk_size=seq_len, rank=0)
    # Qh : (num_kv_heads, seq_len, query_per_kvhead, head_dim)
    # Rearrange the two stream dimensions to match the child’s expectation:
    Qh = Qh.permute(0, 2, 1, 3)  # (num_kv_heads, query_per_kvhead, seq_len, head_dim)

    # ----------------------------------------------------------------------
    # 3.  Transform K → Kh  (shape: (num_kv_heads, 1, seq_len, head_dim))
    #     The KV‑head dimension (tile‑rows) is turned into a stream dimension,
    #     and the sequence length becomes the new tile‑rows.
    # ----------------------------------------------------------------------
    K_retiled = retile_streamify(K, chunk=1, split_row=True)
    # K_retiled : (seq_len * num_kv_heads, 1, head_dim)

    Kh = reshape_stream(K_retiled, chunk_size=seq_len, rank=0)
    # Kh : (num_kv_heads, seq_len, 1, head_dim)
    Kh = Kh.permute(0, 2, 1, 3)  # (num_kv_heads, 1, seq_len, head_dim)

    # ----------------------------------------------------------------------
    # 4.  Transform V → Vh  (same pattern as K)
    # ----------------------------------------------------------------------
    V_retiled = retile_streamify(V, chunk=1, split_row=True)
    Vh = reshape_stream(V_retiled, chunk_size=seq_len, rank=0)
    Vh = Vh.permute(0, 2, 1, 3)  # (num_kv_heads, 1, seq_len, head_dim)

    # ----------------------------------------------------------------------
    # 5.  Call the heavy‑weight attention child.
    # ----------------------------------------------------------------------
    attn = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=out_shapes, out_perms=out_perms
    )
    # attn shape (num_kv_heads, query_per_kvhead, seq_len, head_dim)
    # Stream dims: (num_kv_heads, query_per_kvhead)
    # Tile rows: seq_len , Tile cols: head_dim

    # ----------------------------------------------------------------------
    # 6.  Convert attn back to (seq_len, num_heads, head_dim)
    #     * First merge the two stream dimensions (kv‑head & query‑per‑kv) into a
    #       single “heads” stream dimension.
    #     * Then swap that stream dimension with the tile‑rows (sequence) to
    #       obtain the original layout.
    # ----------------------------------------------------------------------
    # Merge (num_kv_heads, query_per_kvhead) → (num_heads)
    attn_merged = flatten(attn, min_rank=0, max_rank=1)   # (num_heads, seq_len, head_dim)

    # Now we have stream‑dim = num_heads, tile‑rows = seq_len.
    # To obtain the desired stream‑dim = seq_len, tile‑rows = num_heads we
    # split the stream dimension again.
    attn_out = reshape_stream(attn_merged, chunk_size=num_heads, rank=0)
    # attn_out : (seq_len, num_heads, 1, head_dim)

    # Remove the spurious singleton tile‑row dimension.
    # The only way to drop a dimension without using a tensor method is to
    # reinterpret it as the stream dimension by a final `flatten`.
    attn_out = flatten(attn_out, min_rank=0, max_rank=0)  # (seq_len, num_heads, head_dim)

    # ----------------------------------------------------------------------
    # 7.  Return the final stream tensor.
    # ----------------------------------------------------------------------
    return attn_out