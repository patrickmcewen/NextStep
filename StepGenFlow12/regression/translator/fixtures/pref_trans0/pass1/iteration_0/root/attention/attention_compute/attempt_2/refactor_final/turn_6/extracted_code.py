# Attention_compute node.
#   * Convert Q, K, V from (seq, heads, dim) → GQA layout required by the child:
#       Qh : (num_kv_heads, query_per_kv, seq, dim)
#       Kh/Vh : (num_kv_heads, 1, seq, dim)
#   * Call the heavy‑weight child.
#   * Collapse the (num_kv_heads, query_per_kv) stream dimensions back to a single
#     heads stream and swap that stream with the tile‑row dimension to obtain the
#     contract‑required layout (seq, heads, dim).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalars from the input shapes (plain Python ints – allowed)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # Helper: reshape a (seq, heads) stream into GQA layout.
    #   tensor: stream(seq_len,) × tile(num_heads, head_dim)
    #   q_per_kv: queries‑per‑KV‑head (4 for Q, 1 for K/V)
    #   Returns: stream(num_kv_heads, q_per_kv) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def _to_gqa(tensor, q_per_kv):
        # 1) split the tile‑row dimension (num_heads) into chunks of size q_per_kv
        x = retile_streamify(tensor, chunk=q_per_kv, split_row=True)
        # 2) split the combined stream (seq_len * num_kv_heads) into (num_kv_heads, seq_len)
        x = reshape_stream(x, chunk_size=seq_len, rank=0)
        # 3) move seq_len into the tile‑row dimension (merge stream → tile)
        x = accum_retile_row(x)
        # 4) split the large tile‑row (seq_len * q_per_kv) back into seq_len,
        #    promoting q_per_kv to a new stream dimension
        x = retile_streamify(x, chunk=seq_len, split_row=True)
        # 5) finally split the merged stream (num_kv_heads * q_per_kv) into the two
        #    desired stream dimensions
        x = reshape_stream(x, chunk_size=q_per_kv, rank=0)
        return x

    # ------------------------------------------------------------------
    # Build Qh, Kh, Vh in the child's expected layout
    # ------------------------------------------------------------------
    Qh = _to_gqa(Q, query_per_kvhead)   # (num_kv_heads, query_per_kv) × tile(seq_len, head_dim)
    Kh = _to_gqa(K, 1)                  # (num_kv_heads, 1)               × tile(seq_len, head_dim)
    Vh = _to_gqa(V, 1)                  # (num_kv_heads, 1)               × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # Heavy‑weight attention (child).  Its vanilla output shape is
    # (num_kv_heads, query_per_kv, seq_len, head_dim).
    # ------------------------------------------------------------------
    child_out_shape = ((num_kv_heads, query_per_kvhead, seq_len, head_dim),)
    attn_raw = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=child_out_shape, out_perms=None
    )  # shape: stream(num_kv_heads, query_per_kv) × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # Collapse the two GQA stream dimensions → (num_heads) stream
    # ------------------------------------------------------------------
    attn_flat = flatten(attn_raw, min_rank=0, max_rank=1)   # stream(num_heads) × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # Swap the stream (heads) with the tile‑row (seq) to match the contract:
    #   1) merge stream into tile rows,
    #   2) add a leading singleton stream dimension,
    #   3) retile to pull seq_len back out as the stream.
    # ------------------------------------------------------------------
    tmp = accum_retile_row(attn_flat)          # tile(num_heads * seq_len, head_dim), no stream
    tmp = promote(tmp, rank=0)                 # stream(1) × tile(num_heads * seq_len, head_dim)
    out = retile_streamify(tmp, chunk=num_heads, split_row=True)  # stream(seq_len) × tile(num_heads, head_dim)

    # ------------------------------------------------------------------
    # Return the final stream tensor (shape (seq_len, num_heads, head_dim))
    # ------------------------------------------------------------------
    return out