# -------------------------------------------------------------
# attention_compute
# Implements the reference logic entirely with DSL ops.
#   * Q, K, V are on‑chip streams (seq, heads, dim).
#   * They are reshaped into the GQA layout expected by the child
#     `attention_compute__root_attention_attention_compute`:
#         Qh : (num_kv_heads, query_per_kv, seq, dim)
#         Kh/Vh : (num_kv_heads, 1, seq, dim)
#   * After the child returns its attention tensor, the two GQA stream
#     dimensions are merged back to the original “heads” stream and then
#     swapped with the sequence dimension to obtain the contract‑required
#     layout (seq, heads, dim).
# -------------------------------------------------------------
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalar constants (plain Python ints – allowed)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # Helper: reshape a (seq, heads) stream into GQA layout.
    #   tensor : stream(seq_len,) × tile(num_heads, head_dim)
    #   qpkv   : queries‑per‑KV‑head (4 for Q, 1 for K/V)
    #   Returns: stream(num_kv_heads, qpkv) × tile(seq_len, head_dim)
    # ------------------------------------------------------------------
    def _to_gqa(tensor, qpkv):
        # 1) split the tile‑row dimension (num_heads) into chunks of size qpkv
        x = retile_streamify(tensor, chunk=qpkv, split_row=True)

        # 2) split the combined stream (seq_len * num_kv_heads) into
        #    (num_kv_heads, seq_len) while keeping tile rows = qpkv
        x = reshape_stream(x, chunk_size=seq_len, rank=0)

        # 3) merge the seq_len stream dimension into the tile rows
        x = accum_retile_row(x)

        # 4) split the enlarged tile rows (seq_len * qpkv) back into
        #    seq_len‑sized chunks, promoting qpkv to a new stream dimension
        x = retile_streamify(x, chunk=seq_len, split_row=True)

        # 5) finally split the merged stream (num_kv_heads * qpkv) into the
        #    two desired stream dimensions
        x = reshape_stream(x, chunk_size=qpkv, rank=0)
        return x

    # ------------------------------------------------------------------
    # 1️⃣  Build Qh, Kh, Vh in the layout the child expects.
    # ------------------------------------------------------------------
    Qh = _to_gqa(Q, query_per_kvhead)   # (num_kv_heads, query_per_kv) × tile(seq_len, head_dim)
    Kh = _to_gqa(K, 1)                  # (num_kv_heads, 1)               × tile(seq_len, head_dim)
    Vh = _to_gqa(V, 1)                  # (num_kv_heads, 1)               × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # 2️⃣  Heavy‑weight attention (child).  Its vanilla output shape is
    #     (num_kv_heads, query_per_kv, seq_len, head_dim).
    # ------------------------------------------------------------------
    child_out_shape = ((num_kv_heads, query_per_kvhead, seq_len, head_dim),)
    attn_raw = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=child_out_shape, out_perms=None
    )   # stream(num_kv_heads, query_per_kv) × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # 3️⃣  Collapse the GQA stream dimensions back to a single “heads”
    #     stream dimension.
    # ------------------------------------------------------------------
    attn_flat = flatten(attn_raw, min_rank=0, max_rank=1)   # stream(num_heads) × tile(seq_len, head_dim)

    # ------------------------------------------------------------------
    # 4️⃣  Swap the heads stream with the sequence tile‑row dimension.
    #     After this step the tensor has the contract‑required layout:
    #         stream(seq_len) × tile(num_heads, head_dim)
    # ------------------------------------------------------------------
    tmp = promote_outer(attn_flat)           # (1, num_heads) × tile(seq_len, head_dim)
    tmp = accum_retile_row(tmp)              # (1,) × tile(num_heads * seq_len, head_dim)
    out = retile_streamify(tmp, chunk=num_heads, split_row=True)  # (seq_len,) × tile(num_heads, head_dim)

    # ------------------------------------------------------------------
    # 5️⃣  Return the final stream tensor.
    # ------------------------------------------------------------------
    return out