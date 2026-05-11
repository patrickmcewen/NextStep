# Attention_compute implementation.
# 1️⃣  Convert the input tensors (Q, K, V) from the
#     vanilla (seq, heads, dim) layout to the GQA layout expected by the
#     heavy‑weight child:
#         Qh : (num_kv_heads, query_per_kv, seq, dim)
#         Kh/Vh : (num_kv_heads, 1, seq, dim)
#     The transformation swaps the “seq” stream dimension with the tile‑row
#     dimension (heads) using only DSL primitives – no tensor methods.
# 2️⃣  Call the child to perform the scaled‑dot‑product attention on the
#     GQA tensors.
# 3️⃣  The child returns a stream shaped (num_kv_heads, query_per_kv, seq,
#     dim).  Flatten the two leading stream dimensions to obtain a stream of
#     shape (num_heads, seq, dim) and then swap the stream dimension (heads)
#     back to the tile‑row dimension, yielding the contract‑required layout
#     (seq, num_heads, dim).
def attention_compute(Q, K, V, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalars extracted from the input shapes (plain Python ints)
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]                     # 64
    num_heads = Q.shape[1]                   # 16
    head_dim = Q.shape[2]                    # 32
    num_kv_heads = K.shape[1]                # 4
    query_per_kvhead = num_heads // num_kv_heads  # 4

    # ------------------------------------------------------------------
    # Helper: convert a (seq, heads) stream into (kv, q_per_kv) stream.
    # ------------------------------------------------------------------
    def _to_gqa(tensor, q_per_kv):
        # tensor: stream(seq_len,)×tile(num_heads, dim),
        # where num_heads = num_kv_heads * q_per_kv.
        # 1) split the tile‑row dimension into (kv, q_per_kv)
        x = retile_streamify(tensor, chunk=q_per_kv, split_row=True)
        # now: stream(seq_len * num_kv_heads,)×tile(q_per_kv, dim)

        # 2) split the combined stream dimension into (kv, seq_len)
        x = reshape_stream(x, chunk_size=seq_len, rank=0)
        # now: stream(num_kv_heads, seq_len)×tile(q_per_kv, dim)

        # 3) swap the second stream dimension (seq_len) with the tile rows
        #    (q_per_kv) -> final GQA layout.
        x = promote_outer(x)                               # (1, kv, seq_len)×tile(q_per_kv, dim)
        x = accum_retile_row(x)                            # (1, kv)×tile(seq_len * q_per_kv, dim)
        x = retile_streamify(x, chunk=seq_len, split_row=True)  # (1, kv * q_per_kv)×tile(seq_len, dim)
        x = reshape_stream(x, chunk_size=q_per_kv, rank=0)       # (1, kv, q_per_kv)×tile(seq_len, dim)
        x = flatten(x, min_rank=1, max_rank=2)                 # (kv, q_per_kv)×tile(seq_len, dim)
        return x

    # ------------------------------------------------------------------
    # 1️⃣  Build Qh, Kh, Vh in the child’s expected layout.
    # ------------------------------------------------------------------
    Qh = _to_gqa(Q, query_per_kvhead)   # (kv, q_per_kv)×tile(seq, dim)
    Kh = _to_gqa(K, 1)                  # (kv, 1)×tile(seq, dim)
    Vh = _to_gqa(V, 1)                  # (kv, 1)×tile(seq, dim)

    # ------------------------------------------------------------------
    # 2️⃣  Heavy‑weight attention (child).  Request its natural output shape.
    # ------------------------------------------------------------------
    child_out_shape = ((num_kv_heads, query_per_kvhead, seq_len, head_dim),)
    attn_pre = attention_compute__root_attention_attention_compute(
        Qh, Kh, Vh, out_shapes=child_out_shape, out_perms=None
    )  # shape (kv, q_per_kv)×tile(seq, dim)

    # ------------------------------------------------------------------
    # 3️⃣  Collapse the GQA stream dimensions → (heads)×tile(seq, dim)
    # ------------------------------------------------------------------
    attn_flat = flatten(attn_pre, min_rank=0, max_rank=1)   # (num_heads,)×tile(seq, dim)

    # ------------------------------------------------------------------
    # 4️⃣  Swap the stream (heads) with the tile‑row (seq) to obtain
    #     (seq)×tile(num_heads, dim), which matches the contract.
    # ------------------------------------------------------------------
    attn = promote_outer(attn_flat)          # (1, num_heads)×tile(seq, dim)
    attn = accum_retile_row(attn)            # (1,)×tile(seq * num_heads, dim)
    attn = retile_streamify(attn, chunk=num_heads, split_row=True)  # (seq,)×tile(num_heads, dim)

    # ------------------------------------------------------------------
    # 5️⃣  Return the final stream tensor.
    # ------------------------------------------------------------------
    return attn