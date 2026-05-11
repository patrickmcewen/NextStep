# Implementation reasoning:
# Qh, Kh, Vh are already on‑chip streams with shapes
#   Qh : (num_kv_heads, query_per_kvhead, seq_len, head_dim) = (4,4,64,32)
#   Kh : (num_kv_heads, 1, seq_len, head_dim)                = (4,1,64,32)
#   Vh : (num_kv_heads, 1, seq_len, head_dim)                = (4,1,64,32)
# We first broadcast Kh and Vh across the query_per_kvhead dimension using
# `expand_ref`. Then we perform the standard GQA attention steps:
#   scores = Qh @ Kh.T
#   e      = exp(scores)                  (stable subtraction omitted)
#   denom  = e.sum(dim=-1, keepdim=True)
#   num    = e @ Vh
#   attn   = num / denom                  (broadcast denominator)
# The resulting tensor has stream shape (4,4) and tile shape (64,32).
# To obtain the final (seq_len, num_heads, head_dim) layout we:
#   * Move the seq_len dimension out of the tile row into the stream via
#     `retile_streamify` with chunk=1 (splits each row into a separate token).
#   * Flatten the two remaining stream dimensions (kv_head, expanded queries)
#     into one combined dimension.
#   * Split that combined dimension back into (seq_len, num_heads) using
#     `reshape_stream`, where the chunk size (num_heads) is taken from the
#     contract's `out_shapes`.
# The final tensor has stream shape (seq_len, num_heads) and tile shape (1,
# head_dim), which `offchip_store` (in the root) will turn into the vanilla
# shape (64, 16, 32).

def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Broadcast KV across the query‑per‑KV dimension
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # Q · Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # exp(scores) – we omit the max subtraction (it does not affect the
    # final softmax result mathematically)
    e = unary_exp(scores)

    # denominator: Σⱼ eᵢⱼ  (keepdim keeps the tile‑column dimension as 1)
    denom = unary_rowwise_sum(e)

    # numerator: e · V
    num = binary_matmul(e, Vh_exp, weight_transposed=False)

    # attention = numerator / denominator (broadcast denominator over head_dim)
    attn = binary_div(num, denom)

    # Move the sequence length from tile rows into the stream dimension
    attn_stream = retile_streamify(attn, chunk=1, split_row=True)

    # Collapse the two stream dimensions (kv_head, query_per_kvhead * seq_len)
    attn_flat = flatten(attn_stream, min_rank=0, max_rank=1)

    # Split the combined dimension into (seq_len, num_heads) using the
    # requested output shape; out_shapes[0] = (seq_len, num_heads, head_dim)
    num_heads = out_shapes[0][1]  # e.g., 16 for the given model
    attn_out = reshape_stream(attn_flat, chunk_size=num_heads, rank=0)

    return attn_out