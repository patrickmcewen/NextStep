# Compute stable soft‑max attention weights using DSL ops.
# Qh and Kh are already on‑chip streams with shapes:
#   Qh: (4, 4, 64, 32)  # stream (kv_head, q_per_kv), tile (seq_len, head_dim)
#   Kh: (4, 1, 64, 32)  # stream (kv_head, 1),       tile (seq_len, head_dim)
# To make the stream shapes match for the matmul we broadcast Kh across the
# second stream dimension using `expand_ref`.  The rest follows the stable‑softmax
# pattern: scores = Qh @ Khᵀ, exp = exp(scores), denom = sum_exp(scores,
# dim=-1, keepdim=True), and finally softmax = exp / denom.
def attention_weights(Qh, Kh, *, out_shapes, out_perms=None):
    # Broadcast Kh from stream shape (kv_head, 1) → (kv_head, q_per_kv)
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)

    # Qh @ Khᵀ → (kv_head, q_per_kv, seq_len, seq_len)
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # exp(scores)
    exp_scores = unary_exp(scores)

    # sum over the last tile dimension (columns) → (kv_head, q_per_kv, seq_len, 1)
    denom = unary_rowwise_sum(exp_scores)

    # stable softmax: exp(scores) / sum_exp
    attn_weights = binary_div(exp_scores, denom)

    return attn_weights