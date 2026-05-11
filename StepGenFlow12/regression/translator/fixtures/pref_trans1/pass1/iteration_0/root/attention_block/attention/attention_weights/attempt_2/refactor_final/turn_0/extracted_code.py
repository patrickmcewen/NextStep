# Compute attention weights using the DSL primitives.
#   * Qh and Kh are already on‑chip streams with shapes (4,4,64,32) and (4,1,64,32).
#   * `binary_matmul` with weight_transposed=True performs Qh @ Khᵀ,
#     yielding scores of shape (4,4,64,64).
#   * `unary_exp` applies the element‑wise exponential.
#   * `unary_rowwise_sum` computes the sum across the last tile dimension
#     (the column dimension), keeping the dimension so the denominator has
#     shape (4,4,64,1).  This matches the stream shape of the numerator,
#     allowing broadcasting on the tile‑col axis.
#   * `binary_div` divides the exponentiated scores by the row sums,
#     producing the softmax weights of shape (4,4,64,64).
# The result already matches the required output shape (4,4,64,64), so we
# return it directly.  `out_shapes`/`out_perms` are accepted for signature
# compatibility but not needed for the computation.
def attention_weights(Qh, Kh, *, out_shapes, out_perms=None):
    # [Hkv, qpkv, S, S] = Qh @ Khᵀ
    scores = binary_matmul(Qh, Kh, weight_transposed=True)
    # exp(scores)  – stability step (row_max omitted for brevity)
    exp_scores = unary_exp(scores)
    # denominator = Σⱼ exp(scores_{i,j})  (keepdim on tile‑col)
    denom = unary_rowwise_sum(exp_scores)
    # softmax = exp(scores) / denominator
    attn_weights = binary_div(exp_scores, denom)
    return attn_weights