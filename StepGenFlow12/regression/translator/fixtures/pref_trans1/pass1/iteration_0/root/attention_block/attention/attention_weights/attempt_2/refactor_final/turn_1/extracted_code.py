# Compute attention weights using DSL ops.
#   * Qh is already on‑chip with stream shape (4,4).
#   * Kh has stream shape (4,1); we expand the trailing stream dimension
#     (the “1”) to match Qh’s second stream dim using `expand_ref`.
#   * `binary_matmul` (with weight_transposed=True) produces the
#     raw scores of shape (4,4,64,64).
#   * `unary_exp` exponentiates the scores.
#   * `unary_rowwise_sum` sums over the last tile dimension (the column
#     axis), yielding a denominator of shape (4,4,64,1) that broadcasts
#     across columns.
#   * `binary_div` divides the exponentiated scores by the denominator,
#     giving the soft‑max weights with shape (4,4,64,64).
def attention_weights(Qh, Kh, *, out_shapes, out_perms=None):
    # Expand Kh's singleton stream dimension to match Qh
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)

    # scores = Qh @ Khᵀ  → (4,4,64,64)
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # e = exp(scores)
    exp_scores = unary_exp(scores)

    # denom = Σ_col exp(scores)  → (4,4,64,1), kept for broadcasting
    denom = unary_rowwise_sum(exp_scores)

    # attn_weights = e / denom  → (4,4,64,64)
    attn_weights = binary_div(exp_scores, denom)

    return attn_weights