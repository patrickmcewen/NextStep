# attention_compute
# ----------------------------------------------------------------------
#  Compute causal attention using the DSL primitives.
#  Qh, Kh, Vh are already on‑chip streams with shape (4, 4, 64, 32).
#  1. Expand Kh and Vh so their second stream dimension (size 1) matches Qh’s
#     (size 4) – `expand_ref`.
#  2. Scores = Qh @ Khᵀ – `binary_matmul(..., weight_transposed=True)`.
#  3. Stable‑softmax: subtract the per‑row maximum.
#       * Split the column dimension into a stream dimension
#         (`retile_streamify(..., split_row=False, chunk=1)`) and reduce it
#         with `accum_max` → row_max (tile (64, 1)).
#       * Negate row_max (`unary_mul_imm`) and add to scores (`binary_add`).
#       * Exponentiate (`unary_exp`).
#  4. Numerator = exp(scores) @ Vh – `binary_matmul`.
#  5. Denominator = sum over columns of exp(scores) – `unary_rowwise_sum`.
#  6. Attention = numerator / denominator – `binary_div`.
#  7. Rearrange to the required output shape (64, 16, 32):
#       * Split the tile‑row dimension (64) into chunks of 16,
#         turning the inner stream dimension into size 16
#         (`retile_streamify(..., split_row=True, chunk=16)`),
#         yielding stream shape (4, 16) and tile shape (16, 32).
#       * Flatten the two stream dimensions into one (4 × 16 = 64)
#         (`flatten(..., min_rank=0, max_rank=1)`).
#  The final tensor has the exact shape (64, 16, 32) required by the contract.
# ----------------------------------------------------------------------
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Kh and Vh to Qh's stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # 2. Compute raw attention scores: Qh @ Khᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # 3a. Compute per‑row maximum (stable softmax)
    #     Split columns into a stream dim, then reduce with max.
    scores_split = retile_streamify(scores, chunk=1, split_row=False)
    row_max = accum_max(scores_split, rank=1)

    # 3b. Subtract the max: scores - row_max
    row_max_neg = unary_mul_imm(row_max, -1.0)
    scores_centered = binary_add(scores, row_max_neg)

    # 3c. Exponential
    e = unary_exp(scores_centered)

    # 4. Numerator: e @ Vh
    num = binary_matmul(e, Vh_exp, weight_transposed=False)

    # 5. Denominator: sum over columns of e
    denom = unary_rowwise_sum(e)

    # 6. Attention = num / denom
    attn = binary_div(num, denom)

    # 7a. Split the tile‑row dimension (64) into chunks of 16,
    #     turning the inner stream dim into size 16.
    attn_retile = retile_streamify(attn, chunk=16, split_row=True)

    # 7b. Flatten the two stream dimensions (4, 16) → 64.
    attn_out = flatten(attn_retile, min_rank=0, max_rank=1)

    return attn_out