# attention_weights:
#   1. Expand Kh so its stream shape matches Qh (broadcast over the query-per‑kv‑head dimension).
#   2. Compute raw attention scores via a matmul with Kh transposed (weight_transposed=True).
#   3. Apply the exponential (softmax numerator).
#   4. Sum across the column dimension (tile_c) to obtain the denominator.
#   5. Divide the numerator by the denominator – broadcasting the denominator across columns.
#   The resulting tensor has shape (4, 4, 64, 64), matching the required output shape.
def attention_weights(Qh, Kh, *, out_shapes, out_perms=None):
    # 1. Broadcast Kh over the query‑per‑kv‑head dimension.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)          # shape: (4, 4, 64, 32)

    # 2. Compute Q·Kᵀ.
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)  # (4, 4, 64, 64)

    # 3. Exponential of scores.
    e = unary_exp(scores)                               # (4, 4, 64, 64)

    # 4. Row‑wise sum to get denominator (shape: (4, 4, 64, 1)).
    denom = unary_rowwise_sum(e)

    # 5. Softmax: divide numerator by denominator (broadcast across tile_c).
    attn_weights = binary_div(e, denom)                 # (4, 4, 64, 64)

    return attn_weights