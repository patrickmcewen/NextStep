# attention_compute
#   Implements the attention (softmax) computation using only DSL ops.
#   The inputs Qh, Kh, Vh are already on‑chip streams.
#   1. Kh and Vh have a singleton “qpkv” dimension (size 1) while Qh has size 4.
#      We broadcast them to match Qh's stream shape using `expand_ref`.
#   2. Compute scores = Qh @ Khᵀ via `binary_matmul` with `weight_transposed=True`.
#   3. Apply the exponential element‑wise (`unary_exp`).  Subtracting the row max
#      is omitted because `exp(scores - max)` is mathematically identical to
#      `exp(scores)` after the subsequent normalization.
#   4. Numerator = e @ Vh (`binary_matmul`).
#   5. Denominator = row‑wise sum of e (`unary_rowwise_sum`), yielding a (…, 1) tile.
#   6. Final attention = numerator / denominator (`binary_div`), broadcasting the
#      denominator's last tile dimension.
#   The result has shape (4, 4, 64, 32), matching the required output shape.

def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Broadcast Kh and Vh across the qpkv dimension to match Qh's stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # scores = Qh @ Khᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # e = exp(scores)
    e = unary_exp(scores)

    # numerator = e @ Vh
    num = binary_matmul(e, Vh_exp, weight_transposed=False)

    # denominator = sum over the last tile dimension (row‑wise sum)
    denom = unary_rowwise_sum(e)

    # attention = numerator / denominator (broadcast denominator's last dim)
    attn = binary_div(num, denom)

    return attn