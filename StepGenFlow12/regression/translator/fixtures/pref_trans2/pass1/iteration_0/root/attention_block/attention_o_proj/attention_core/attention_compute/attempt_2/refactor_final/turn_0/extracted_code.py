# Attention computation using DSL ops.
#   1. Expand Kh and Vh (which have a singleton qpkv dimension) to match Qh's stream shape.
#   2. Compute raw scores = Qh @ Kh^T via binary_matmul with weight_transposed=True.
#   3. Apply exp to scores (softmax numerator).  The max‑sub stabilization step is omitted
#      because the DSL provides no direct reduction over the last tile dimension.
#   4. Compute the numerator = exp(scores) @ Vh.
#   5. Compute the denominator = sum over the column dimension of exp(scores) via unary_rowwise_sum.
#   6. Divide numerator by denominator with broadcasting (binary_div) to obtain the final attention.
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Broadcast the KV tensors across the qpkv stream dimension of Qh.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # 1. Scores: Qh @ Kh^T
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # 2. Exponential of scores (softmax numerator)
    e = unary_exp(scores)

    # 3. Numerator: e @ Vh
    num = binary_matmul(e, Vh_exp)

    # 4. Denominator: sum of e over the column (tile) dimension
    denom = unary_rowwise_sum(e)

    # 5. Final attention: divide numerator by denominator (broadcast along column dim)
    attn = binary_div(num, denom)

    return attn