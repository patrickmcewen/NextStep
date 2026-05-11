# Implements attention: attn = (exp(Q @ Kᵀ) @ V) / exp(Q @ Kᵀ).sum(dim=-1,keepdim=True)
# Row‑wise max is omitted because softmax is invariant to a constant shift.
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Qh @ Kᵀ  → scores (tile shape 64×64)
    scores = binary_matmul(Qh, Kh, weight_transposed=True)
    # e = exp(scores)  (softmax numerator before weighting)
    e = unary_exp(scores)
    # Numerator: e @ Vh  → shape (64×32)
    num = binary_matmul(e, Vh, weight_transposed=False)
    # Denominator: row‑wise sum of e (keeps dim) → shape (64×1), broadcastable to (64×32)
    denom = unary_rowwise_sum(e)
    # Final attention: element‑wise division (broadcast denom across columns)
    attn = binary_div(num, denom)
    return attn