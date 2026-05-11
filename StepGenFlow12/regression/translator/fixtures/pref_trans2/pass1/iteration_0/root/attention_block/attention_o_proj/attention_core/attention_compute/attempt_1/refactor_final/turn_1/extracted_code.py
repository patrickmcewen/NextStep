# Attention computed as  (exp(Q @ Kᵀ) @ V) / exp(Q @ Kᵀ).sum(dim=-1,keepdim=True)
# The KV tensors (Kh, Vh) have a singleton “queries‑per‑KV‑head” stream dim.
# We repeat them across that dimension with `repeat_static` and then merge the
# extra dim using `flatten` so their stream shape matches Qh’s (Hkv, qpkv).
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Number of query‑per‑KV‑head streams (the second stream dim of Qh)
    factor = Qh.shape[-3]  # int = 4 for the given config

    # Align Kh to Qh's stream shape (4,4,64,32)
    Kh_rep = repeat_static(Kh, factor)                 # -> (4,1,4,64,32)
    Kh_aligned = flatten(Kh_rep, min_rank=0, max_rank=1)  # merge (1,4) → (4,4)

    # Align Vh similarly
    Vh_rep = repeat_static(Vh, factor)                 # -> (4,1,4,64,32)
    Vh_aligned = flatten(Vh_rep, min_rank=0, max_rank=1)  # -> (4,4,64,32)

    # Scores = Qh @ Kᵀ
    scores = binary_matmul(Qh, Kh_aligned, weight_transposed=True)

    # Exponential of scores (softmax numerator)
    e = unary_exp(scores)

    # Numerator = e @ Vh
    num = binary_matmul(e, Vh_aligned, weight_transposed=False)

    # Denominator = row‑wise sum of e (keeps a singleton column for broadcasting)
    denom = unary_rowwise_sum(e)

    # Final attention output
    attn = binary_div(num, denom)
    return attn