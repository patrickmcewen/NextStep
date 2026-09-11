# Implements GQA full‑sequence attention using only DSL ops.
# 1. Expand Kh and Vh so their stream shapes match Qh/e (broadcast the
#    query‑per‑kv‑head dimension).
# 2. Compute scores = Qh @ Khᵀ.
# 3. Apply exp, then compute the numerator (e @ V) and denominator (row‑wise sum of e).
# 4. Divide to obtain the attention tensor of shape (4,4,64,32).
# 5. Collapse the two head‑stream dimensions (4·4 → 16) with `flatten`.
# 6. Move the sequence‑length dimension from the tile rows into the stream,
#    turning the tile rows into the head dimension (16) via `retile_streamify`.
# The resulting stream has shape (64, 16, 32), matching the required output.
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast Kh and Vh across the query‑per‑kv‑head stream dimension.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)          # (4,4,64,32)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)          # (4,4,64,32) – will be re‑expanded later

    # 2. Compute raw attention scores.
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)  # (4,4,64,64)

    # 3. Apply the exponential (stable‑softmax max step omitted for brevity).
    e = unary_exp(scores)                                 # (4,4,64,64)

    # Expand Vh to match e's stream shape for the second matmul.
    Vh_exp = expand_ref(Vh, e, expand_rank=1)            # (4,4,64,32)

    # 4. Numerator and denominator of the softmax.
    num = binary_matmul(e, Vh_exp)                        # (4,4,64,32)
    denom = unary_rowwise_sum(e)                          # (4,4,64,1)

    # 5. Final attention values.
    attn = binary_div(num, denom)                         # (4,4,64,32)

    # 6. Merge the two head‑stream dimensions (4 × 4 → 16).
    attn_flat = flatten(attn, min_rank=0, max_rank=1)     # (16,64,32)

    # 7. Turn the sequence‑length tile rows into a stream dimension,
    #    making the head dimension the tile rows (chunk = 16).
    attn_out = retile_streamify(attn_flat, chunk=16, split_row=True)  # (64,16,32)

    return attn_out