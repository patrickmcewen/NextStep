# Compute scaled‑dot‑product attention with max‑sub stability using only DSL ops.
#   • Qh, Kh, Vh are already on‑chip streams.
#   • Kh and Vh have a singleton qpkv dimension; we broadcast them to Qh's qpkv dimension.
#   • scores   = Qh @ Khᵀ  (shape: …×tile(64,64))
#   • row_max  = max over the column dimension of scores.
#       – Split the column tile into a stream dimension (retile_streamify + reshape_stream).
#       – Reduce that stream dimension with accum_max.
#   • scores_centered = scores - row_max   (broadcasted subtraction via binary_add + unary_mul_imm).
#   • e = exp(scores_centered)
#   • num    = e @ Vh
#   • denom  = sum over columns of e (unary_rowwise_sum)
#   • attn   = num / denom
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # Broadcast Kh and Vh across the qpkv stream dimension of Qh.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # Raw attention scores.
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # ---- Row‑wise max for numerical stability ----
    # Move the column tile dimension into a stream dimension.
    scores_retiled = retile_streamify(scores, chunk=1, split_row=False)
    # Separate the original qpkv dimension from the column index.
    scores_reshaped = reshape_stream(scores_retiled, chunk_size=64, rank=0)
    # Reduce over the column stream dimension to obtain max per row.
    row_max = accum_max(scores_reshaped, rank=1)          # shape: stream(4,4)×tile(64,1)

    # Subtract max from scores (broadcasted across the column tile).
    scores_centered = binary_add(scores, unary_mul_imm(row_max, -1.0))

    # Exponential of the stabilized scores.
    e = unary_exp(scores_centered)

    # Numerator of softmax.
    num = binary_matmul(e, Vh_exp)

    # Denominator (sum over the column dimension).
    denom = unary_rowwise_sum(e)

    # Final attention output.
    attn = binary_div(num, denom)

    return attn