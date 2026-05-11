# attention_weights:
#   1. Broadcast Kh over the query‑per‑kv‑head dimension (expand_ref).
#   2. Compute raw attention scores with a transposed matmul.
#   3. Obtain the per‑row maximum (stable softmax) by
#        a) moving the column dimension (tile_c) into the stream via retile_streamify,
#        b) separating the combined stream dimension back into (query‑head, column) using reshape_stream,
#        c) reducing over the column stream dimension with accum_max.
#   4. Subtract the row‑wise max from the scores (implemented as addition with a negated max).
#   5. Apply exp to obtain the softmax numerator.
#   6. Sum over the column dimension to get the denominator.
#   7. Divide numerator by denominator → final attention weights.
# The resulting tensor has shape (4, 4, 64, 64), matching the required output.
def attention_weights(Qh, Kh, *, out_shapes, out_perms=None):
    # 1. Broadcast Kh to match Qh's query‑per‑kv‑head dimension.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)                # (4, 4, 64, 32)

    # 2. Compute Q · Kᵀ.
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)  # (4, 4, 64, 64)

    # 3. Compute row‑wise max for numerical stability.
    #    a) Move tile columns into the stream (split_col) – chunk=1 leaves tile_c=1.
    scores_split = retile_streamify(scores, chunk=1, split_row=False)   # (4, 256, 64, 1)
    #    b) Split the combined stream dimension (256 = 4 * 64) back into
    #       (query‑head=4, column=64).
    #    The column size equals the original tile_c dimension.
    col_dim = scores.shape[-1]                               # 64
    scores_reshaped = reshape_stream(scores_split, chunk_size=col_dim, rank=0)  # (4, 4, 64, 64, 1)
    #    c) Reduce over the column stream dimension to get the max per row.
    row_max = accum_max(scores_reshaped, rank=1)             # (4, 4, 64, 1)

    # 4. Subtract the max from the scores (a + (‑max)).
    neg_max = unary_mul_imm(row_max, -1.0)                    # (4, 4, 64, 1)
    shifted = binary_add(scores, neg_max)                    # (4, 4, 64, 64)

    # 5. Exponential for the softmax numerator.
    e = unary_exp(shifted)                                    # (4, 4, 64, 64)

    # 6. Row‑wise sum to obtain the denominator.
    denom = unary_rowwise_sum(e)                              # (4, 4, 64, 1)

    # 7. Softmax: divide numerator by denominator (broadcast across columns).
    attn_weights = binary_div(e, denom)                       # (4, 4, 64, 64)

    return attn_weights