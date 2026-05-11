# Compute stable attention weights.
#   1. Kh is a singleton stream on the second dimension; expand it to match Qh's stream shape.
#   2. Multiply Qh with the transpose of Kh to obtain raw scores (shape (4,4,64,64)).
#   3. To get the per‑row maximum we turn the column tile dimension into a stream:
#        a) retile the column dimension (split_col=True, chunk=1) → stream dim combines qpkv * col
#        b) reshape that combined stream dimension into separate (qpkv, col) streams
#        c) reduce over the column stream with accum_max, yielding shape (4,4,64,1)
#   4. Subtract the max from the scores (broadcast across columns) to improve numerical stability.
#   5. Exponentiate, sum across columns, and divide to obtain softmax weights.
def attention_weights(Qh, Kh, *, out_shapes, out_perms=None):
    # Expand Kh's singleton stream dimension to match Qh's (4,4) stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)

    # Raw attention scores: Qh @ Khᵀ  → (4,4,64,64)
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # Turn the column tile dimension into a stream so we can take a max per row.
    #  a) Split the column tile into a new stream dimension (qpkv * col).
    scores_col_stream = retile_streamify(scores, chunk=1, split_col=True)

    #  b) Separate the combined stream into (qpkv, col) streams.
    scores_split = reshape_stream(scores_col_stream, chunk_size=64, rank=0)

    #  c) Max over the column stream (last stream dim) → (4,4,64,1)
    row_max = accum_max(scores_split, rank=1)

    # Subtract the max from the original scores (broadcast over tile_c).
    scores_centered = binary_add(scores, unary_mul_imm(row_max, -1.0))

    # Exponentiate.
    exp_scores = unary_exp(scores_centered)

    # Denominator = sum over columns (keepdim → shape (4,4,64,1)).
    denom = unary_rowwise_sum(exp_scores)

    # Softmax weights.
    attn_weights = binary_div(exp_scores, denom)

    return attn_weights