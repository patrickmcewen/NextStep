# attention_compute
# ----------------------------------------------------------------------
# Compute the full‑sequence GQA attention using only DSL ops.
#   1. Expand K and V to match Qh’s stream shape.
#   2. Scores = Qh @ Kᵀ                     (binary_matmul, weight_transposed=True)
#   3. Stable‑softmax:
#        • Promote the scores to add a dummy innermost stream dim.
#        • Split the *column* tile into a new stream dimension
#          (retile_streamify, split_row=False) – this keeps the original
#          query‑per‑kv‑head dimension separate.
#        • Reduce that column‑stream dim with accum_max → row_max.
#        • Negate row_max and add to scores (binary_add) → scores_centered.
#        • Exponentiate (unary_exp) → e.
#   4. Numerator = e @ V                     (binary_matmul)
#   5. Denominator = row‑wise sum of e       (unary_rowwise_sum)
#   6. Attention = numerator / denominator   (binary_div)
#   7. Reshape to the contract‑required vanilla shape
#        (seq_len, num_heads, head_dim) = (64, 16, 32):
#        • Promote → split the *row* tile into a stream dimension,
#          turning seq_len into a stream dim while keeping the head‑dim
#          tile unchanged.
#        • Flatten the three stream dimensions (kv, query, seq_len) → one
#          long stream.
#        • reshape_stream splits that long stream into (64, 16) where the
#          innermost dim (16) represents the number of heads.
#        • accum_retile_row absorbs the innermost dim (16) into the tile
#          rows, yielding the final stream shape (64,) and tile shape
#          (16, 32).  The underlying torch tensor therefore has shape
#          (64, 16, 32), exactly matching the contract.
# ----------------------------------------------------------------------
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # 1. Broadcast K and V to Qh’s stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # 2. Scores = Qh @ Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # 3. Stable‑softmax -------------------------------------------------
    #    a) Promote to expose a singleton innermost stream dim.
    scores_prom = promote(scores, rank=0)

    #    b) Split the column tile (size 64) into a stream dimension.
    #       Because we promoted, the new column‑stream merges with the
    #       singleton, leaving the original query dimension untouched.
    scores_split = retile_streamify(scores_prom, chunk=1, split_row=False)

    #    c) Max‑reduce over that new column‑stream dimension.
    row_max = accum_max(scores_split, rank=1)

    #    d) Subtract the max (scores - row_max).
    row_max_neg = unary_mul_imm(row_max, -1.0)
    scores_centered = binary_add(scores, row_max_neg)

    #    e) Exponential.
    e = unary_exp(scores_centered)

    # 4. Numerator = e @ V
    num = binary_matmul(e, Vh_exp, weight_transposed=False)

    # 5. Denominator = sum over columns of e
    denom = unary_rowwise_sum(e)

    # 6. Attention = num / denom
    attn = binary_div(num, denom)

    # 7. Reshape to (seq_len=64, num_heads=16, head_dim=32) ------------
    #    a) Promote then split the row tile (seq_len) into a stream dim.
    attn_prom = promote(attn, rank=0)
    attn_split = retile_streamify(attn_prom, chunk=1, split_row=True)

    #    b) Flatten the three stream dims (kv, query, seq_len) → one.
    attn_flat = flatten(attn_split, min_rank=0, max_rank=2)

    #    c) Split that long stream into (64, 16).
    attn_resh = reshape_stream(attn_flat, chunk_size=16, rank=0)

    #    d) Absorb the innermost dim (16) into tile rows → final shape.
    attn_out = accum_retile_row(attn_resh, rank=1)

    return attn_out