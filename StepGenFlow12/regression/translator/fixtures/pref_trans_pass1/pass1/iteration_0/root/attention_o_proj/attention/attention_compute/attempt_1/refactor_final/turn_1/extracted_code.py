# attention_compute implements GQA full‑sequence attention using only DSL ops.
# The original implementation omitted the row‑wise max subtraction used for
# numerically‑stable softmax, which caused overflow → NaNs.  We compute the
# max per row by moving the column dimension (tile_c) into a stream dimension
# with `promote` + `retile_streamify`, then reduce it with `accum_max`.  The
# max is negated (`unary_mul_imm`) and added back to the scores, after which
# we proceed with the regular softmax steps and finally reshape the result
# to (seq_len, num_heads, head_dim) = (64, 16, 32) expressed as stream(64,)
# × tile(16,32).
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # 1. Expand the single KV head (Kh, Vh) across the query‑per‑kv dimension.
    # -----------------------------------------------------------------
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)   # (kv, q_per_kv) → (4,4)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # -----------------------------------------------------------------
    # 2. Compute raw attention scores: Q · Kᵀ
    #    Shape: stream(4,4) × tile(64,64)
    # -----------------------------------------------------------------
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # -----------------------------------------------------------------
    # 3. Stable softmax – subtract per‑row max.
    #    a) Promote a singleton inner stream dim.
    #    b) Split the tile‑column dimension into a stream dimension.
    #    c) Reduce that new stream dim with max.
    #    d) Negate the max and broadcast‑add to the original scores.
    # -----------------------------------------------------------------
    # a) add a singleton stream dim just before the tiles
    scores_prom = promote(scores, rank=0)          # stream(4,4,1) × tile(64,64)

    # b) move the 64 tile‑columns into an innermost stream dim (chunk=1)
    scores_split = retile_streamify(
        scores_prom, chunk=1, split_row=False
    )                                              # stream(4,4,64) × tile(64,1)

    # c) max over that new dim (rank=1 reduces the innermost stream dim)
    row_max = accum_max(scores_split, rank=1)      # stream(4,4) × tile(64,1)

    # d) negate and add back to the original scores
    neg_max = unary_mul_imm(row_max, -1.0)         # stream(4,4) × tile(64,1)
    scores_centered = binary_add(scores, neg_max) # broadcast → tile(64,64)

    # -----------------------------------------------------------------
    # 4. Exponentiate the centered scores.
    # -----------------------------------------------------------------
    e = unary_exp(scores_centered)                 # stream(4,4) × tile(64,64)

    # -----------------------------------------------------------------
    # 5. Compute denominator = Σₖ e  (row‑wise sum over the column dim)
    # -----------------------------------------------------------------
    denom = unary_rowwise_sum(e)                   # stream(4,4) × tile(64,1)

    # -----------------------------------------------------------------
    # 6. Compute numerator = e · V
    # -----------------------------------------------------------------
    num = binary_matmul(e, Vh_exp)                 # stream(4,4) × tile(64,32)

    # -----------------------------------------------------------------
    # 7. Final attention = num / denom (broadcast division)
    # -----------------------------------------------------------------
    attn = binary_div(num, denom)                  # stream(4,4) × tile(64,32)

    # -----------------------------------------------------------------
    # 8. Reshape to (seq_len, num_heads, head_dim):
    #    a) merge the two head‑related stream dims → (16,)
    #    b) promote a singleton inner dim
    #    c) split the tile‑row (seq_len) into stream dim of size 4
    #    d) merge that with the existing head stream → (64,)
    # -----------------------------------------------------------------
    attn = flatten(attn, min_rank=0, max_rank=1)          # stream(16,) × tile(64,32)
    attn = promote(attn, rank=0)                         # stream(16,1) × tile(64,32)
    attn = retile_streamify(attn, chunk=16, split_row=True)  # stream(16,4) × tile(16,32)
    attn = flatten(attn, min_rank=0, max_rank=1)          # stream(64,) × tile(16,32)

    return attn