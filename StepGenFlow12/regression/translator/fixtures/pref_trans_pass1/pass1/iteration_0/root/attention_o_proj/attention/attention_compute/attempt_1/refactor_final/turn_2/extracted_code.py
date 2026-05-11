# attention_compute implements GQA full‑sequence attention using only DSL ops.
# It follows the reference computation:
#   scores = Q·Kᵀ
#   e      = exp(scores - max(scores, dim=-1, keepdim=True))
#   attn   = (e·V) / e.sum(dim=-1, keepdim=True)
# Finally it transposes the result to shape (seq_len, num_heads, head_dim).
def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    # ---------------------------------------------------------------
    # 1. Expand the single KV head (Kh, Vh) across the query‑per‑kv dim.
    # ---------------------------------------------------------------
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)   # (4,4) stream
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)   # (4,4) stream

    # ---------------------------------------------------------------
    # 2. Raw scores = Q · Kᵀ
    # ---------------------------------------------------------------
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # ---------------------------------------------------------------
    # 3. Stable soft‑max: subtract per‑row max before exponentiation.
    #    a) Move the column dimension into a stream slot.
    #    b) Reduce that new stream dim with max.
    #    c) Negate and broadcast‑add the max to the original scores.
    # ---------------------------------------------------------------
    scores_prom = promote(scores, rank=0)                     # add singleton stream dim
    scores_split = retile_streamify(
        scores_prom, chunk=1, split_row=False                  # split columns → stream
    )
    row_max = accum_max(scores_split, rank=1)                # max over new stream dim
    neg_max = unary_mul_imm(row_max, -1.0)                    # −max
    scores_centered = binary_add(scores, neg_max)            # scores – max

    # ---------------------------------------------------------------
    # 4. Exponentiate, compute denominator and numerator.
    # ---------------------------------------------------------------
    e = unary_exp(scores_centered)                           # exp(scores - max)
    denom = unary_rowwise_sum(e)                             # Σⱼ exp(...)
    num = binary_matmul(e, Vh_exp)                           # weighted sum of V

    # ---------------------------------------------------------------
    # 5. Attention = numerator / denominator.
    # ---------------------------------------------------------------
    attn = binary_div(num, denom)                             # (heads, seq, head_dim)

    # ---------------------------------------------------------------
    # 6. Transpose to (seq_len, num_heads, head_dim):
    #    a) Absorb both head stream dimensions into the tile‑row dimension.
    #    b) Promote a singleton stream dim so retile_streamify can act.
    #    c) Split the enlarged tile‑row (seq_len·num_heads) into a
    #       stream of size seq_len and a tile‑row of size num_heads.
    # ---------------------------------------------------------------
    # a) merge the two head stream dims (kv and q_per_kv) into the tile rows
    attn_abs = accum_retile_row(attn, rank=2)                 # stream → (), tile (1024,32)

    # b) add a singleton stream dim
    attn_abs = promote(attn_abs, rank=0)                      # stream (1,), tile (1024,32)

    # c) split tile rows back into (seq_len, num_heads)
    num_heads = Qh.tensor.shape[0] * Qh.tensor.shape[1]       # 4 × 4 = 16
    attn_out = retile_streamify(
        attn_abs, chunk=num_heads, split_row=True              # split rows → (seq, heads)
    )                                                          # result: stream(64,) × tile(16,32)

    return attn_out