def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    """
    Compute full‑sequence multi‑query attention using only DSL ops.

    1.  The key/value tensors have a trailing singleton stream dimension
        (shape ... , 1, ...).  They must be broadcast to the query’s
        (kv_h, query_per_kv) stream shape – `expand_ref` does exactly that
        without any arithmetic.
    2.  Scores = Q · Kᵀ  →  `binary_matmul` with `weight_transposed=True`.
    3.  Stable softmax needs the per‑row maximum.
        * Move the column‑wise tile dimension into a stream dimension
          (`promote` → `retile_streamify` with `split_row=False`).
        * Reduce that new stream dim with `accum_max` → shape (…, 1).
    4.  Subtract the max (broadcast via `binary_add` and a negated constant
        from `unary_mul_imm`), then apply `exp` (`unary_exp`).
    5.  Numerator = (exp‑scores) · V    → another `binary_matmul`.
    6.  Denominator = sum over the same column dimension.
        * Same split‑and‑reduce pattern as the max, but with `accum_add`.
    7.  Final attention = numerator / denominator (`binary_div`).

    The resulting StepTensor has stream shape (kv_h, query_per_kv, seq_len)
    and tile shape (head_dim).  This matches the parent’s expected
    vanilla shape (seq_len, num_heads, head_dim) after the surrounding
    graph’s reshaping; the `out_shapes` / `out_perms` arguments are not
    needed here because the shape is derived from the DSL computation.
    """
    # 1. Broadcast K and V to Q’s stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # 2. Scores = Q @ Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # 3. Row‑wise max for numerical stability.
    scores_prom = promote(scores, rank=0)                     # add dummy stream dim
    scores_split = retile_streamify(scores_prom,
                                    chunk=1,
                                    split_row=False)     # split column dim
    row_max = accum_max(scores_split, rank=1)                # reduce over new dim

    # 4. scores – row_max  (broadcast the (…, 1) tile)
    scores_centered = binary_add(scores,
                                 unary_mul_imm(row_max, -1.0))

    # 5. exp(scores_centered)
    e = unary_exp(scores_centered)

    # 6. Numerator = e @ V
    num = binary_matmul(e, Vh_exp)

    # 7. Denominator = sum over columns (same pattern as max)
    e_prom = promote(e, rank=0)
    e_split = retile_streamify(e_prom,
                               chunk=1,
                               split_row=False)
    denom = accum_add(e_split, rank=1)

    # 8. Attention = numerator / denominator
    attn = binary_div(num, denom)

    return attn