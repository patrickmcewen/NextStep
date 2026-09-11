def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
    """
    Full‑sequence multi‑query attention using only DSL primitives.

    Steps
    -----
    1. Broadcast the singleton KV‑head dimension of K and V to match Q’s
       `(kv_h, query_per_kv)` stream shape via ``expand_ref``.
    2. Compute raw attention scores:  ``scores = Q · Kᵀ`` (`binary_matmul` with
       ``weight_transposed=True``).
    3. Stable‑softmax – row‑wise maximum:
        * ``promote`` adds a dummy stream dim so that we can split the
          column tile dimension.
        * ``retile_streamify(..., split_row=False)`` moves each column into a
          new stream slot (chunk size 1).
        * ``accum_max`` reduces over that new dim, yielding a tensor of shape
          `(…, 1)` that can be broadcast.
    4. Subtract the max (broadcast via ``binary_add`` after negating the max
       with ``unary_mul_imm``) and apply ``exp``.
    5. Numerator = ``exp(scores) @ V`` (`binary_matmul`).
    6. Denominator = sum over the same column dimension (same split‑and‑reduce
       pattern as the max, but using ``accum_add``).
    7. Attention = numerator / denominator (`binary_div`).

    The result now has stream shape ``(kv_h, query_per_kv)`` and tile shape
    ``(seq_len, head_dim)`` – i.e. exactly the shape produced by the reference
    model before the final permute/reshape.

    8. Convert to the contract’s vanilla shape ``(seq_len, num_heads, head_dim)``:
        * ``retile_streamify(..., split_row=True)`` moves part of the tile‑row
          dimension (the sequence length) into the last stream slot.
          The chunk size is the total number of heads
          ``kv_h * query_per_kv`` (16 for the current configuration).
        * ``flatten`` merges the remaining two stream dimensions into a single
          ``seq_len`` stream dimension.

    After these two ops the combined stream‑plus‑tile shape is
    ``(64, 16, 32)`` which matches the declared ``out_shapes``.
    """
    # 1. Broadcast K and V to Q’s stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)
    Vh_exp = expand_ref(Vh, Qh, expand_rank=1)

    # 2. Scores = Q @ Kᵀ
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # 3. Row‑wise max (for numerical stability).
    scores_prom = promote(scores, rank=0)                     # add dummy stream dim
    scores_split = retile_streamify(scores_prom,
                                    chunk=1,
                                    split_row=False)      # split column dim
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

    # -----------------------------------------------------------------
    # Reshape to the contract‑specified vanilla shape (seq_len, num_heads, head_dim)
    # -----------------------------------------------------------------
    # Total number of heads = kv_h * query_per_kvhead
    num_heads = Qh.tensor.shape[0] * Qh.tensor.shape[1]   # 4 * 4 = 16

    # Move part of the tile‑row dimension (seq_len) into the stream.
    # After this: stream shape = (kv_h, query_per_kv * (seq_len // num_heads))
    #                tile shape   = (num_heads, head_dim)
    attn_retiled = retile_streamify(attn,
                                    chunk=num_heads,
                                    split_row=True)

    # Merge the two stream dimensions into a single seq_len dimension.
    attn_out = flatten(attn_retiled, min_rank=0, max_rank=1)

    return attn_out