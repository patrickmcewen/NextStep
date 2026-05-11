# Implementation reasoning:
# 1. `Kh` has a trailing stream dimension of size 1 while `Qh` has size 4.
#    We broadcast `Kh` to match `Qh` using `expand_ref`.
# 2. Compute the raw attention scores with a transposed matmul:
#    `scores = Qh @ Khᵀ` via `binary_matmul(..., weight_transposed=True)`.
# 3. To obtain a per‑row maximum across the column dimension we:
#    - move the column tiles into the stream with `retile_streamify(..., split_row=False, chunk=1)`;
#      this turns the tile‑column dimension into a stream dimension, merging it with the
#      second stream dimension.
#    - split the merged stream dimension back into the original second stream dimension
#      and an explicit column‑index stream using `reshape_stream` (chunk size = number of columns).
#    - reduce over this new column‑index stream with `accum_max(rank=1)`, yielding `row_max`
#      of shape (4, 4, 64, 1).
# 4. Shift the scores by the row‑wise max (`scores - row_max`) using `binary_add`
#    together with `unary_mul_imm(row_max, -1.0)`.
# 5. Apply element‑wise exponentiation via `unary_exp`.
# 6. The result has the required shape (4, 4, 64, 64).

def compute_e(Qh, Kh, *, out_shapes, out_perms=None):
    # 1. Broadcast Kh to match Qh's stream shape.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)

    # 2. Compute scores = Qh @ Khᵀ.
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # 3. Move column tiles into the stream (tile_c → stream).
    scores_col_stream = retile_streamify(scores, chunk=1, split_row=False)

    # 3a. Split the combined stream dimension (S1 × col) back into (S1, col).
    col_len = scores.shape[-2]  # original number of columns (64)
    scores_reshaped = reshape_stream(scores_col_stream, chunk_size=col_len, rank=0)

    # 3b. Compute per‑row max over the column stream dimension.
    row_max = accum_max(scores_reshaped, rank=1)

    # 4. scores – row_max (broadcast column dimension).
    shifted = binary_add(scores, unary_mul_imm(row_max, -1.0))

    # 5. Exponentiate.
    e = unary_exp(shifted)

    return e