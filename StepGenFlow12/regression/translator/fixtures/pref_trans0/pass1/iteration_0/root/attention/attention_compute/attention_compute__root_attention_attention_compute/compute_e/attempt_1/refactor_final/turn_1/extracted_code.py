# compute_e implements the soft‑max exponent (e = exp(scores - row_max)):
#   1. Broadcast Kh across the query‑per‑kv‑head dimension.
#   2. Compute raw attention scores = Qh @ Khᵀ via binary_matmul.
#   3. Move the column tile dimension (size 64) into the stream using
#      retile_streamify(split_row=False) so we can reduce over it.
#   4. Split that fused stream dimension into (query_per_kvhead, col_index)
#      with reshape_stream (chunk_size = seq_len = 64).
#   5. Reduce over the column‑index stream dimension with accum_max to get
#      the per‑row maximum (row_max).
#   6. Subtract row_max from the original scores (broadcast over the column
#      tile) and exponentiate the result.
# The final tensor has shape (4, 4, 64, 64), matching the declared output shape.
def compute_e(Qh, Kh, *, out_shapes, out_perms=None):
    # 1. Expand Kh so it has the same stream shape as Qh.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)          # (4, 4, 64, 32)

    # 2. Compute attention scores = Qh @ Khᵀ (tile: 64×64).
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)  # (4, 4, 64, 64)

    # 3. Pull the column tile (size 64) into the stream.
    scores_split = retile_streamify(scores, chunk=1, split_row=False)  # (4, 256, 64, 1)

    # 4. Split the fused stream dim (256) into (query_per_kvhead=4, col_index=64).
    scores_reshaped = reshape_stream(scores_split, chunk_size=64, rank=0)  # (4, 4, 64, 64, 1)

    # 5. Row‑wise max: reduce over the column‑index stream dimension.
    row_max = accum_max(scores_reshaped, rank=1)       # (4, 4, 64, 1)

    # 6. Shift scores by the max and exponentiate.
    shifted = binary_add(scores, unary_mul_imm(row_max, -1.0))
    e = unary_exp(shifted)

    return e