# compute_e implements the stable softmax exponent part:
#   1. Expand Kh to broadcast across the query‑per‑kvhead dimension.
#   2. Compute attention scores = Qh @ Khᵀ with binary_matmul.
#   3. Move the column tile dimension into a stream dimension (retile_streamify, split_col).
#   4. Reshape that fused stream dimension so that the column index becomes its own stream dim
#      (reshape_stream with chunk_size = seq_len = 64).
#   5. Reduce over the column‑index stream dimension using accum_max → row‑wise max.
#   6. Subtract the max from the original scores (unary_mul_imm + binary_add).
#   7. Exponentiate the shifted scores (unary_exp).
# The resulting tensor has shape (4, 4, 64, 64), matching the declared output shape.
def compute_e(Qh, Kh, *, out_shapes, out_perms=None):
    # 1. Broadcast Kh across the query‑per‑kvhead stream dimension.
    Kh_exp = expand_ref(Kh, Qh, expand_rank=1)  # (4,4,64,32)

    # 2. Scores = Qh @ Khᵀ   (tile dims: 64×32  @  32×64 → 64×64)
    scores = binary_matmul(Qh, Kh_exp, weight_transposed=True)

    # 3. Pull the column tile (size 64) into the stream.
    #    After this, tile_c becomes 1 and the column index is merged into the stream.
    scores_split = retile_streamify(scores, chunk=1, split_col=True)  # (4,256,64,1)

    # 4. Split the fused stream dimension (256 = 4 × 64) into
    #    (query_per_kvhead = 4) and (col_index = 64).
    scores_reshaped = reshape_stream(scores_split, chunk_size=64, rank=0)  # (4,4,64,64,1)

    # 5. Max over the column‑index stream dimension (the last stream dim).
    row_max = accum_max(scores_reshaped, rank=1)  # (4,4,64,1)

    # 6. Subtract the max from the original scores.
    #    row_max is broadcast over the tile's column dimension.
    shifted = binary_add(scores, unary_mul_imm(row_max, -1.0))

    # 7. Exponentiate.
    e = unary_exp(shifted)

    return e