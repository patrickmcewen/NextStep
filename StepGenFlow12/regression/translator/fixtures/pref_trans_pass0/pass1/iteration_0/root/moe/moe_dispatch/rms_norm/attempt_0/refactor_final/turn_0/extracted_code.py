# RMSNorm implementation using DSL ops.
# We compute: y = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps)
#   1. square the input (unary_square)
#   2. sum across the last tile dimension (unary_rowwise_sum) → shape (…,1,1)
#   3. divide by the tile‑column size to obtain the mean (unary_mul_imm)
#   4. add epsilon (unary_add_imm)
#   5. take reciprocal square‑root (unary_rsqrt)
#   6. multiply the factor with the original tensor (binary_mul)
# The stream shape is preserved throughout; the final tensor already matches the
# required output shape `(64, 1, 512)`, so we return it directly.

def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    eps = 1e-6
    # 1. x²
    sq = unary_square(res_add_0)
    # 2. sum over the last tile dimension (col)
    sum_sq = unary_rowwise_sum(sq)
    # 3. mean = sum / tile_col; tile_col = res_add_0.shape[-1]
    tile_col = res_add_0.shape[-1]
    mean_sq = unary_mul_imm(sum_sq, 1.0 / tile_col)
    # 4. add epsilon
    mean_sq_eps = unary_add_imm(mean_sq, eps)
    # 5. rsqrt
    rsqrt_val = unary_rsqrt(mean_sq_eps)
    # 6. scale original tensor
    out = binary_mul(res_add_0, rsqrt_val)
    return out