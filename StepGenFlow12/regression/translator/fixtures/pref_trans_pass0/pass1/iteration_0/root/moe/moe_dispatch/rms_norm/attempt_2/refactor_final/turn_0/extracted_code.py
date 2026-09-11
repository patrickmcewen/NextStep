# RMSNorm implemented with DSL ops.
# Computes:  out = x * rsqrt( mean(x^2, dim=-1, keepdim=True) + eps )
#   * x^2  -> unary_pow2
#   * sum over hidden dim (last tile dim) -> unary_rowwise_sum
#   * divide by hidden_dim (scale by 1/hidden_dim) -> unary_mul_imm
#   * add epsilon -> unary_add_imm
#   * rsqrt -> unary_rsqrt
#   * element‑wise multiply with original x -> binary_mul
def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    eps = 1e-6

    # 1. x squared
    sq = unary_pow2(res_add_0)

    # 2. Sum over the hidden dimension (last tile column)
    sum_sq = unary_rowwise_sum(sq)

    # 3. Compute the mean: scale by 1 / hidden_dim (tile column size)
    hidden_dim = res_add_0.shape[-1]          # tile column dimension
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)

    # 4. Add epsilon
    mean_eps = unary_add_imm(mean_sq, eps)

    # 5. rsqrt of the denominator
    inv_std = unary_rsqrt(mean_eps)

    # 6. Scale the original input
    out = binary_mul(res_add_0, inv_std)

    return out