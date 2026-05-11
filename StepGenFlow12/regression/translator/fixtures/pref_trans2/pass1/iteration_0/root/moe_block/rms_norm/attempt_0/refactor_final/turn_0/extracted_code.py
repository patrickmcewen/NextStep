# RMSNorm implementation using only DSL operations.
# The formula is:  y = x * rsqrt(mean(x²) + eps)
# Steps:
#   1. Square the input (`unary_square`).
#   2. Sum over the column dimension (`unary_rowwise_sum`), producing a (1)‑element
#      tile per token (shape ...×1×1).
#   3. Divide by the hidden dimension to obtain the mean (multiply by 1/hidden_dim
#      via `unary_mul_imm` – the constant is a Python scalar, not a tensor).
#   4. Add the epsilon constant (`unary_add_imm`).
#   5. Compute reciprocal square‑root (`unary_rsqrt`).
#   6. Multiply the original input with the scaling factor (`binary_mul`);
#      broadcasting expands the (1)‑column tile to the full hidden dimension.
def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    eps = 1e-6

    # 1. x²
    x_sq = unary_square(res_add_0)

    # 2. sum_{c} x²   → shape (..., 1, 1)
    sum_sq = unary_rowwise_sum(x_sq)

    # 3. mean = sum / hidden_dim
    hidden_dim = res_add_0.shape[-1]            # Python int, safe to use
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)

    # 4. mean + eps
    mean_sq_eps = unary_add_imm(mean_sq, eps)

    # 5. rsqrt(mean + eps)
    scale = unary_rsqrt(mean_sq_eps)

    # 6. x * scale   (broadcast across the hidden dimension)
    out = binary_mul(res_add_0, scale)

    return out