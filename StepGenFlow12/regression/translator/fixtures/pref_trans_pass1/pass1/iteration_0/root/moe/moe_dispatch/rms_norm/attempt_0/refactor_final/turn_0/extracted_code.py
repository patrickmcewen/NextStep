# RMSNorm implementation using DSL ops.
#   res_add_0 : on‑chip StepTensor with shape (seq_len, 1, hidden) → (64,1,512)
#   RMSNorm = x * rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)
#   • Square the tensor (unary_square)
#   • Sum across the hidden dimension (unary_rowwise_sum)
#   • Divide by hidden_dim to obtain the mean (unary_mul_imm with 1/hidden_dim)
#   • Add epsilon (unary_add_imm)
#   • Compute reciprocal square‑root (unary_rsqrt)
#   • Multiply the original tensor by this scaling factor (binary_mul)
# The hidden dimension is taken from the tile‑column size (static).
def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    # Static hidden dimension (tile column size)
    hidden_dim = res_add_0.shape[-1]          # e.g. 512
    eps = 1e-6

    # x²
    sq = unary_square(res_add_0)

    # Σ_{hidden} x²  → shape (...,1,1)
    sum_sq = unary_rowwise_sum(sq)

    # (1/hidden_dim) * Σ x²  → mean of squares
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)

    # mean + eps
    mean_eps = unary_add_imm(mean_sq, eps)

    # 1 / sqrt(mean + eps)
    inv_rsqrt = unary_rsqrt(mean_eps)

    # x * (1 / sqrt(...))
    out = binary_mul(res_add_0, inv_rsqrt)

    return out