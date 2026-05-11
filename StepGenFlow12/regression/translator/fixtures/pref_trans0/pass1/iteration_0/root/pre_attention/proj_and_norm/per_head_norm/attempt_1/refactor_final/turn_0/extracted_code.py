# Per‑head RMSNorm implemented with DSL ops only.
# For each input tensor (shape: stream × heads × head_dim) we compute:
#   factor = 1 / sqrt(mean(x**2, dim=-1, keepdim=True) + eps)
#   output = x * factor
#   mean is obtained by summing over the last tile dimension and scaling by 1/head_dim.
# All arithmetic uses the provided DSL primitives; no raw tensor ops are used.
def per_head_norm(Q, K, *, out_shapes, out_perms=None):
    eps = 1e-6

    # ----- Q -------------------------------------------------
    # x²
    Q_sq = unary_square(Q)                       # (..., heads, head_dim)
    # Σ x² over head_dim (keepdim -> last dim = 1)
    Q_sum = unary_rowwise_sum(Q_sq)              # (..., heads, 1)
    # mean = Σ x² / head_dim
    Q_mean = unary_mul_imm(Q_sum, 1.0 / Q.shape[-1])
    # mean + eps
    Q_mean_eps = unary_add_imm(Q_mean, eps)
    # 1 / sqrt(mean + eps)
    Q_factor = unary_rsqrt(Q_mean_eps)
    # Apply factor (broadcast over head_dim)
    Q_norm = binary_mul(Q, Q_factor)

    # ----- K -------------------------------------------------
    K_sq = unary_square(K)
    K_sum = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / K.shape[-1])
    K_mean_eps = unary_add_imm(K_mean, eps)
    K_factor = unary_rsqrt(K_mean_eps)
    K_norm = binary_mul(K, K_factor)

    return Q_norm, K_norm