# RMSNorm: y = x * rsqrt(mean(x**2) + eps)
# - Square the input.
# - Sum across the feature dimension (tile column) → shape (...,1,1).
# - Divide by the feature size to obtain the mean.
# - Add epsilon, take reciprocal square‑root, and multiply back onto the original tensor.
# The input `res_add_0` is already an on‑chip stream (shape (64, 1, 512)),
# so we can directly apply the DSL compute ops. The output shape matches the
# required `(64, 1, 512)`, thus no additional reshaping is needed.
def post_attn_rms_norm(res_add_0, *, out_shapes, out_perms=None):
    # x²
    sq = unary_square(res_add_0)
    # Σ x² over the feature dimension (keepdim → (...,1,1))
    sum_sq = unary_rowwise_sum(sq)
    # mean = sum / feature_dim
    feature_dim = res_add_0.shape[-1]               # 512 in this case
    inv_feat = 1.0 / feature_dim                    # scalar constant
    mean_sq = unary_mul_imm(sum_sq, inv_feat)
    # mean + eps
    eps = 1e-6
    mean_eps = unary_add_imm(mean_sq, eps)
    # rsqrt(mean + eps)
    rsqrt = unary_rsqrt(mean_eps)
    # x * rsqrt(...)
    out = binary_mul(res_add_0, rsqrt)
    return out