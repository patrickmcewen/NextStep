# rms_norm implementation using STeP DSL ops.
# Input ``res_add_0`` is already on‑chip (shape (1, 64, 1, 512)).
# RMSNorm = x * rsqrt(mean(x²) + eps)
#   1. Square the input                → binary_mul(x, x)
#   2. Sum over the hidden dimension   → unary_rowwise_sum(...)
#   3. Divide by hidden_dim (mean)     → unary_mul_imm(..., 1/hidden_dim)
#   4. Add epsilon                     → unary_add_imm(..., eps)
#   5. Compute rsqrt                   → unary_rsqrt(...)
#   6. Scale original input            → binary_mul(x, rsqrt_result)
def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    # 1. x²
    sq = binary_mul(res_add_0, res_add_0)                     # (1,64,1,512)

    # 2. sum_{dim=-1} x²  → shape (1,64,1,1)
    sum_sq = unary_rowwise_sum(sq)

    # 3. mean = sum / hidden_dim
    hidden_dim = res_add_0.shape[-1]          # 512 (python int)
    inv_hidden = 1.0 / hidden_dim             # scalar float = 1/512
    mean_sq = unary_mul_imm(sum_sq, inv_hidden)

    # 4. add epsilon
    eps = 1e-6
    mean_eps = unary_add_imm(mean_sq, eps)

    # 5. rsqrt(mean + eps)
    norm_factor = unary_rsqrt(mean_eps)       # (1,64,1,1)

    # 6. x * rsqrt(...)
    out = binary_mul(res_add_0, norm_factor) # (1,64,1,512)

    return out