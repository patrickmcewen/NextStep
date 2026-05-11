# post_attn_rms_norm implements RMSNorm as:
#   out = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps)
# Steps (all using DSL ops):
#   1. Square the input (unary_square)
#   2. Sum the squares across the tile‑column dimension (unary_rowwise_sum)
#   3. Divide by the number of columns to get the mean (unary_mul_imm with 1/col)
#   4. Add epsilon (unary_add_imm)
#   5. Compute reciprocal sqrt (unary_rsqrt)
#   6. Multiply the original tensor by the rsqrt term (binary_mul)
# The input `res_add_0` is already an on‑chip stream (shape (64,1,512)),
# so no off‑chip loading is needed.
def post_attn_rms_norm(res_add_0, *, out_shapes, out_perms=None):
    eps = 1e-6

    # 1. x²
    x_sq = unary_square(res_add_0, compute_bw=1)

    # 2. sum_{c} x²  (keep dim -> shape (S,1,1))
    sum_sq = unary_rowwise_sum(x_sq, compute_bw=1)

    # 3. mean = sum / C  ; C is the tile‑column size
    tile_col = res_add_0.shape[-1]          # Python scalar, allowed
    mean_sq = unary_mul_imm(sum_sq, 1.0 / tile_col, compute_bw=1)

    # 4. add epsilon
    mean_eps = unary_add_imm(mean_sq, eps, compute_bw=1)

    # 5. rsqrt(mean + eps)
    rsqrt_val = unary_rsqrt(mean_eps, compute_bw=1)

    # 6. x * rsqrt(...)
    out = binary_mul(res_add_0, rsqrt_val, compute_bw=1)

    return out