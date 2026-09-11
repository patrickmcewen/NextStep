# RMSNorm (post‑attention) implemented with the DSL.
# -------------------------------------------------
# The input `res_add_0` is already an on‑chip stream tensor with shape
# (64, 1, 512) → stream dim 64, tile (1, 512).
# RMSNorm = x * rsqrt( mean(x²) + eps )
#
# 1. Square the tensor (element‑wise) – `unary_square`.
# 2. Sum across the column dimension of the tile (i.e. over the hidden dim).  
#    `unary_rowwise_sum` reduces dim=-1, keeping the tile‑row dim (1) and
#    producing a (64, 1, 1) stream.
# 3. Divide by the number of columns (512) to obtain the mean.  This is a
#    scalar multiplication with a constant, performed by `unary_mul_imm`.
# 4. Add epsilon (1e‑6) – `unary_add_imm`.
# 5. Compute the reciprocal square‑root – `unary_rsqrt`.
# 6. Multiply the original tensor by the rsqrt factor – `binary_mul`.
#
# The final tensor has the same stream and tile shape as the input,
# matching the required output shape (64, 1, 512).

def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    # 1) x²
    sq = unary_square(res_add_0)
    # 2) sum over hidden dimension (tile columns)
    sum_sq = unary_rowwise_sum(sq)               # shape (64, 1, 1)
    # 3) mean = sum / 512  (512 = tile column size)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / 512.0)  # divide by constant
    # 4) add epsilon
    mean_eps = unary_add_imm(mean_sq, 1e-6)
    # 5) rsqrt
    rsqrt = unary_rsqrt(mean_eps)
    # 6) x * rsqrt
    out = binary_mul(res_add_0, rsqrt)
    return out