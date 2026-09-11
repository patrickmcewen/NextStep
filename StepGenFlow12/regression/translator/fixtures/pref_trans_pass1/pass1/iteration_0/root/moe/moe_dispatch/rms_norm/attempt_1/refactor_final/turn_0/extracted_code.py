# RMSNorm implementation using DSL primitives.
# The operation is:  y = x * rsqrt( mean(x², dim=-1, keepdim=True) + eps )
#   1. Square the input tensor (element‑wise).
#   2. Sum across the tile‑column dimension (the last tile axis) – this yields a (1,1) tile per token.
#   3. Convert the sum to a mean by multiplying with 1/num_cols (the tile‑column size).
#   4. Add the epsilon constant.
#   5. Apply reciprocal‑square‑root.
#   6. Multiply the original tensor by the scaling factor (broadcasted).
#
# All steps are expressed with the provided unary_/binary_ DSL calls; no raw
# tensor arithmetic or new torch tensors are created.

def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    eps = 1e-6

    # 1. x²
    sq = unary_square(res_add_0)

    # 2. Σ over the hidden dimension (tile columns)
    sum_sq = unary_rowwise_sum(sq)          # shape (stream…, 1, 1)

    # 3. Mean = sum / hidden_dim
    hidden_dim = int(res_add_0.shape[-1])   # tile‑column count (concrete)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)

    # 4. + eps
    mean_sq_eps = unary_add_imm(mean_sq, eps)

    # 5. rsqrt
    inv_std = unary_rsqrt(mean_sq_eps)

    # 6. x * inv_std (broadcasted)
    out = binary_mul(res_add_0, inv_std)

    return out