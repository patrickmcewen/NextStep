# RMSNorm = x * rsqrt(mean(x**2) + eps)
# The input `res_add_0` is already a tile‑stream of shape (seq_len, 1, hidden_dim).
# We square the tensor, sum across the feature dimension (the column tile), divide
# by the hidden size to obtain the mean, add epsilon, take the reciprocal square‑root,
# and finally multiply the original tensor by this scaling factor.
def rms_norm(res_add_0, *, out_shapes, out_perms=None):
    # eps used in the original PyTorch reference
    eps = 1e-6

    # Hidden dimension = product of the two tile dimensions.
    # For the given tiling this is 1 * 512 = 512.
    hidden_dim = res_add_0.shape[-2] * res_add_0.shape[-1]

    # x²
    sq = unary_square(res_add_0)

    # Sum of squares across the hidden dimension (column tile).
    # After this reduction the tile shape becomes (1, 1).
    sum_sq = unary_rowwise_sum(sq)

    # Mean of squares: divide by hidden_dim (implemented as multiplication by its reciprocal).
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)

    # Add epsilon and take rsqrt.
    rsqrt = unary_rsqrt(unary_add_imm(mean_sq, eps))

    # Scale the original tensor.
    out = binary_mul(res_add_0, rsqrt)

    return out