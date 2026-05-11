# The per‑head RMS norm is applied independently to each head (i.e.
# across the last (head‑dim) tile axis).  For an input tile stream `x` we
# compute:
#   x_norm = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps)
# This can be expressed with the DSL primitives:
#   - `unary_square`    : x**2
#   - `unary_rowwise_sum` : sum over the last tile dimension → shape (..., rows, 1)
#   - `unary_mul_imm`  : multiply by (1 / head_dim) to obtain the mean
#   - `unary_add_imm`  : add the epsilon constant
#   - `unary_rsqrt`    : reciprocal square‑root
#   - `binary_mul`     : broadcast multiply the original tensor by the scale
# The inputs `Q` and `K` are already on‑chip streams, so no off‑chip load is
# required.  The operations preserve the original stream and tile shapes,
# yielding exactly the required output shapes.
def per_head_norm(Q, K, *, out_shapes, out_perms=None):
    eps = 1e-6

    def _rms_norm(x):
        # x: (*stream, rows, cols) where cols = head_dim
        head_dim = x.shape[-1]                # number of columns in each tile
        sq = unary_square(x)                  # x**2
        sum_sq = unary_rowwise_sum(sq)        # sum over cols → (..., rows, 1)
        mean_sq = unary_mul_imm(sum_sq, constant=1.0 / head_dim)  # divide by head_dim
        mean_sq_eps = unary_add_imm(mean_sq, constant=eps)       # + eps
        scale = unary_rsqrt(mean_sq_eps)                     # rsqrt(...)
        return binary_mul(x, scale)          # broadcast multiply

    Q_norm = _rms_norm(Q)
    K_norm = _rms_norm(K)

    # out_perms are all None for this node, so we return the tensors directly.
    return Q_norm, K_norm