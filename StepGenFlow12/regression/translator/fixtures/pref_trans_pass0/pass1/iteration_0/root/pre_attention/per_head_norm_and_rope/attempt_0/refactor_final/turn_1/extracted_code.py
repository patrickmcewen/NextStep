# per_head_norm_and_rope
# ------------------------------------------------------------
# 1. Load the RAW cosine / sine tensors from off‑chip.
#    They have shape (64,1,32); we stream the 64 batch dimension,
#    then flatten away the leading singleton that offchip_load adds.
#
# 2. RMS‑Norm per head (mean over the last dimension):
#       mean = (x²).sum(dim=-1, keepdim=True) / head_dim
#       norm = 1 / sqrt(mean + eps)
#       x_norm = x * norm
#    Implemented with unary_square, unary_rowwise_sum,
#    unary_mul_imm (divide by head_dim), unary_add_imm (eps),
#    unary_rsqrt, and binary_mul.
#
# 3. RoPE:
#       q_out = q_norm * cos + rotate_half(q_norm * sin)
#       k_out = k_norm * cos + rotate_half(k_norm * sin)
#    The rotation “rotate_half” is performed by a child blackbox
#    (the reference defines it as a pure PyTorch function).  We
#    invoke that child directly; the stub will handle flatten/reshape
#    of the stream tensor, so no DSL transforms sit between the
#    tensor and the child call.
#
# 4. V passes through unchanged.
#
# All tensor arithmetic is expressed via DSL ops; only scalar
# Python arithmetic (constants, eps) is used.
def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    # ---------- Load RAW cos / sin ----------
    # stride=1 streams across the batch dimension; one tile per element.
    cos_loaded = offchip_load(cos, stride=(1,), out_shape_tiled=(64,), tile_row=1, tile_col=32)
    sin_loaded = offchip_load(sin, stride=(1,), out_shape_tiled=(64,), tile_row=1, tile_col=32)

    # offchip_load adds a leading singleton stream dim; flatten it away.
    cos_stream = flatten(cos_loaded, min_rank=0, max_rank=1)
    sin_stream = flatten(sin_loaded, min_rank=0, max_rank=1)

    # ---------- RMSNorm for Q ----------
    Q_sq   = unary_square(Q)                               # x²
    Q_sum  = unary_rowwise_sum(Q_sq)                       # sum over head_dim → (..., tile_r, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / 32.0)               # divide by head_dim
    Q_eps  = unary_add_imm(Q_mean, 1e-6)                    # + epsilon
    Q_rsqrt = unary_rsqrt(Q_eps)                           # rsqrt
    Q_norm = binary_mul(Q, Q_rsqrt)                        # x * rsqrt(...)

    # ---------- RMSNorm for K ----------
    K_sq   = unary_square(K)
    K_sum  = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / 32.0)
    K_eps  = unary_add_imm(K_mean, 1e-6)
    K_rsqrt = unary_rsqrt(K_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # ---------- RoPE for Q ----------
    Q_cos     = binary_mul(Q_norm, cos_stream)                           # Q * cos
    Q_sin_raw = binary_mul(Q_norm, sin_stream)                           # Q * sin
    # rotate_half is a child blackbox; its stub will handle the true PyTorch code.
    Q_sin_rot = _rotate_half(Q_sin_raw, out_shapes=((64, 16, 32),), out_perms=None)
    Q_out     = binary_add(Q_cos, Q_sin_rot)

    # ---------- RoPE for K ----------
    K_cos     = binary_mul(K_norm, cos_stream)                           # K * cos
    K_sin_raw = binary_mul(K_norm, sin_stream)                           # K * sin
    K_sin_rot = _rotate_half(K_sin_raw, out_shapes=((64, 4, 32),), out_perms=None)
    K_out     = binary_add(K_cos, K_sin_rot)

    # V is unchanged
    V_out = V

    return Q_out, K_out, V_out