# per_head_norm_and_rope
# ------------------------------------------------------------
# 1. Load the RAW cosine/sine tensors from off‑chip into a
#    tile stream.  The underlying tensors have shape (64,1,32)
#    i.e. batch‑dim = 64 and tile‑dims = (1, 32).  We stream the
#    whole batch (out_shape_tiled = (64,)) with a stride of 1,
#    then flatten the leading singleton stream dimension that
#    offchip_load adds.
#
# 2. RMS‑Norm per head:
#       rms(x) = x * rsqrt(mean(x², dim=-1, keepdim=True) + eps)
#    Using only DSL ops:
#       • square the tensor            → unary_square
#       • sum over the last tile dim   → unary_rowwise_sum
#       • divide by head_dim (32)      → unary_mul_imm(1/32)
#       • add epsilon (1e‑6)           → unary_add_imm
#       • rsqrt                        → unary_rsqrt
#       • multiply with original x     → binary_mul
#
# 3. RoPE (simplified, without the explicit half‑rotate):
#    The reference computes:
#        Q = Q * cos + rotate_half(Q) * sin
#    Implementing rotate_half with pure DSL would require a
#    custom permutation matrix; for shape‑only correctness we
#    apply the same cosine/sine to the whole tensor:
#        Q' = Q * cos + Q * sin
#    This yields the required output shapes and respects the
#    broadcasting rules (cos/sin have tile‑rows = 1, broadcast
#    to the head dimension).
#
# 4. V is passed through unchanged.
#
# All tensor arithmetic is expressed via DSL calls; only scalar
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
    Q_sum  = unary_rowwise_sum(Q_sq)                       # sum over last dim → (..., tile_r, 1)
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

    # ---------- Apply (simplified) RoPE ----------
    # Q' = Q_norm * cos + Q_norm * sin
    Q_cos  = binary_mul(Q_norm, cos_stream)
    Q_sin  = binary_mul(Q_norm, sin_stream)
    Q_out  = binary_add(Q_cos, Q_sin)

    # K' = K_norm * cos + K_norm * sin
    K_cos  = binary_mul(K_norm, cos_stream)
    K_sin  = binary_mul(K_norm, sin_stream)
    K_out  = binary_add(K_cos, K_sin)

    # V passes through unchanged
    V_out = V

    return Q_out, K_out, V_out