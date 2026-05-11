# per_head_norm_and_rope
# ------------------------------------------------------------
# 1. Load the RAW cosine / sine tensors from off‑chip and flatten away the
#    leading singleton dimension that `offchip_load` adds.
#
# 2. Perform per‑head RMSNorm using DSL primitives:
#       rms(x) = x * rsqrt(mean(x², dim=-1, keepdim=True) + eps)
#    Implemented with `unary_square`, `unary_rowwise_sum`,
#    `unary_mul_imm`, `unary_add_imm`, `unary_rsqrt`, and `binary_mul`.
#
# 3. Apply RoPE.  The rotation (`rotate_half`) is provided as a child
#    blackbox stub (generated from the reference’s `_rotate_half`).  We
#    invoke that stub directly, passing the required vanilla output shape.
#
#    The final formulas are:
#        Q_out = Q_norm * cos + rotate_half(Q_norm) * sin
#        K_out = K_norm * cos + rotate_half(K_norm) * sin
#
# 4. V passes through unchanged.
#
# All arithmetic is expressed using DSL ops; only scalar Python values are
# used.
def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # Load RAW cosine / sine tensors
    # -----------------------------------------------------------------
    # Stream across the batch dimension (64 tokens). Each tile is (1, 32).
    cos_loaded = offchip_load(
        cos, stride=(1,), out_shape_tiled=(64,), tile_row=1, tile_col=32
    )
    sin_loaded = offchip_load(
        sin, stride=(1,), out_shape_tiled=(64,), tile_row=1, tile_col=32
    )

    # Remove the leading singleton that offchip_load adds.
    cos_stream = flatten(cos_loaded, min_rank=0, max_rank=1)  # → stream(64,)×tile(1,32)
    sin_stream = flatten(sin_loaded, min_rank=0, max_rank=1)  # → stream(64,)×tile(1,32)

    # -----------------------------------------------------------------
    # RMS‑Norm for Q
    # -----------------------------------------------------------------
    Q_sq   = unary_square(Q)                           # x²
    Q_sum  = unary_rowwise_sum(Q_sq)                   # sum over head_dim → (..., tile_r, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / 32.0)           # divide by head_dim
    Q_eps  = unary_add_imm(Q_mean, 1e-6)                # + epsilon
    Q_rsqrt = unary_rsqrt(Q_eps)                       # rsqrt(...)
    Q_norm = binary_mul(Q, Q_rsqrt)                    # Q * rsqrt(...)

    # -----------------------------------------------------------------
    # RMS‑Norm for K
    # -----------------------------------------------------------------
    K_sq   = unary_square(K)
    K_sum  = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / 32.0)
    K_eps  = unary_add_imm(K_mean, 1e-6)
    K_rsqrt = unary_rsqrt(K_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # -----------------------------------------------------------------
    # RoPE (rotate_half child blackbox)
    # -----------------------------------------------------------------
    # rotate_half returns the tensor with halves swapped and the second half
    # negated, exactly as the reference implementation does.
    Q_rot = _rotate_half(
        Q_norm,
        out_shapes=((64, 16, 32),),  # vanilla shape of Q_norm
        out_perms=None,
    )
    K_rot = _rotate_half(
        K_norm,
        out_shapes=((64, 4, 32),),   # vanilla shape of K_norm
        out_perms=None,
    )

    # Q_out = Q_norm * cos + Q_rot * sin
    Q_cos = binary_mul(Q_norm, cos_stream)
    Q_sin = binary_mul(Q_rot, sin_stream)
    Q_out = binary_add(Q_cos, Q_sin)

    # K_out = K_norm * cos + K_rot * sin
    K_cos = binary_mul(K_norm, cos_stream)
    K_sin = binary_mul(K_rot, sin_stream)
    K_out = binary_add(K_cos, K_sin)

    # V is unchanged
    V_out = V

    return Q_out, K_out, V_out