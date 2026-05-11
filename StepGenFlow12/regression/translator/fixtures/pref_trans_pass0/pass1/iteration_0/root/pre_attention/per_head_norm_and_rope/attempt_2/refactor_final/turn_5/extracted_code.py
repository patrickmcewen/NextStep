# NOTE:
# - RMSNorm is built from DSL ops: square → sum → scale → +eps → rsqrt → mul.
# - `cos` and `sin` are RAW inputs; they are streamed onto‑chip with `offchip_load`
#   and flattened so their stream shape matches that of Q/K (seq_len,).
# - The half‑rotation required by RoPE is delegated to a child blackbox `rotate_half`.
#   The blackbox is defined below and may freely use regular torch operations
#   (it is not subject to the DSL‑only restriction of the parent function).
# - RoPE is applied as Q = Q_norm * cos + rotate_half(Q_norm) * sin (similarly for K).
# - V passes through unchanged.
# - `out_shapes` and `out_perms` are required by the signature but are not used
#   inside the computation.

def rotate_half(x, *, out_shapes, out_perms=None):
    """Child blackbox that implements the half‑rotation used in RoPE.

    The implementation uses ordinary torch indexing and arithmetic, which is
    permitted inside a blackbox stub.  The input `x` is a tiled stream with shape
    (… , tile_rows, tile_cols); the operation swaps the two halves of the last
    dimension and negates the former second half.
    """
    half = x.shape[-1] // 2
    # Negate the second half and concatenate with the first half.
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    eps = 1e-6  # RMSNorm epsilon

    # ---------- RMSNorm for Q ----------
    Q_sq = binary_mul(Q, Q)                                 # x * x
    Q_sum = unary_rowwise_sum(Q_sq)                         # sum over head_dim (tile cols)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / Q.shape[-1])        # divide by head_dim
    Q_mean_eps = unary_add_imm(Q_mean, eps)                 # add epsilon
    Q_rsqrt = unary_rsqrt(Q_mean_eps)                       # rsqrt(...)
    Q_norm = binary_mul(Q, Q_rsqrt)                         # x * rsqrt(...)

    # ---------- RMSNorm for K ----------
    K_sq = binary_mul(K, K)
    K_sum = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / K.shape[-1])
    K_mean_eps = unary_add_imm(K_mean, eps)
    K_rsqrt = unary_rsqrt(K_mean_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # ---------- Load positional tensors (cos, sin) ----------
    seq_len = Q.shape[0]                     # sequence length (e.g., 64)
    stride = (1,)                            # linear indexing stride
    out_shape = (seq_len,)                   # streaming shape for the load
    tile_row = 1
    tile_col = cos.shape[-1]                 # head_dim (32)

    cos_loaded = offchip_load(cos, stride, out_shape, tile_row, tile_col)
    cos_loaded = flatten(cos_loaded, min_rank=0, max_rank=1)

    sin_loaded = offchip_load(sin, stride, out_shape, tile_row, tile_col)
    sin_loaded = flatten(sin_loaded, min_rank=0, max_rank=1)

    # ---------- Rotate half (child blackbox) ----------
    Q_rot = rotate_half(Q_norm,
                        out_shapes=(Q_norm.shape,),
                        out_perms=(None,))
    K_rot = rotate_half(K_norm,
                        out_shapes=(K_norm.shape,),
                        out_perms=(None,))

    # ---------- Apply RoPE ----------
    Q_cos = binary_mul(Q_norm, cos_loaded)
    Q_sin = binary_mul(Q_rot, sin_loaded)
    Q_out = binary_add(Q_cos, Q_sin)

    K_cos = binary_mul(K_norm, cos_loaded)
    K_sin = binary_mul(K_rot, sin_loaded)
    K_out = binary_add(K_cos, K_sin)

    # V is unchanged
    V_out = V

    return Q_out, K_out, V_out