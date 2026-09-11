# Implementation notes:
# 1. RMSNorm is built from square → sum → (1/head_dim) → +eps → rsqrt → mul.
# 2. The RAW tensors `cos` and `sin` are loaded with `offchip_load` and then flattened
#    so that their stream shape matches that of Q/K ((seq_len,)).
# 3. The half‑rotation required by RoPE is delegated to the child blackbox `_rotate_half`,
#    which implements the exact `torch.cat([-x[..., half:], x[..., :half]], dim=-1)`
#    operation.  The child receives the normalized tensor and is asked to emit the same
#    tiled shape.
# 4. RoPE is then applied as Q = Q_norm * cos + _rotate_half(Q_norm) * sin,
#    and similarly for K. V is passed through unchanged.
# 5. `out_shapes` and `out_perms` are part of the required signature but are not needed
#    in the internal DSL logic.

def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    eps = 1e-6  # RMSNorm epsilon

    # ---------- RMSNorm for Q ----------
    Q_sq = binary_mul(Q, Q)                                 # x * x
    Q_sum = unary_rowwise_sum(Q_sq)                         # sum over last (head) dim
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

    # ---------- Load positional tensors ----------
    seq_len = Q.shape[0]                     # sequence length (e.g. 64)
    stride = (1,)                            # linear indexing stride
    out_shape = (seq_len,)                   # streaming shape
    tile_row = 1
    tile_col = cos.shape[-1]                 # head dimension (32)

    cos_loaded = offchip_load(cos, stride, out_shape, tile_row, tile_col)
    cos_loaded = flatten(cos_loaded, min_rank=0, max_rank=1)

    sin_loaded = offchip_load(sin, stride, out_shape, tile_row, tile_col)
    sin_loaded = flatten(sin_loaded, min_rank=0, max_rank=1)

    # ---------- Rotate half (child blackbox) ----------
    Q_rot = _rotate_half(Q_norm,
                         out_shapes=(Q_norm.shape,),
                         out_perms=(None,))
    K_rot = _rotate_half(K_norm,
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