# Implementation reasoning:
# - `cos` and `sin` are RAW off‑chip tensors; they are loaded with `offchip_load`
#   using a tile of (1, head_dim) and a stream length equal to the sequence length.
# - Per‑head RMSNorm is expressed with the DSL building blocks:
#       x² → unary_square
#       sum over head_dim → unary_rowwise_sum
#       mean = sum / head_dim → unary_mul_imm (multiply by 1/head_dim)
#       ε added → unary_add_imm
#       rsqrt → unary_rsqrt
#       x * rsqrt(mean+ε) → binary_mul
# - RoPE requires the helper `rotate_half`.  Since there is no dedicated DSL op,
#   we perform the rotation on the underlying torch tensor and immediately wrap
#   the result back into a `StepTensor` preserving the original metadata.
# - The final RoPE computation follows the reference:
#       out = Q * cos + rotate_half(Q) * sin   (same for K)
#   using `binary_mul` and `binary_add`.
# - `V` is returned unchanged.
# - All arithmetic and shape manipulations are performed via DSL calls; the only
#   non‑DSL operation is the `rotate_half` tensor manipulation, which is
#   encapsulated in a tiny helper that re‑wraps the result.

def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Load the positional sin/cos tables (RAW off‑chip tensors).
    # Tile shape is (1, head_dim); stream length = seq_len.
    # ----------------------------------------------------------------------
    head_dim = Q.stream_dtype.shape[1]           # 32 for the given config
    seq_len  = Q.tensor.shape[0]                # 64

    # offchip_load arguments:
    #   stride = (1,)  – advance one tile per stream step (grid_c = 1)
    #   out_shape_tiled = (seq_len,) – one stream element per position
    cos_st = offchip_load(
        cos,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=head_dim,
    )
    sin_st = offchip_load(
        sin,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=head_dim,
    )

    # ----------------------------------------------------------------------
    # RMSNorm for Q and K (eps = 1e‑6)
    # ----------------------------------------------------------------------
    eps = 1e-6
    inv_head_dim = 1.0 / head_dim

    # ----- Q -----
    q_sq   = unary_square(Q)                                 # Q²
    q_sum  = unary_rowwise_sum(q_sq)                         # Σ Q² over head_dim  → (seq, heads, 1)
    q_mean = unary_mul_imm(q_sum, inv_head_dim)              # mean = sum / head_dim
    q_mean = unary_add_imm(q_mean, eps)                      # mean + eps
    q_rsqrt = unary_rsqrt(q_mean)                            # 1 / sqrt(mean + eps)
    q_norm = binary_mul(Q, q_rsqrt)                          # Q * rsqrt

    # ----- K -----
    k_sq   = unary_square(K)
    k_sum  = unary_rowwise_sum(k_sq)
    k_mean = unary_mul_imm(k_sum, inv_head_dim)
    k_mean = unary_add_imm(k_mean, eps)
    k_rsqrt = unary_rsqrt(k_mean)
    k_norm = binary_mul(K, k_rsqrt)

    # ----------------------------------------------------------------------
    # Helper: rotate_half implemented on the raw torch tensor,
    # then re‑wrapped as a StepTensor with the same metadata.
    # ----------------------------------------------------------------------
    def _rotate_half_step(tensor_st):
        raw = tensor_st.tensor
        half = raw.shape[-1] // 2
        rotated = torch.cat([-raw[..., half:], raw[..., :half]], dim=-1)
        return StepTensor(
            rotated,
            stream_dtype=tensor_st.stream_dtype,
            dyn_mask=tensor_st.dyn_mask,
            dyn_origins=tensor_st.dyn_origins,
            offsets=tensor_st.offsets,
        )

    # ----------------------------------------------------------------------
    # RoPE for Q and K: out = x * cos + rotate_half(x) * sin
    # ----------------------------------------------------------------------
    # Q
    q_cos_part = binary_mul(q_norm, cos_st)                  # Q * cos
    q_rot      = _rotate_half_step(q_norm)                   # rotate_half(Q)
    q_sin_part = binary_mul(q_rot, sin_st)                   # rotate_half(Q) * sin
    q_out = binary_add(q_cos_part, q_sin_part)

    # K
    k_cos_part = binary_mul(k_norm, cos_st)
    k_rot      = _rotate_half_step(k_norm)
    k_sin_part = binary_mul(k_rot, sin_st)
    k_out = binary_add(k_cos_part, k_sin_part)

    # V is passed through unchanged.
    return q_out, k_out, V