# Implementation notes:
# - `cos` and `sin` are RAW (off‑chip) tensors, so they must be loaded with
#   `offchip_load` before any DSL consumer sees them.
# - `offchip_load` emits a leading singleton stream dimension; we flatten it
#   together with the actual streaming dimension (`seq_len`) using `flatten`
#   so that its stream shape matches the on‑chip tensors `Q` and `K`
#   (both have stream shape `(seq_len,)`).
# - The RoPE computation is
#       Q_out = Q * cos + rotate_half(Q) * sin
#       K_out = K * cos + rotate_half(K) * sin
#   where `rotate_half` swaps the two halves of the last dimension and negates
#   the former second half.  The helper is expressed with plain PyTorch ops
#   (`torch.cat`, slicing, and unary negation); the arithmetic parts are
#   expressed with the DSL binary primitives (`binary_mul`, `binary_add`).
# - All arithmetic (`*`, `+`) is replaced by the corresponding DSL calls.
# - The function returns a tuple `(Q_out, K_out)` whose shapes conform to the
#   contract `(64, 16, 32)` and `(64, 4, 32)` respectively.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Helper: rotate_half(x) – swaps the two halves of the last dimension.
    # ----------------------------------------------------------------------
    def _rotate_half(x):
        # x: (..., D) where D is even.
        half = x.shape[-1] // 2
        # First half (real) stays, second half (imag) is negated and moved to front.
        # The slicing uses Python indexing (required for this simple rearrangement).
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    # ----------------------------------------------------------------------
    # Load `cos` and `sin` from off‑chip memory.
    # ----------------------------------------------------------------------
    # All tensors share the same sequence length, which is the first element
    # of the first output shape.
    seq_len = out_shapes[0][0]          # 64 for the given config
    dim = out_shapes[0][2]              # 32 (head_dim)

    # Load with stride 1 so each logical position maps to the corresponding tile.
    # The tile size matches the raw tensor: (1, dim).
    cos_loaded = offchip_load(
        cos,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=dim,
    )
    sin_loaded = offchip_load(
        sin,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=dim,
    )

    # `offchip_load` adds a leading singleton stream dimension (shape (1, seq_len, 1, dim)).
    # Flatten it together with the real streaming dimension so that its stream shape is (seq_len,).
    cos_stream = flatten(cos_loaded, min_rank=0, max_rank=1)
    sin_stream = flatten(sin_loaded, min_rank=0, max_rank=1)

    # ----------------------------------------------------------------------
    # Compute Q_out = Q * cos + rotate_half(Q) * sin
    # ----------------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_stream)                     # Q * cos
    Q_rot = _rotate_half(Q)                               # rotate_half(Q)
    Q_rot_sin = binary_mul(Q_rot, sin_stream)             # rotate_half(Q) * sin
    Q_out = binary_add(Q_cos, Q_rot_sin)                  # sum the two terms

    # ----------------------------------------------------------------------
    # Compute K_out = K * cos + rotate_half(K) * sin
    # ----------------------------------------------------------------------
    K_cos = binary_mul(K, cos_stream)                     # K * cos
    K_rot = _rotate_half(K)                               # rotate_half(K)
    K_rot_sin = binary_mul(K_rot, sin_stream)             # rotate_half(K) * sin
    K_out = binary_add(K_cos, K_rot_sin)                  # sum the two terms

    # ----------------------------------------------------------------------
    # Return the two streamed tensors.
    # ----------------------------------------------------------------------
    return Q_out, K_out