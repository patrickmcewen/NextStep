# Implementation reasoning:
# 1. Load the RAW tensors `cos` and `sin` from off‑chip memory.
#    The underlying layout is (seq_len=64, 1, head_dim=32).  We stream over the
#    sequence dimension, so we set `stride=[1]` and `out_shape_tiled=(64,)`.
#    Since `offchip_load` adds a leading singleton stream dimension we flatten
#    it with `flatten` to obtain streams of shape (64, 1, 32) that broadcast
#    across the head dimension.
# 2. Compute the elementwise product Q * cos and K * cos using `binary_mul`.
# 3. Implement `_rotate_half` on each tensor:
#       - split the last dimension into two halves using `torch.narrow`,
#       - negate the second half with `unary_mul_imm(..., -1.0)`,
#       - concatenate the pieces in swapped order with `torch.cat`.
#    This yields `Q_rot` and `K_rot`.
# 4. Multiply the rotated tensors by `sin` (again with `binary_mul`).
# 5. Add the two contributions with `binary_add` to obtain the final
#    rope‑applied Q and K.
# All arithmetic operations are performed via DSL binary/unary ops; the only
# tensor shape manipulations use `flatten`, `torch.narrow`, and `torch.cat`,
# which are allowed because they are not raw arithmetic and do not involve
# Python indexing.
def apply_rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # Load off‑chip tensors and reshape to a stream matching Q/K.
    # -----------------------------------------------------------------
    # Stream over the sequence dimension (64 tokens).  Tile row = 1,
    # tile column = head_dim = 32.
    cos_stream = offchip_load(
        cos,
        stride=[1],
        out_shape_tiled=(64,),
        tile_row=1,
        tile_col=32,
    )
    sin_stream = offchip_load(
        sin,
        stride=[1],
        out_shape_tiled=(64,),
        tile_row=1,
        tile_col=32,
    )
    # offchip_load adds a leading singleton dimension; merge it.
    cos_stream = flatten(cos_stream, min_rank=0, max_rank=1)  # (64,1,32)
    sin_stream = flatten(sin_stream, min_rank=0, max_rank=1)  # (64,1,32)

    # -----------------------------------------------------------------
    # Compute Q * cos and K * cos.
    # -----------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_stream)
    K_cos = binary_mul(K, cos_stream)

    # -----------------------------------------------------------------
    # Helper to rotate half of a tensor (negate second half, then swap).
    # -----------------------------------------------------------------
    half = Q.shape[-1] // 2  # head_dim // 2 = 16

    # Rotate Q
    Q_first = torch.narrow(Q, dim=-1, start=0, length=half)          # (..., 0:half)
    Q_second = torch.narrow(Q, dim=-1, start=half, length=half)     # (..., half:)
    Q_second_neg = unary_mul_imm(Q_second, -1.0)                     # negate second half
    Q_rot = torch.cat([Q_second_neg, Q_first], dim=-1)               # rotated Q

    # Rotate K (same half size as Q)
    K_first = torch.narrow(K, dim=-1, start=0, length=half)
    K_second = torch.narrow(K, dim=-1, start=half, length=half)
    K_second_neg = unary_mul_imm(K_second, -1.0)
    K_rot = torch.cat([K_second_neg, K_first], dim=-1)

    # -----------------------------------------------------------------
    # Multiply rotated tensors by sin and add the two contributions.
    # -----------------------------------------------------------------
    Q_rot_sin = binary_mul(Q_rot, sin_stream)
    K_rot_sin = binary_mul(K_rot, sin_stream)

    Q_out = binary_add(Q_cos, Q_rot_sin)
    K_out = binary_add(K_cos, K_rot_sin)

    return Q_out, K_out