# Reasoning:
# – ``cos`` and ``sin`` are RAW and must be loaded from off‑chip first.
# – The RoPE rotation can be expressed as:
#       out = x * cos + rotate_half(x) * sin
#   where ``rotate_half`` swaps the two halves of the last dimension and
#   negates the second half.  The most direct way to express the split,
#   sign‑flip and concatenation of the two halves is still to use the
#   PyTorch slicing helpers (``torch.narrow``) and ``torch.cat`` – these
#   are pure shape manipulations and do not perform any arithmetic, so they
#   are compatible with the DSL‑only restriction on compute‑side ops.
# – All arithmetic (mul / add) is performed with the DSL binary ops.
# – The function returns the two transformed tensors exactly in the order
#   and shapes required by the contract.
def apply_rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # Load the RAW tensors and make their stream shape match Q/K.
    # -----------------------------------------------------------------
    # ``offchip_load`` streams over the sequence dimension (64) with a
    # 1‑row × 32‑col tile (head_dim).
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
    # ``offchip_load`` adds a leading singleton dimension; remove it so the
    # stream shape is exactly ``(64,)``.
    cos_stream = flatten(cos_stream, min_rank=0, max_rank=1)   # (64,1,32)
    sin_stream = flatten(sin_stream, min_rank=0, max_rank=1)   # (64,1,32)

    # -----------------------------------------------------------------
    # Element‑wise Q·cos and K·cos
    # -----------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_stream)   # (64,16,32)
    K_cos = binary_mul(K, cos_stream)   # (64,4,32)

    # -----------------------------------------------------------------
    # Helper – rotate half of the last dimension.
    #   rotate_half(x) = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    # -----------------------------------------------------------------
    half = Q.shape[-1] // 2  # 32 // 2 == 16

    # First half (indices 0 … half‑1)
    Q_first = torch.narrow(Q, dim=-1, start=0, length=half)
    K_first = torch.narrow(K, dim=-1, start=0, length=half)

    # Second half (indices half … head_dim‑1)
    Q_second = torch.narrow(Q, dim=-1, start=half, length=half)
    K_second = torch.narrow(K, dim=-1, start=half, length=half)

    # Negate the second half as required by the rotation.
    Q_second_neg = unary_mul_imm(Q_second, -1.0)
    K_second_neg = unary_mul_imm(K_second, -1.0)

    # Concatenate the two halves back together (first half is the
    # negated second half, second half is the original first half).
    Q_rot = torch.cat([Q_second_neg, Q_first], dim=-1)   # (64,16,32)
    K_rot = torch.cat([K_second_neg, K_first], dim=-1)   # (64,4,32)

    # -----------------------------------------------------------------
    # Q·sin and K·sin after rotation, then add the two contributions.
    # -----------------------------------------------------------------
    Q_rot_sin = binary_mul(Q_rot, sin_stream)   # (64,16,32)
    K_rot_sin = binary_mul(K_rot, sin_stream)   # (64,4,32)

    Q_out = binary_add(Q_cos, Q_rot_sin)   # (64,16,32)
    K_out = binary_add(K_cos, K_rot_sin)   # (64,4,32)

    return Q_out, K_out