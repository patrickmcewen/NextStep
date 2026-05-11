# The rope node applies rotary positional embeddings.
# • cos / sin are off‑chip tensors, so they are first streamed onto‑chip with `offchip_load`.
#   We stream them to the same (seq_len, num_heads) layout as Q/K using stride (1,0) – the
#   head dimension is broadcast (stride = 0) while the sequence dimension advances (stride = 1).
# • The RoPE formula is  
#       out = x * cos + rotate_half(x) * sin  
#   where `rotate_half` swaps the two halves of the last dimension and negates the
#   second half.
# • Element‑wise arithmetic is expressed with the DSL binary ops (`binary_mul`, `binary_add`);
#   the sign‑flip is performed with `unary_mul_imm`.
# • `rotate_half` uses plain PyTorch slicing and `torch.cat` because the DSL does not
#   provide a dedicated column‑wise concat primitive.  Slicing is a view operation
#   (no new data) and `torch.cat` only recombines existing tiles, which is acceptable
#   under the current DSL capabilities.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # Tile dimensions (head dim)
    seq_len, num_heads, head_dim = Q.shape  # Q shape: (S, H, D)

    # Load cosine and sine into on‑chip streams that broadcast over heads.
    #   stride = (1, 0) advances per‑token, repeats the same tile for each head.
    stride = (1, 0)
    out_shape_tiled = (seq_len, num_heads)
    tile_row = 1           # cos / sin have a single row per token
    tile_col = head_dim    # same column size as Q/K

    cos_stream = offchip_load(
        cos,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=tile_row,
        tile_col=tile_col,
    )
    sin_stream = offchip_load(
        sin,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=tile_row,
        tile_col=tile_col,
    )

    # Helper: rotate the last dimension by swapping halves and negating the second half.
    def _rotate_half(x):
        half = x.shape[-1] // 2
        first_half = x[..., :half]
        second_half = x[..., half:]
        # Negate the second half then concatenate.
        neg_second = unary_mul_imm(second_half, -1.0)
        return torch.cat([neg_second, first_half], dim=-1)

    # --- Q ---------------------------------------------------------
    Q_cos = binary_mul(Q, cos_stream)
    Q_rot = _rotate_half(Q)
    Q_rot_sin = binary_mul(Q_rot, sin_stream)
    Q_out = binary_add(Q_cos, Q_rot_sin)

    # --- K ---------------------------------------------------------
    K_cos = binary_mul(K, cos_stream)
    K_rot = _rotate_half(K)
    K_rot_sin = binary_mul(K_rot, sin_stream)
    K_out = binary_add(K_cos, K_rot_sin)

    return Q_out, K_out