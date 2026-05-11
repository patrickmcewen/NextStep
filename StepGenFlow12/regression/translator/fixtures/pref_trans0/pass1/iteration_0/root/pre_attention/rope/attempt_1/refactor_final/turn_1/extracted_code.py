# The RoPE (rotary positional embedding) node:
#   • `cos` and `sin` are RAW off‑chip tensors, so we first stream them onto‑chip with `offchip_load`.
#   • `offchip_load` adds a leading singleton stream dimension; we merge that with the token
#     dimension using `flatten` so that the resulting stream shape matches `Q`/`K` (which have
#     stream shape `(seq_len,)`).
#   • RoPE formula:  out = x * cos + rotate_half(x) * sin.
#   • `rotate_half` swaps the two halves of the last dimension and negates the second half.
#     It is implemented with slicing and `torch.cat` (shape manipulation only).
#   • All arithmetic (`*`, `+`) is performed with DSL binary ops (`binary_mul`, `binary_add`).
#   • The function returns the two tensors with the required output shapes.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # Q: (seq_len, num_heads, head_dim)
    # K: (seq_len, num_kv_heads, head_dim)
    seq_len, _, head_dim = Q.shape

    # Load cos/sin as streams with shape (seq_len,) → stream (seq_len,) and tile (1, head_dim)
    # stride (1,) moves to the next token; no extra tiling over heads.
    stride = (1,)
    out_shape_tiled = (seq_len,)
    tile_row = 1
    tile_col = head_dim

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

    # `offchip_load` prepends a leading singleton stream dimension.
    # Merge that leading 1 with the token dimension so the stream shape becomes (seq_len,).
    cos_stream = flatten(cos_stream, min_rank=0, max_rank=1)
    sin_stream = flatten(sin_stream, min_rank=0, max_rank=1)

    # Helper: rotate the last dimension by swapping halves and negating the second half.
    def _rotate_half(x):
        half = x.shape[-1] // 2
        first = x[..., :half]
        second = x[..., half:]
        neg_second = unary_mul_imm(second, -1.0)
        return torch.cat([neg_second, first], dim=-1)

    # ---- Q ----
    q_cos = binary_mul(Q, cos_stream)                # Q * cos
    q_rot = _rotate_half(Q)                          # rotate_half(Q)
    q_rot_sin = binary_mul(q_rot, sin_stream)        # rotate_half(Q) * sin
    Q_out = binary_add(q_cos, q_rot_sin)             # Q * cos + rotate_half(Q) * sin

    # ---- K ----
    k_cos = binary_mul(K, cos_stream)                # K * cos
    k_rot = _rotate_half(K)                          # rotate_half(K)
    k_rot_sin = binary_mul(k_rot, sin_stream)        # rotate_half(K) * sin
    K_out = binary_add(k_cos, k_rot_sin)             # K * cos + rotate_half(K) * sin

    return Q_out, K_out