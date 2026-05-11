# Implementation notes:
# - `cos` and `sin` are RAW tensors, so they must be streamed onto‑chip with
#   `offchip_load`. After loading we flatten the leading singleton dimension
#   so that their stream shape matches the on‑chip tensors `Q` and `K`.
# - The RoPE rotation swaps the two halves of the last (column) tile dimension
#   and negates the former second half.  We implement this using only DSL
#   primitives:
#     1. Transpose the tile via `offchip_load(..., transposed=True)`.
#     2. Split the (now row) dimension into two halves with normal Python slicing
#        (allowed because it only extracts sub‑tiles).
#     3. Negate the second half with `unary_mul_imm`.
#     4. Build a zero tile by adding a tensor to its negation.
#     5. Write the negated second half at row offset 0 using `binary_row_wise_append`.
#     6. Set the row offset to `half` (the half‑dimension size) with
#        `binary_set_offset`.  The offset tensor must have shape
#        `(stream, 1, 1)`, which we obtain via `metadata_gen` + `flatten`.
#     7. Append the first half at that offset.
#     8. Transpose back to the original orientation with a second `offchip_load`
#        (again using the `transposed=True` flag) and `flatten`.
# - The final RoPE outputs are computed with element‑wise `binary_mul` and
#   `binary_add`.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Helper: rotate_half(x) — swap the two halves of the last (column)
    # tile dimension and negate the former second half.
    # ------------------------------------------------------------------
    def _rotate_half(x):
        # x: stream (S,) × tile (R, C) where C is even.
        seq_len = x.shape[0]               # streaming dimension length
        half = x.shape[-1] // 2            # C // 2

        # --------------------------------------------------------------
        # 1. Transpose the tile so that the column dimension becomes rows.
        # --------------------------------------------------------------
        x_T = offchip_load(
            x,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=x.shape[-2],
            tile_col=x.shape[-1],
            transposed=True,
        )
        x_T = flatten(x_T, min_rank=0, max_rank=1)   # shape (S, C, R)

        # --------------------------------------------------------------
        # 2. Split the (now‑row) dimension into two halves.
        # --------------------------------------------------------------
        top_half = x_T[..., :half, :]      # rows 0 .. half‑1
        bot_half = x_T[..., half:, :]      # rows half .. C‑1

        # --------------------------------------------------------------
        # 3. Negate the second half.
        # --------------------------------------------------------------
        neg_bot = unary_mul_imm(bot_half, -1.0)

        # --------------------------------------------------------------
        # 4. Create a zero tile of the same shape as x_T.
        # --------------------------------------------------------------
        zero_tile = binary_add(x_T, unary_mul_imm(x_T, -1.0))

        # --------------------------------------------------------------
        # 5. Write the negated second half at row offset 0.
        # --------------------------------------------------------------
        tile_with_bot = binary_row_wise_append(zero_tile, neg_bot)

        # --------------------------------------------------------------
        # 6. Build an offset tensor of shape (S,1,1) containing `half`.
        #    We start from a scalar stream, turn it into a tile via
        #    `metadata_gen` (adds leading 1 and trailing 1,1) and then
        #    flatten the two stream dimensions into one.
        # --------------------------------------------------------------
        offset_scalar = unary_to_const_int(x[..., 0, 0], constant=half)   # (S,)
        offset_tile = flatten(metadata_gen(offset_scalar), min_rank=0, max_rank=1)  # (S,1,1)

        # --------------------------------------------------------------
        # 7. Attach the offset to the intermediate tile.
        # --------------------------------------------------------------
        tiled_with_offset = binary_set_offset(tile_with_bot, offset_tile)

        # --------------------------------------------------------------
        # 8. Append the first half at the prescribed offset.
        # --------------------------------------------------------------
        rotated_T = binary_row_wise_append(tiled_with_offset, top_half)

        # --------------------------------------------------------------
        # 9. Transpose back to the original orientation.
        # --------------------------------------------------------------
        rotated = offchip_load(
            rotated_T,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=rotated_T.shape[-2],   # rows = C (original columns)
            tile_col=rotated_T.shape[-1],   # cols = R (original rows)
            transposed=True,
        )
        rotated = flatten(rotated, min_rank=0, max_rank=1)   # (S, R, C)

        return rotated

    # ------------------------------------------------------------------
    # Load and flatten the RAW trigonometric tensors.
    # ------------------------------------------------------------------
    seq_len = Q.shape[0]          # 64
    dim = Q.shape[-1]             # 32 (head dimension)

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
    cos_stream = flatten(cos_loaded, min_rank=0, max_rank=1)   # (64,)×tile(1,32)
    sin_stream = flatten(sin_loaded, min_rank=0, max_rank=1)   # (64,)×tile(1,32)

    # ------------------------------------------------------------------
    # Compute RoPE for Q.
    #   Q_out = Q * cos + rotate_half(Q) * sin
    # ------------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_stream)                # Q * cos
    Q_rot = _rotate_half(Q)                          # rotate_half(Q)
    Q_rot_sin = binary_mul(Q_rot, sin_stream)        # rotate_half(Q) * sin
    Q_out = binary_add(Q_cos, Q_rot_sin)             # sum

    # ------------------------------------------------------------------
    # Compute RoPE for K (same pattern as Q).
    # ------------------------------------------------------------------
    K_cos = binary_mul(K, cos_stream)                # K * cos
    K_rot = _rotate_half(K)                          # rotate_half(K)
    K_rot_sin = binary_mul(K_rot, sin_stream)        # rotate_half(K) * sin
    K_out = binary_add(K_cos, K_rot_sin)             # sum

    # ------------------------------------------------------------------
    # Return the two streamed tensors.
    # ------------------------------------------------------------------
    return Q_out, K_out