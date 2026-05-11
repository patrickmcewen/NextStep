# Implementation notes:
# - `cos` and `sin` are RAW tensors, so they must be streamed onto‑chip with `offchip_load`.
#   After loading we `flatten` the leading singleton dimension so that their stream
#   shape matches the on‑chip `Q`/`K` tensors (shape (seq_len,)×tile(...)).
# - The RoPE rotation `rotate_half` swaps the two halves of the last (column) tile
#   dimension and negates the former second half.  This is implemented with only
#   DSL primitives:
#     1. Transpose the tile (swap rows ↔ columns) via `offchip_load(..., transposed=True)`.
#        After the transposition the half‑dimension becomes the *row* dimension.
#     2. Extract the two row‑halves with Python slicing (allowed because it only
#        selects sub‑tiles, not arithmetic).
#     3. Negate the second half using `unary_mul_imm`.
#     4. Build a zero tile by adding a tensor to its negation.
#     5. Write the negated second half at row offset 0 and the first half at row
#        offset half using `binary_row_wise_append` together with `binary_set_offset`.
#        Offsets are generated with `unary_to_const_int`, which creates a tensor
#        filled with the integer offset while preserving the stream shape.
#     6. Transpose back to the original orientation with a second `offchip_load`
#        (again using the `transposed=True` flag) and `flatten`.
# - With `rotate_half` expressed as a pure DSL transformation we can compute the
#   final RoPE outputs using element‑wise `binary_mul` and `binary_add`.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Helper: rotate_half(x) – swaps the two halves of the last (column) tile
    # dimension and negates the former second half.  Implemented entirely with
    # DSL primitives; see the notes above.
    # ----------------------------------------------------------------------
    def _rotate_half(x):
        # x: stream (S,) × tile (R, C) where C is even.
        seq_len = x.shape[0]          # streaming dimension length
        half = x.shape[-1] // 2       # C // 2

        # Transpose the tile so that the column dimension becomes a row
        # dimension that we can manipulate with row‑wise primitives.
        x_T = offchip_load(
            x,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=x.shape[-2],
            tile_col=x.shape[-1],
            transposed=True,
        )
        x_T = flatten(x_T, min_rank=0, max_rank=1)   # shape (S, C, R)

        # Split the (now‑row) dimension into two halves.
        top_half = x_T[..., :half, :]    # first half rows  (S, half, R)
        bot_half = x_T[..., half:, :]    # second half rows (S, half, R)

        # Negate the second half.
        neg_bot = unary_mul_imm(bot_half, -1.0)

        # Create a zero tile of the same shape as x_T.
        zero_tile = binary_add(x_T, unary_mul_imm(x_T, -1.0))

        # Write the negated second half at row offset 0.
        tile_with_bot = binary_row_wise_append(zero_tile, neg_bot)

        # Offsets for the next append (place the first half starting at `half`).
        # We generate an offset tensor of shape (S,1,1) filled with the integer
        # value `half`.  `unary_to_const_int` creates a float tensor; the DSL
        # conversion to integer happens inside `binary_set_offset`.
        offset_half = unary_to_const_int(x[..., 0, 0], constant=half)

        # Attach the offset to the intermediate tile.
        tiled_with_offset = binary_set_offset(tile_with_bot, offset_half)

        # Append the first half at the prescribed offset.
        rotated_T = binary_row_wise_append(tiled_with_offset, top_half)

        # Transpose back to the original orientation.
        rotated = offchip_load(
            rotated_T,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=rotated_T.shape[-2],   # now rows = C (original cols)
            tile_col=rotated_T.shape[-1],   # now cols = R (original rows)
            transposed=True,
        )
        rotated = flatten(rotated, min_rank=0, max_rank=1)   # shape (S, R, C)

        return rotated

    # ----------------------------------------------------------------------
    # Load and flatten the RAW trigonometric tensors.
    # ----------------------------------------------------------------------
    seq_len = Q.shape[0]                     # streaming dimension (64)
    dim = Q.shape[-1]                        # head dimension (32)

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

    # ----------------------------------------------------------------------
    # Compute RoPE for Q.
    #   Q_out = Q * cos + rotate_half(Q) * sin
    # ----------------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_stream)                # Q * cos
    Q_rot = _rotate_half(Q)                          # rotate_half(Q)
    Q_rot_sin = binary_mul(Q_rot, sin_stream)        # rotate_half(Q) * sin
    Q_out = binary_add(Q_cos, Q_rot_sin)             # sum

    # ----------------------------------------------------------------------
    # Compute RoPE for K (same pattern as Q).
    # ----------------------------------------------------------------------
    K_cos = binary_mul(K, cos_stream)                # K * cos
    K_rot = _rotate_half(K)                          # rotate_half(K)
    K_rot_sin = binary_mul(K_rot, sin_stream)        # rotate_half(K) * sin
    K_out = binary_add(K_cos, K_rot_sin)             # sum

    # ----------------------------------------------------------------------
    # Return the two streamed tensors.
    # ----------------------------------------------------------------------
    return Q_out, K_out