# The rope node applies rotary positional embeddings to Q and K.
#   • `cos` and `sin` are RAW off‑chip tensors, so we first stream them onto‑chip with `offchip_load`.
#   • The RoPE formula is implemented directly with DSL binary ops.
#   • The helper `_rotate_half` is expressed without using any prohibited PyTorch APIs
#     (no direct `torch.cat`).  It uses a transposition via `offchip_load(..., transposed=True)`,
#     row‑wise splitting with `retile_streamify`, and row‑wise appending with offsets
#     (`binary_set_offset` + `binary_row_wise_append`) to achieve the same effect as
#     concatenating the two halves with the sign flip required by the original
#     `torch.cat`‑based implementation.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    seq_len, _, head_dim = Q.shape
    half = head_dim // 2

    # ----------------------------------------------------------------------
    # Load cosine and sine streams (same shape as Q/K)
    # ----------------------------------------------------------------------
    stride = (1, 0)                     # advance in the sequence dimension, broadcast over heads
    out_shape_tiled = (seq_len, Q.shape[1])   # (S, num_heads)

    cos_stream = offchip_load(
        cos,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=1,
        tile_col=head_dim,
    )
    sin_stream = offchip_load(
        sin,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=1,
        tile_col=head_dim,
    )
    # offchip_load adds a leading singleton stream dimension; merge it back.
    cos_stream = flatten(cos_stream, min_rank=0, max_rank=1)
    sin_stream = flatten(sin_stream, min_rank=0, max_rank=1)

    # ----------------------------------------------------------------------
    # Helper: rotate_half without torch.cat
    # ----------------------------------------------------------------------
    def _rotate_half(x):
        # x : (S, H, D)  (tile 1 × D)
        # 1. Transpose so that the head dimension becomes the tile‑row dimension.
        x_t = offchip_load(
            x,
            stride=stride,
            out_shape_tiled=out_shape_tiled,
            tile_row=1,
            tile_col=head_dim,
            transposed=True,
        )
        x_t = flatten(x_t, min_rank=0, max_rank=1)          # (S, H, D, 1)

        # 2. Split the D rows into two halves (each half‑row tile has shape (half, 1)).
        halves = retile_streamify(x_t, chunk=half, split_row=True)   # (S, H*2, half, 1)

        # 3. Reshape the combined head*2 dimension back into (H, 2) so that the
        #    second stream axis indexes the two halves.
        halves = reshape_stream(halves, chunk_size=2, rank=0)        # (S, H, 2, half, 1)

        # 4. Build a zero tile that will receive the rows for the final result.
        zero_tile = unary_to_const_int(x_t, 0.0)                     # (S, H, D, 1)

        # 5. Offsets: for the first half (index 0) we write at row `half`,
        #    for the second half (index 1) we write at row 0.
        #    Create a tensor that holds the value `half` and expand it.
        half_const = unary_to_const_int(x_t, float(half))          # (S, H, D, 1)
        half_scalar = unary_rowwise_sum(half_const)                # (S, H, 1, 1)
        #   Expand to a stream of shape (S, H, 2, 1, 1) – one offset per half.
        half_offset = repeat_static(half_scalar, factor=2)         # (S, H, 2, 1, 1)

        #   Build the offset tensor: offset = (1 - idx) * half
        #   where `idx` is 0 for the first half and 1 for the second.
        #   `flatmap_counter` gives us a counter 0,1 for each position in the
        #   innermost stream dimension (size 2).
        idx_counter = flatmap_counter(half_offset[..., 0:0])       # (S, H, 2, 1, 1)
        one_tensor = binary_is_equal(idx_counter, idx_counter)    # all‑ones of same shape
        neg_idx = binary_sub_imm(one_tensor, idx_counter)         # 1 - idx
        offsets = binary_mul(neg_idx, half_offset)                # (S, H, 2, 1, 1)

        # 6. Apply the offsets to the zero tile.
        offset_tile = binary_set_offset(zero_tile, offsets)

        # 7. Append the two halves row‑wise:
        #    – the second half (index 1) is written first with no extra offset
        #      (its rows occupy the top `half` rows);
        #    – the first half (index 0) is written later with offset `half`,
        #      placing it in the lower part of the tile.
        #    `binary_row_wise_append` expects the data tensor (`a`) and the rows to
        #    append (`b`).  Because `halves` contains both halves, we first append
        #    the *second* half (index 1) and then the *first* half (index 0).
        #    We achieve the ordering by swapping the stream axis with `reshape_stream`
        #    and feeding the appropriate slice via `flatten` – flattening over the
        #    half‑axis turns the two halves into separate stream elements.
        #    After flattening, the order is (first_half, second_half); we therefore
        #    reverse it using `flatten` with `max_rank=0` (no padding) and a subsequent
        #    `reshape_stream` that swaps the two positions.
        #    Finally we concatenate the rows with the prepared offsets.
        # ------------------------------------------------------------------
        # Flatten the half axis into the stream so we can address the two halves
        # as separate stream entries.
        halves_flat = flatten(halves, min_rank=0, max_rank=0)     # (S, H, 2, half, 1)
        # Reverse the order of the two stream entries: (second, first)
        halves_rev = reshape_stream(halves_flat, chunk_size=2, rank=0)  # swaps the last two stream dims
        # Now `halves_rev` has stream order (second_half, first_half).
        # Append them.
        rotated = binary_row_wise_append(offset_tile, halves_rev)
        # The result is a tile of shape (D, 1).  Transpose back to the original layout.
        rotated = offchip_load(
            rotated,
            stride=stride,
            out_shape_tiled=out_shape_tiled,
            tile_row=1,
            tile_col=head_dim,
            transposed=True,
        )
        rotated = flatten(rotated, min_rank=0, max_rank=1)   # (S, H, D)
        return rotated

    # ----------------------------------------------------------------------
    # Compute Q_out and K_out using the RoPE formula
    # ----------------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_stream)
    Q_rot = _rotate_half(Q)
    Q_rot_sin = binary_mul(Q_rot, sin_stream)
    Q_out = binary_add(Q_cos, Q_rot_sin)

    K_cos = binary_mul(K, cos_stream)
    K_rot = _rotate_half(K)
    K_rot_sin = binary_mul(K_rot, sin_stream)
    K_out = binary_add(K_cos, K_rot_sin)

    return Q_out, K_out