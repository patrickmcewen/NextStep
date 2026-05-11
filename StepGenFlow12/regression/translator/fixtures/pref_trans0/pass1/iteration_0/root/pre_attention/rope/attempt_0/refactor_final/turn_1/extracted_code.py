# Implementation reasoning:
# - The RAW tensors ``cos`` and ``sin`` must be streamed onto‑chip with ``offchip_load``.
# - The RoPE rotation can be expressed without any Python slicing or ``torch.cat`` by:
#   1. Splitting the tile‑column dimension into two equal halves using ``retile_streamify``.
#   2. Converting the two halves into separate stream elements with ``flatten`` and ``parallelize``.
#   3. Swapping the halves (and applying the required sign) and then interleaving them back
#      into a single stream with ``static_reassemble`` (using a simple alternating control mask).
#   4. Merging the interleaved stream dimension back into the tile‑column dimension with
#      ``reshape_stream`` followed by ``accum_retile_col``.
# - This pipeline implements ``_rotate_half`` completely with DSL calls.
# - Finally, the RoPE formula ``Q_out = Q * cos + _rotate_half(Q) * sin`` (and similarly for ``K``)
#   is realized with the binary compute DSL ops.

def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Helper: rotate_half implemented with DSL ops only (no torch.cat)
    # ------------------------------------------------------------------
    def _rotate_half(x):
        # x : stream (B, H) × tile (R, C) where C = 2 * half
        half = x.shape[-1] // 2  # Python scalar

        # 1️⃣ Split the column dimension into two halves → extra stream dim of size 2
        x_split = retile_streamify(x, chunk=half, split_row=False)          # (B, H*2, R, half)
        x_split = reshape_stream(x_split, chunk_size=2, rank=0)            # (B, H, 2, R, half)

        # 2️⃣ Collapse the leading stream dims so we can operate on the two halves separately
        x_flat = flatten(x_split, min_rank=0, max_rank=1)                  # (B*H*2, R, half)

        # 3️⃣ Separate the two halves into distinct streams
        halves = parallelize(x_flat, 2)                                    # [half0, half1]
        half0, half1 = halves[0], halves[1]                               # each (B*H, R, half)

        # 4️⃣ Apply the rotation: first half = -second half, second half = first half
        half0_rot = unary_mul_imm(half1, -1.0)                             # -second half
        half1_rot = half0                                                  #  first half

        # 5️⃣ Interleave the rotated halves back together.
        #    Build a control mask that alternates between the two inputs.
        total = half0_rot.shape[0] * 2
        ctrl = []
        for i in range(total):
            if i % 2 == 0:
                ctrl.append([1, 0])
            else:
                ctrl.append([0, 1])
        control = torch.tensor(ctrl, dtype=torch.float32)                 # (total, 2)

        interleaved = static_reassemble([half0_rot, half1_rot], control) # (2*B*H, R, half)

        # 6️⃣ Reshape stream dim back to (B*H, 2, R, half)
        interleaved = reshape_stream(interleaved, chunk_size=2, rank=0)   # (B*H, 2, R, half)

        # 7️⃣ Merge the component stream dim into the tile‑column dimension
        rotated = accum_retile_col(interleaved)                            # (B*H, R, 2*half)

        # 8️⃣ Restore the original two‑level stream shape (B, H, R, C)
        heads = x.shape[1]
        rotated = reshape_stream(rotated, chunk_size=heads, rank=0)        # (B, H, R, C)

        return rotated

    # ------------------------------------------------------------------
    # Load the RAW ``cos`` and ``sin`` tensors onto‑chip.
    # ------------------------------------------------------------------
    tile_row = 1
    tile_col = Q.shape[-1]                     # 32
    stride = [1]

    cos_loaded = flatten(
        offchip_load(cos, stride=stride, out_shape_tiled=[64], tile_row=tile_row, tile_col=tile_col),
        min_rank=0,
        max_rank=1,
    )
    sin_loaded = flatten(
        offchip_load(sin, stride=stride, out_shape_tiled=[64], tile_row=tile_row, tile_col=tile_col),
        min_rank=0,
        max_rank=1,
    )

    # ------------------------------------------------------------------
    # RoPE for Q
    # ------------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_loaded)                # Q * cos
    Q_rot = _rotate_half(Q)                          # rotate_half(Q)
    Q_rot_sin = binary_mul(Q_rot, sin_loaded)        # rotate_half(Q) * sin
    Q_out = binary_add(Q_cos, Q_rot_sin)             # final Q_out

    # ------------------------------------------------------------------
    # RoPE for K
    # ------------------------------------------------------------------
    K_cos = binary_mul(K, cos_loaded)                # K * cos
    K_rot = _rotate_half(K)                          # rotate_half(K)
    K_rot_sin = binary_mul(K_rot, sin_loaded)        # rotate_half(K) * sin
    K_out = binary_add(K_cos, K_rot_sin)             # final K_out

    return Q_out, K_out