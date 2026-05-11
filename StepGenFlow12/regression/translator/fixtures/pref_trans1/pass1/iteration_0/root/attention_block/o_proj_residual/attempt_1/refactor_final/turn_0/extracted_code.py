# Implementation reasoning:
# * `attn` is already on‑chip with shape (64, 16, 32) → stream dim=64, tile (16, 32).
#   We need to flatten the tile into (1, 32) so it can be multiplied with the weight.
#   We split the tile rows into a new stream dimension using `retile_streamify`
#   (chunk=1 makes each row a separate stream element) and then reshape that
#   stream back into (64, 16) using `reshape_stream`.
# * `o_proj_weight` is raw (off‑chip)  (512, 512).  We tile it as (32, 512) and stream
#   it over (seq_len=64, num_heads=16).  Stride [0, 1] replicates the same head slice
#   across all sequence positions.  `offchip_load` yields a leading singleton stream
#   dimension; we merge it with the sequence dimension via `flatten`.
# * Perform per‑head matmul with `binary_matmul`; result shape (64, 16, 1, 512).
# * Sum over the head stream dimension with `accum_add(rank=1)` → (64, 1, 512).
# * Load the residual `input_tensor` (64, 512) as a tiled stream (1, 512) over the
#   sequence dimension, then flatten the leading singleton → (64, 1, 512).
# * Finally, add the residual via `binary_add` and return the stream.
def o_proj_residual(attn, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Convert attn (64,16,32) → (64,16,1,32)
    attn_split = retile_streamify(attn, chunk=1, split_row=True)          # (1024, 1, 32)
    attn_reshaped = reshape_stream(attn_split, chunk_size=16, rank=0)    # (64, 16, 1, 32)

    # 2. Load and tile the projection weight (512,512) → (64,16,32,512)
    #    stride [0,1]: repeat same head slice across sequence positions.
    weight_loaded = offchip_load(
        o_proj_weight,
        stride=[0, 1],
        out_shape_tiled=(64, 16),
        tile_row=32,
        tile_col=512,
    )                                                                    # (1, 64, 16, 32, 512)
    weight_tiled = flatten(weight_loaded, min_rank=1, max_rank=2)       # (64, 16, 32, 512)

    # 3. Per‑head matrix multiplication
    proj = binary_matmul(attn_reshaped, weight_tiled)                    # (64, 16, 1, 512)

    # 4. Sum over heads → (64, 1, 512)
    proj_sum = accum_add(proj, rank=1)

    # 5. Load residual input_tensor (64,512) → (64, 1, 512)
    input_loaded = offchip_load(
        input_tensor,
        stride=[1],
        out_shape_tiled=(64,),
        tile_row=1,
        tile_col=512,
    )                                                                    # (1, 64, 1, 512)
    input_reshaped = flatten(input_loaded, min_rank=0, max_rank=1)       # (64, 1, 512)

    # 6. Add residual
    output = binary_add(proj_sum, input_reshaped)                        # (64, 1, 512)
    return output