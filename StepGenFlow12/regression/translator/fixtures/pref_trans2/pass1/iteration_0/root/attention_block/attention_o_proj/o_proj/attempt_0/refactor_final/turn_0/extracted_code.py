# Implementation reasoning:
# 1. Load the projection weight from off‑chip memory, broadcasting it over the
#    sequence dimension.  The weight has shape (512, 512).  We tile it with
#    tile_row=32 and tile_col=512, yielding tiles of shape (32, 512).  The
#    underlying grid has 16 row‑tiles, so we stream over (seq_len, num_heads) =
#    (64, 16) with stride (0, 1) to broadcast each head’s tile across all 64
#    sequence positions:
#        weight = offchip_load(o_proj_weight,
#                               stride=(0, 1),
#                               out_shape_tiled=(64, 16),
#                               tile_row=32,
#                               tile_col=512)
#
# 2. The input `attn` has tile shape (16, 32) where the 16 rows encode the
#    `num_heads`.  We need the heads to become a separate stream dimension so
#    that they line up with the streamed weight.  First move the tile‑row
#    dimension into a stream dimension with `retile_streamify(split_row=True,
#    chunk=1)`, turning (1, 64, 16, 32) → (1, 1024, 1, 32).  Then split that
#    combined stream dimension back into (seq_len, num_heads) using
#    `reshape_stream(chunk_size=16, rank=0)`, yielding a stream of shape
#    (1, 64, 16, 1, 32).  Now the stream shape matches that of `weight`.
#
# 3. Perform a batched matrix multiplication: each tile (1 × 32) from `attn`
#    multiplies the corresponding weight tile (32 × 512), producing tiles of
#    shape (1 × 512).  This is done with `binary_matmul`.
#
# 4. The result still has a separate head stream dimension (size 16).  The
#    original flatten operation corresponds to summing over this dimension,
#    which we achieve with `accum_add` (rank = 1) that reduces the innermost
#    stream axis.
#
# 5. The final tensor has shape (1, 64, 1, 512) as required.

def o_proj(attn, o_proj_weight, *, out_shapes, out_perms=None):
    # Step 1: load weight and broadcast over sequence length (64)
    weight = offchip_load(
        o_proj_weight,
        stride=(0, 1),
        out_shape_tiled=(64, 16),
        tile_row=32,
        tile_col=512,
    )
    # Step 2: move the head dimension (tile rows = 16) to a stream axis
    attn_stream = retile_streamify(attn, chunk=1, split_row=True)
    # Split the combined stream (1024) into (seq_len=64, heads=16)
    attn_stream = reshape_stream(attn_stream, chunk_size=16, rank=0)
    # Step 3: batched matrix multiplication (1×32) @ (32×512) → (1×512)
    proj = binary_matmul(attn_stream, weight)
    # Step 4: sum over the head stream dimension to obtain (1, 64, 1, 512)
    out = accum_add(proj, rank=1)
    return out