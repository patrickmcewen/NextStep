# Attention node implementation.
# Steps:
#   1. Call the child `attention_compute` to obtain the vanilla attention tensor
#      of shape (64, 16, 32).  This tensor is already a stream
#      (stream dim = 64, tile dims = 16×32).
#   2. Convert the head dimension (tile rows = 16) into a stream dimension:
#      - `retile_streamify(..., chunk=1, split_row=True)` splits each of the 16
#        rows into separate stream elements, yielding shape (1024, 1, 32)
#        (stream dim = 64 × 16, tile = 1×32).
#      - `reshape_stream(..., chunk_size=16, rank=0)` merges the combined
#        stream dimension back into two separate dimensions: (seq_len=64,
#        heads=16, tile = 1×32).  Now the tile row is 1 and the tile column is
#        the head dimension (32).
#   3. Load the O‑projection weight (512 × 512) from off‑chip:
#      - Tile rows = head_dim = 32, tile cols = hidden_dim = 512.
#      - Stream over (seq_len, heads) with strides (0, 1) so that the same
#        head‑specific weight tile is reused for every position in the sequence.
#      - `offchip_load(..., out_shape_tiled=(64, 16), stride=(0, 1), …)` emits a
#        tensor of shape (1, 64, 16, 32, 512).  The leading singleton is merged
#        with the sequence dimension using `flatten` (min_rank=1, max_rank=2),
#        yielding a stream shape (64, 16) and tile dims (32, 512).
#   4. Perform per‑head matrix multiplication:
#      `binary_matmul` between the reshaped attention (64,16,1,32) and the
#      weight (64,16,32,512) → (64,16,1,512).
#   5. Accumulate across the head dimension (the innermost stream dim) with
#      `accum_add(rank=1)` → (64,1,512).
#   6. Load the residual input tensor (64, 512):
#      - Tile row = 1, tile col = 512, stream over the sequence dimension.
#      - After `offchip_load` we get (1,64,1,512); `flatten` (min_rank=0,
#        max_rank=1) merges the leading singleton with the sequence stream,
#        giving (64,1,512).
#   7. Add the residual: `binary_add` → final output (64,1,512), matching the
#      contract.
#   8. Return the result; the parent will handle the off‑chip store.

def attention(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Compute vanilla attention (already tiled as (64,16,32)).
    attn = attention_compute(
        Q,
        K,
        V,
        out_shapes=((64, 16, 32),),
        out_perms=(None,),
    )  # shape: (64, 16, 32)

    # 2. Split head dimension into a stream dimension.
    attn_tiled = retile_streamify(attn, chunk=1, split_row=True)   # (1024, 1, 32)
    attn_reshaped = reshape_stream(attn_tiled, chunk_size=16, rank=0)  # (64, 16, 1, 32)

    # 3. Load O‑projection weight and align its stream shape to (seq_len, heads).
    #    Underlying weight shape: (512, 512)
    #    Tile: (32, 512)  ->  grid_r = 16, grid_c = 1
    #    Stream over (64, 16) with stride (0, 1) to broadcast across seq_len.
    weight_raw = offchip_load(
        o_proj_weight,
        stride=(0, 1),
        out_shape_tiled=(64, 16),
        tile_row=32,
        tile_col=512,
    )  # shape: (1, 64, 16, 32, 512)
    # Merge the leading singleton with the sequence dimension.
    weight = flatten(weight_raw, min_rank=1, max_rank=2)  # (64, 16, 32, 512)

    # 4. Per‑head matrix multiplication.
    proj = binary_matmul(attn_reshaped, weight)  # (64, 16, 1, 512)

    # 5. Sum over heads (innermost stream dimension).
    proj_sum = accum_add(proj, rank=1)  # (64, 1, 512)

    # 6. Load the residual input tensor.
    resid_raw = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(64,),
        tile_row=1,
        tile_col=512,
    )  # shape: (1, 64, 1, 512)
    resid = flatten(resid_raw, min_rank=0, max_rank=1)  # (64, 1, 512)

    # 7. Residual addition.
    out = binary_add(proj_sum, resid)  # (64, 1, 512)

    # 8. Return the final streamed tensor.
    return out