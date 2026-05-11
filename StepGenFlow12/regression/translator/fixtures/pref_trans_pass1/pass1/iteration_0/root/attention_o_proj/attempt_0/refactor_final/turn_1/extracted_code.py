# Implementation reasoning:
# 1. Call the `attention` child to obtain the attention output as a stream with
#    tile shape (num_heads, head_dim) = (16, 32).
# 2. Convert each tile (16,32) into a single‑row tile (1, 16*32=512) by:
#    - Splitting the tile rows into a new stream dimension (retile_streamify,
#      split_row=True)
#    - Splitting the tile columns similarly (split_row=False)
#    - Reshaping the long stream into (seq_len, 512) using `reshape_stream`
#    - Absorbing the inner stream dimension into the tile columns with
#      `accum_retile_col`, yielding shape (seq_len, 1, 512).
# 3. Load the projection weight (512 × 512) from off‑chip memory, broadcasting
#    it across the sequence dimension:
#    - `offchip_load` with `stride=(0,)` (same tile for every position) and
#      `out_shape_tiled=(seq_len,)` creates a stream shape (1, seq_len) tile
#      (512,512).
#    - `flatten` merges the leading singleton stream dimension with the seq_len
#      dimension, giving shape (seq_len, 512, 512).
# 4. Perform the matrix multiplication between the flattened attention and the
#    broadcast weight using `binary_matmul`, resulting in (seq_len, 1, 512).
# 5. Load the residual tensor (seq_len × 512) similarly, broadcasting each row
#    across the stream:
#    - `offchip_load` with `stride=(1,)` maps each stream element to the
#      corresponding row tile.
#    - `flatten` removes the leading singleton stream dim, yielding
#      shape (seq_len, 1, 512).
# 6. Add the residual to the projected attention with `binary_add` and return
#    the final stream tensor.

def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Attention block (vanilla shape (seq_len, num_heads, head_dim))
    attn = attention(
        Q, K, V,
        out_shapes=((Q.shape[0], Q.shape[1], Q.shape[2]),),
        out_perms=(None,),
    )  # stream: (seq_len, num_heads, head_dim) tile (16,32)

    # 2. Flatten tile (num_heads, head_dim) → (1, num_heads*head_dim)
    attn = retile_streamify(attn, chunk=1, split_row=True)   # → stream( seq_len*num_heads ) tile(1,32)
    attn = retile_streamify(attn, chunk=1, split_row=False)  # → stream( seq_len*num_heads*head_dim ) tile(1,1)
    attn = reshape_stream(
        attn,
        chunk_size=Q.shape[1] * Q.shape[2],   # 16*32 = 512
        rank=0,
    )  # → stream( seq_len, 512 ) tile(1,1)
    attn = accum_retile_col(attn, rank=1)  # → stream( seq_len ) tile(1,512)

    # 3. Load and broadcast the projection weight (512×512) across the sequence
    proj_w_loaded = offchip_load(
        o_proj_weight,
        stride=(0,),
        out_shape_tiled=(Q.shape[0],),   # seq_len = 64
        tile_row=512,
        tile_col=512,
        transposed=False,
    )
    proj_w = flatten(proj_w_loaded, 0, 1)  # merge leading 1 with seq_len → stream(seq_len) tile(512,512)

    # 4. Matrix multiplication: (seq_len, 1, 512) × (seq_len, 512, 512)
    projected = binary_matmul(attn, proj_w)  # → stream(seq_len) tile(1,512)

    # 5. Load and broadcast the residual (seq_len × 512)
    resid_loaded = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(Q.shape[0],),   # seq_len = 64
        tile_row=1,
        tile_col=512,
        transposed=False,
    )
    resid = flatten(resid_loaded, 0, 1)   # → stream(seq_len) tile(1,512)

    # 6. Add residual
    out = binary_add(projected, resid)    # → stream(seq_len) tile(1,512)

    return out