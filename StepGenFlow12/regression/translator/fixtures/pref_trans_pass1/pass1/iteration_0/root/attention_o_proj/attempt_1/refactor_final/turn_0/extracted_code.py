# The node first calls the `attention` child to obtain the raw attention
# tensor (shape (64,16,32) → stream (1,64) tile (16,32)).
# To perform the O‑projection we need to treat the tile (16,32) as a
# concatenated vector of length 512.  This is done by:
#   1. Splitting the tile rows into a new stream dimension with
#      `retile_streamify(chunk=1, split_row=True)` → tile becomes (1,32)
#      and the stream dimension size grows from 64 to 64*16 = 1024.
#   2. Reshaping that stream dimension into two dimensions (64,16)
#      via `reshape_stream(chunk_size=16, rank=0)`.
# Now the attention tensor has stream (1,64,16) and tile (1,32).
#
# The projection weight (512×512) is tiled as 16 blocks of (32,512).
# Using `offchip_load` with `out_shape_tiled=(64,16)` and
# `stride=(0,1)` broadcasts each block across the sequence dimension,
# yielding a tensor with the same stream (1,64,16) and tile (32,512).
#
# A matrix multiplication (`binary_matmul`) between (1,32) and (32,512)
# produces tile (1,512).  Summing over the head stream dimension (the
# innermost stream dim) with `accum_add(rank=1)` collapses the heads,
# giving stream (1,64) and tile (1,512).
#
# Finally, the residual `input_tensor` (shape (64,512)) is loaded as a
# stream of tiles (1,512) with `offchip_load(stride=(1,), out_shape_tiled=(64,),
# tile_row=1, tile_col=512)`.  Adding this to the projected result via
# `binary_add` yields the required output tensor of shape (64,1,512)
# (the leading batch dimension of size 1 is implicit in the DSL).
def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1️⃣ Compute attention (vanilla shape (64, 16, 32))
    attn = attention(Q, K, V,
                     out_shapes=((64, 16, 32),),
                     out_perms=(None,))

    # 2️⃣ Split the tile rows into a stream dimension (chunk=1 ⇒ tile rows become 1)
    attn_split = retile_streamify(attn, chunk=1, split_row=True)

    # 3️⃣ Reshape the enlarged stream dimension (1024) back into (64, 16)
    attn_reshaped = reshape_stream(attn_split,
                                   chunk_size=16,   # each head occupies 16 entries
                                   rank=0)          # innermost stream dim

    # 4️⃣ Load the O‑projection weight and broadcast it over (seq_len, heads)
    #    - Tile rows = 32 (head_dim), tile cols = 512 (hidden_dim)
    #    - out_shape_tiled = (64, 16) to match the attn stream shape
    #    - stride = (0, 1): broadcast across sequence, advance across heads
    weight = offchip_load(o_proj_weight,
                          (0, 1),                 # stride
                          (64, 16),               # out_shape_tiled
                          32,                     # tile_row
                          512)                    # tile_col

    # 5️⃣ Matrix multiply each (1 × 32) query slice with its corresponding (32 × 512) weight slice
    proj = binary_matmul(attn_reshaped, weight)

    # 6️⃣ Accumulate over the head dimension (the innermost stream dim) to obtain (64, 1, 512)
    proj_summed = accum_add(proj, rank=1)

    # 7️⃣ Load the residual input tensor (shape (64, 512)) as tiles (1 × 512)
    residual = offchip_load(input_tensor,
                            (1,),                  # stride: advance per token
                            (64,),                 # out_shape_tiled: one tile per token
                            1,                     # tile_row
                            512)                   # tile_col

    # 8️⃣ Add the residual to the projected output
    out = binary_add(proj_summed, residual)

    return out