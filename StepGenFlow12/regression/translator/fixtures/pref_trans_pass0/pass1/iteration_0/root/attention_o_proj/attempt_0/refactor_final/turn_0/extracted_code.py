# Implementation reasoning:
# 1. Compute the attention output using the child `attention`. Its output
#    has tile shape (seq_len, num_heads, head_dim) i.e. (64,16,32).
# 2. Turn the per‑token (num_heads×head_dim) tile into a flat (1,512)
#    tile:
#    a) `retile_streamify(..., chunk=1)` splits the tile rows (num_heads)
#       into a separate stream dimension, yielding shape (seq_len*num_heads, 1, head_dim).
#    b) `reshape_stream(..., chunk_size=num_heads, rank=0)` splits that combined
#       stream back into (seq_len, num_heads) streams, giving shape (seq_len, num_heads, 1, head_dim).
#    c) `accum_retile_col(..., rank=1)` merges the innermost stream dimension
#       (num_heads) into the tile‑column dimension, producing (seq_len, 1, num_heads*head_dim)
#       i.e. (64, 1, 512).
# 3. Load the projection matrix (512×512) from off‑chip and broadcast it over the
#    seq_len stream. `offchip_load` introduces a leading singleton stream dimension,
#    which we remove with `flatten(..., min_rank=0, max_rank=1)`, yielding shape
#    (seq_len, 512, 512).
# 4. Perform the matrix multiplication per token with `binary_matmul`,
#    resulting in (seq_len, 1, 512).
# 5. Load the residual `input_tensor` (shape (seq_len, 512)) as a (1, 512) tile per token.
#    After `offchip_load` we flatten the leading singleton stream dimension to obtain
#    (seq_len, 1, 512) to match the projection output.
# 6. Add the residual via `binary_add` and return the result, which matches the
#    required output shape (64, 1, 512).

def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Attention: returns (seq_len, num_heads, head_dim) tile stream.
    attn = attention(
        Q,
        K,
        V,
        out_shapes=((Q.shape[0], Q.shape[1], Q.shape[2]),),
        out_perms=(None,),
    )

    # 2a. Split tile rows (num_heads) into a stream dimension.
    attn_split = retile_streamify(attn, chunk=1)  # (seq_len*num_heads, 1, head_dim)

    # 2b. Reshape the combined stream back into (seq_len, num_heads) streams.
    num_heads = Q.shape[1]
    attn_reshaped = reshape_stream(attn_split, chunk_size=num_heads, rank=0)  # (seq_len, num_heads, 1, head_dim)

    # 2c. Merge the innermost stream dimension (num_heads) into tile columns.
    attn_flat = accum_retile_col(attn_reshaped, rank=1)  # (seq_len, 1, num_heads*head_dim) → (64,1,512)

    # 3. Load the O‑projection weight and broadcast over the seq_len stream.
    w_stream = offchip_load(
        o_proj_weight,
        stride=(0,),
        out_shape_tiled=(Q.shape[0],),   # same seq_len as attention output
        tile_row=512,
        tile_col=512,
    )
    # Remove the leading singleton stream dimension introduced by offchip_load.
    W = flatten(w_stream, min_rank=0, max_rank=1)  # (seq_len, 512, 512)

    # 4. Matrix multiplication: (seq_len,1,512) @ (seq_len,512,512) → (seq_len,1,512)
    proj = binary_matmul(attn_flat, W)

    # 5. Load the residual input tensor and align its shape.
    inp_stream = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(Q.shape[0],),   # seq_len
        tile_row=1,
        tile_col=512,
    )
    inp = flatten(inp_stream, min_rank=0, max_rank=1)  # (seq_len, 1, 512)

    # 6. Residual addition.
    out = binary_add(proj, inp)

    return out