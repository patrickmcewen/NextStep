# The O‑projection adds a residual after the attention block.  
# Steps:
# 1. Call the `attention` child to get a tensor of shape (seq_len, num_heads, head_dim)  
#    → stream shape (seq_len) with tile (num_heads, head_dim).
# 2. Convert that tile (num_heads, head_dim) → a single‑row tile (1, num_heads*head_dim)  
#    using two `retile_streamify` passes (first split rows, then cols), then
#    `reshape_stream` to split the huge stream dimension into (seq_len, num_heads*head_dim),
#    and finally `accum_retile_col` to merge the inner stream dim into the tile columns.
# 3. Load the projection weight from off‑chip and broadcast it across the same stream
#    using `offchip_load_ref`, then collapse the extra singleton stream dim with
#    `accum_retile_row`.
# 4. Perform the matrix multiplication (`binary_matmul`) between the flattened attention
#    and the broadcast weight – result has tile (1, 512) and stream (seq_len).
# 5. Load the residual (`input_tensor`) in the same way, broadcast it, and collapse the
#    extra stream dim to obtain tile (1, 512).
# 6. Add the projected tensor and the residual with `binary_add` and return the result.

def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Attention (vanilla shape (seq_len, num_heads, head_dim))
    attn = attention(
        Q, K, V,
        out_shapes=((Q.shape[0], Q.shape[1], Q.shape[2]),),
        out_perms=(None,),
    )  # → (seq_len, num_heads, head_dim) as a stream

    # 2. Flatten the tile (num_heads, head_dim) → (1, num_heads*head_dim)
    attn = retile_streamify(attn, chunk=1, split_row=True)   # (seq_len*num_heads, 1, head_dim)
    attn = retile_streamify(attn, chunk=1, split_row=False)  # (seq_len*num_heads*head_dim, 1, 1)
    attn = reshape_stream(
        attn,
        chunk_size=Q.shape[1] * Q.shape[2],  # num_heads * head_dim = 512
        rank=0,
    )  # → (seq_len, 512, 1, 1)
    attn = accum_retile_col(attn, rank=1)  # → (seq_len, 1, 512)

    # 3. Load and broadcast the projection weight (512 × 512)
    proj_w = offchip_load_ref(
        attn,
        o_proj_weight,
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=512,
        tile_col=512,
        transposed=False,
    )
    proj_w = accum_retile_row(proj_w, rank=1)  # → (seq_len, 512, 512)

    # 4. Matrix multiplication: (seq_len, 1, 512) × (seq_len, 512, 512)
    projected = binary_matmul(attn, proj_w)  # → (seq_len, 1, 512)

    # 5. Load and broadcast the residual tensor (seq_len × 512)
    resid = offchip_load_ref(
        attn,
        input_tensor,
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=1,
        tile_col=512,
        transposed=False,
    )
    resid = accum_retile_row(resid, rank=1)  # → (seq_len, 1, 512)

    # 6. Residual addition
    out = binary_add(projected, resid)  # → (seq_len, 1, 512)

    return out