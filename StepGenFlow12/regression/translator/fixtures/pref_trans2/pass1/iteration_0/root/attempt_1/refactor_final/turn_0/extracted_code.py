# Implementation reasoning:
# 1. Load the raw input tensor as a stream of tiles.  We tile each token as a
#    1×feature tile, i.e. tile_row=1, tile_col=feature_dim (512).  The stream
#    dimension is the sequence length (64) and we keep a leading singleton
#    dimension added by `offchip_load`.
# 2. Perform RMSNorm using DSL compute ops only:
#       - square the input (binary_mul)
#       - sum across the feature dimension (unary_rowwise_sum)
#       - compute the mean by multiplying with 1/feature_dim (unary_mul_imm)
#       - add epsilon (unary_add_imm)
#       - take reciprocal sqrt (unary_rsqrt)
#       - multiply the original input by the scaling factor (binary_mul)
# 3. Call the attention blackbox with the normalized stream and raw weight /
#    positional arguments.  We request the output stream shape to match the
#    input stream shape ((1, seq_len, 1, feature_dim)).
# 4. Add the residual connection (binary_add) between the attention output and
#    the original input stream.
# 5. Call the MoE blackbox on the residual, again asking for the same stream
#    shape.
# 6. Add the final residual (binary_add) and write the result off‑chip with
#    `offchip_store`, which automatically reshapes the stream back to the
#    vanilla (seq_len, feature_dim) layout.
def tiled_reference(dims, tensors):
    # ---------------------------
    # 1) Load the model input as a tiled stream.
    #    Each token is a tile of shape (1, feature_dim).
    # ---------------------------
    seq_len = tensors["input_tensor"].shape[0]          # 64
    feature_dim = tensors["input_tensor"].shape[1]     # 512
    tile_row = 1
    tile_col = feature_dim
    # stride = (grid_c,) = (1,) because we tile only across rows.
    input_stream = offchip_load(
        tensors["input_tensor"],
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=tile_row,
        tile_col=tile_col,
    )  # shape (1, seq_len, tile_row, tile_col) -> (1,64,1,512)

    # ---------------------------
    # 2) RMSNorm on the input stream.
    #    normed = x * rsqrt(mean(x**2) + eps)
    # ---------------------------
    # x**2
    sq = binary_mul(input_stream, input_stream)
    # sum across feature dimension (tile_col)
    sum_sq = unary_rowwise_sum(sq)                     # (1, seq_len, tile_row, 1)
    # mean = sum / feature_dim
    mean_sq = unary_mul_imm(sum_sq, 1.0 / feature_dim)
    # add epsilon
    eps = 1e-6
    mean_eps = unary_add_imm(mean_sq, eps)
    # rsqrt
    inv_sqrt = unary_rsqrt(mean_eps)
    # normalized output
    normed = binary_mul(input_stream, inv_sqrt)

    # ---------------------------
    # 3) Attention block.
    # ---------------------------
    att_out = attention_block(
        normed,
        tensors["q_proj"],
        tensors["k_proj"],
        tensors["v_proj"],
        tensors["cos"],
        tensors["sin"],
        tensors["o_proj_weight"],
        out_shapes=((1, seq_len, tile_row, tile_col),),  # (1,64,1,512)
    )

    # ---------------------------
    # 4) First residual addition: o_proj_out + input_tensor
    # ---------------------------
    res_add_0 = binary_add(att_out, input_stream)

    # ---------------------------
    # 5) MoE block.
    # ---------------------------
    moe_out = moe_block(
        res_add_0,
        tensors["w_gate"],
        tensors["w_up"],
        tensors["w_down"],
        tensors["expert_weights"],
        tensors["expert_onehot"],
        out_shapes=((1, seq_len, tile_row, tile_col),),  # (1,64,1,512)
    )

    # ---------------------------
    # 6) Final residual addition and write back off-chip.
    # ---------------------------
    final = binary_add(moe_out, res_add_0)

    # Store the result off‑chip (produces a vanilla (seq_len, feature_dim) tensor)
    return offchip_store(final)