# The O‑projection node:
#   1. Calls the `attention` child to obtain the attention tensor (stream (64,) tile (16, 32)).
#   2. Splits the 16‑row tile into a new stream dimension (retile_streamify) and then reshapes
#      that stream dimension back into (seq_len = 64, num_heads = 16) (reshape_stream),
#      yielding stream (64, 16) tile (1, 32).
#   3. Loads the O‑projection weight matrix (512 × 512) as 16 tiles of shape (32, 512) and
#      broadcasts them over the (64, 16) attention stream. The leading singleton added by
#      `offchip_load` is removed with `flatten`, giving a matching stream (64, 16) tile (32, 512).
#   4. Performs a per‑head matrix multiply (binary_matmul) => stream (64, 16) tile (1, 512).
#   5. Sums across the head dimension (accum_add, rank=1) → stream (64,) tile (1, 512).
#   6. Loads the residual `input_tensor` (shape (64, 512)) as tiles (1, 512) with a leading
#      singleton, then flattens away that singleton so its stream matches the projected output.
#   7. Adds the residual to the projected result (binary_add) and returns the final
#      stream‑tensor of shape (64, 1, 512), which satisfies the contract.
def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1️⃣ Compute attention (vanilla shape (64, 16, 32) → stream (64,) tile (16, 32))
    attn = attention(
        Q, K, V,
        out_shapes=((64, 16, 32),),   # stream (64,) tile (16,32)
        out_perms=(None,)
    )

    # 2️⃣ Split the tile rows into a stream dimension (chunk=1 ⇒ tile rows become 1)
    attn_split = retile_streamify(attn, chunk=1, split_row=True)

    # 3️⃣ Reshape the enlarged stream dimension (1024) back into (64, 16)
    attn_reshaped = reshape_stream(
        attn_split,
        chunk_size=16,   # each head occupies 16 entries
        rank=0           # innermost stream dim
    )                     # → stream (64,16) tile (1,32)

    # 4️⃣ Load O‑projection weights and broadcast across (seq_len, heads)
    #    - Tile rows = 32 (head_dim), tile cols = 512 (hidden_dim)
    #    - out_shape_tiled matches the attention stream (64, 16)
    #    - stride (0,1): broadcast across sequence, step across heads
    weight_raw = offchip_load(
        o_proj_weight,
        (0, 1),                # stride
        (64, 16),              # out_shape_tiled
        32,                    # tile_row
        512                    # tile_col
    )                           # → stream (1,64,16) tile (32,512)

    # 5️⃣ Remove the leading singleton so weight stream = (64,16)
    weight = flatten(weight_raw, 1, 2)   # merge dims 1 and 2 → stream (64,16)

    # 6️⃣ Per‑head matrix multiplication (1×32) · (32×512) → (1×512)
    proj = binary_matmul(attn_reshaped, weight)   # stream (64,16) tile (1,512)

    # 7️⃣ Sum over the head dimension → stream (64,) tile (1,512)
    proj_summed = accum_add(proj)   # default rank=1

    # 8️⃣ Load the residual tensor (64,512) as tiles (1,512) with a leading singleton
    residual_raw = offchip_load(
        input_tensor,
        (1,),                 # stride: advance per token
        (64,),                # out_shape_tiled: one tile per token
        1,                    # tile_row
        512                   # tile_col
    )                          # → stream (1,64) tile (1,512)

    # 9️⃣ Flatten away the leading singleton so residual stream = (64,)
    residual = flatten(residual_raw, 0, 1)   # merge dims 0 and 1 → stream (64,)

    # 10️⃣ Add the residual to the projected output
    out = binary_add(proj_summed, residual)   # stream (64,) tile (1,512)

    return out