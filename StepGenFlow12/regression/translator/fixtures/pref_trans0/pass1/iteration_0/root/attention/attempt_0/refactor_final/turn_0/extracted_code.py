# The attention node computes:
#   1. Vanilla attention via the child `attention_compute`.
#   2. Flattens the (num_heads, head_dim) tile into a single row of length
#      `num_heads*head_dim` using `retile_streamify` + `reshape_stream`.
#   3. Loads the projection matrix as a tiled stream sized
#      (head_dim, hidden_dim) and broadcasts it across the sequence dimension.
#   4. Performs a per‑head matmul (`binary_matmul`) and sums over heads
#      (`accum_add`) to obtain the projected tensor.
#   5. Loads the residual `input_tensor` as a stream with tile (1, hidden_dim).
#   6. Adds the residual (`binary_add`) and collapses the leading singleton
#      stream dimension with `flatten` to produce shape (seq_len, 1, hidden_dim).
# All tensor arithmetic is expressed through DSL ops; off‑chip tensors are
# loaded before any compute consumer.

def attention(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
    # 1. Attention output: (seq_len, num_heads, head_dim)
    attn = attention_compute(
        Q, K, V,
        out_shapes=((Q.shape[0], Q.shape[1], Q.shape[2]),),
        out_perms=(None,),
    )

    # 2. Split the head dimension into a separate stream element.
    #    After retile_streamify each head becomes a distinct stream entry.
    attn_rows = retile_streamify(attn, chunk=1, split_row=True)               # (seq_len*num_heads, 1, head_dim)
    attn_rows = reshape_stream(attn_rows, chunk_size=Q.shape[1], rank=0)      # (seq_len, num_heads, 1, head_dim)
    attn_rows = promote_outer(attn_rows)                                     # (1, seq_len, num_heads, 1, head_dim)

    # 3. Load the projection matrix as a tiled stream.
    #    Tile size matches (head_dim, hidden_dim); stride (0,1) broadcasts across seq_len.
    weight = offchip_load(
        o_proj_weight,
        stride=(0, 1),
        out_shape_tiled=(Q.shape[0], Q.shape[1]),
        tile_row=Q.shape[2],               # head_dim
        tile_col=o_proj_weight.shape[1],   # hidden_dim
    )  # (1, seq_len, num_heads, head_dim, hidden_dim)

    # 4. Per‑head projection and sum over heads.
    proj = binary_matmul(attn_rows, weight)   # (1, seq_len, num_heads, 1, hidden_dim)
    proj = accum_add(proj, rank=1)            # (1, seq_len, 1, hidden_dim)

    # 5. Load the residual tensor (seq_len, hidden_dim) as (1, seq_len, 1, hidden_dim).
    residual = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(Q.shape[0],),
        tile_row=1,
        tile_col=o_proj_weight.shape[1],   # hidden_dim
    )  # (1, seq_len, 1, hidden_dim)

    # 6. Residual addition.
    summed = binary_add(proj, residual)       # (1, seq_len, 1, hidden_dim)

    # 7. Collapse the leading singleton stream dimension ⇒ (seq_len, 1, hidden_dim)
    out = flatten(summed, min_rank=0, max_rank=1)   # (seq_len, 1, hidden_dim)

    return out