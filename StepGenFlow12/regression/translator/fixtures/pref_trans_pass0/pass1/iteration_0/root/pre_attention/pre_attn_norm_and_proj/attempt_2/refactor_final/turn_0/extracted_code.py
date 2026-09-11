# This node implements the pre‑attention RMSNorm followed by Q, K, V projections.
# All off‑chip tensors are first loaded with `offchip_load`.  RMSNorm is
# expressed with a sequence of unary/binary ops (square → row‑wise sum →
# mean → epsilon add → rsqrt → scaling).  The three projection matrices are
# broadcast across the sequence dimension.  After the matmul we have a tile
# shape (1, hidden_dim).  The hidden dimension is split into the per‑head
# dimension (head_dim) and the number of heads using `retile_streamify`,
# which creates an extra stream dimension for the heads.  That stream dimension
# is then merged into the tile‑row dimension with `accum_retile_row`,
# yielding a final layout (stream=seq_len, tile_rows=heads, tile_cols=head_dim),
# i.e. the required shape (seq_len, heads, head_dim).  The same steps are
# applied to the K and V projections (with a different number of heads).

def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos,
                          *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Shape constants (derived from raw inputs – scalar Python math allowed)
    # ------------------------------------------------------------------
    seq_len = input_tensor.shape[0]                # 64
    hidden_dim = input_tensor.shape[1]             # 512
    head_dim = cos.shape[-1]                       # 32
    num_heads = q_proj.shape[1] // head_dim        # 16
    num_kv_heads = k_proj.shape[1] // head_dim     # 4
    eps = 1e-6

    # ------------------------------------------------------------------
    # Load raw tensors into the on‑chip stream representation.
    # `offchip_load` adds a leading singleton stream dim.
    # ------------------------------------------------------------------
    # Input tensor: stream over the sequence, tile = (1, hidden_dim)
    inp = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=hidden_dim,
    )  # shape (1, seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # RMSNorm: x * rsqrt(mean(x^2) + eps)
    # ------------------------------------------------------------------
    inp_sq = unary_square(inp)                                 # (1, seq_len, 1, hidden_dim)
    sum_sq = unary_rowwise_sum(inp_sq)                         # (1, seq_len, 1, 1)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)          # (1, seq_len, 1, 1)
    mean_eps = unary_add_imm(mean_sq, eps)                     # (1, seq_len, 1, 1)
    scale = unary_rsqrt(mean_eps)                               # (1, seq_len, 1, 1)
    normed = binary_mul(inp, scale)                            # (1, seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------
    # Helper to perform a projection + reshape to (seq_len, heads, head_dim)
    # ------------------------------------------------------------------
    def proj_and_reshape(weight, out_dim, heads):
        # Broadcast weight across the sequence dimension
        w = offchip_load(
            weight,
            stride=(0,),
            out_shape_tiled=(seq_len,),
            tile_row=hidden_dim,
            tile_col=out_dim,
        )  # shape (1, seq_len, hidden_dim, out_dim)

        # MatMul: (seq_len, hidden_dim) @ (hidden_dim, out_dim) -> (seq_len, out_dim)
        raw = binary_matmul(normed, w)                         # (1, seq_len, 1, out_dim)

        # Insert a dummy stream dimension after the sequence dim
        raw_promoted = promote(raw, rank=0)                    # (1, seq_len, 1, 1, out_dim)

        # Split the tile‑column (out_dim = heads * head_dim) into a new stream dim
        raw_split = retile_streamify(
            raw_promoted,
            chunk=head_dim,
            split_row=False,
        )  # (1, seq_len, heads, 1, head_dim)

        # Merge the new stream dim (heads) into the tile‑row dimension
        final = accum_retile_row(raw_split, rank=1)            # (1, seq_len, heads, head_dim)

        return final

    # ------------------------------------------------------------------
    # Q, K, V projections
    # ------------------------------------------------------------------
    Q = proj_and_reshape(q_proj, q_proj.shape[1], num_heads)          # (1, seq_len, 16, 32)
    K = proj_and_reshape(k_proj, k_proj.shape[1], num_kv_heads)       # (1, seq_len, 4, 32)
    V = proj_and_reshape(v_proj, v_proj.shape[1], num_kv_heads)       # (1, seq_len, 4, 32)

    # The dummy leading dimension is kept (offchip_store will strip it later).
    return Q, K, V