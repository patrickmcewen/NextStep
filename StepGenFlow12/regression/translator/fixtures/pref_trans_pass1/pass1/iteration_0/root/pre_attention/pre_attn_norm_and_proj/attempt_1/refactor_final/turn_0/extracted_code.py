# This node performs the pre‑attention RMSNorm and the Q, K, V linear projections.
#   1. Load the raw tensors as single‑tile streams.
#   2. Compute RMSNorm:  x * rsqrt(mean(x²) + eps)
#        - square, sum rows, scale by 1/hidden, add epsilon, rsqrt, then broadcast‑multiply.
#   3. Multiply the normalized activations with each projection weight (binary_matmul).
#   4. Reshape each result from (seq_len, heads*head_dim) tile to a stream of shape
#        (seq_len) with tile shape (heads, head_dim):
#        a) retile rows → move seq_len to a stream dimension.
#        b) retile columns → move num_heads to the same (innermost) stream dimension.
#        c) reshape_stream → split that combined stream dimension into (seq_len, num_heads).
#        d) accum_retile_row → absorb the num_heads stream dimension into the tile rows.
#        e) flatten → collapse the leading singleton stream dimension with seq_len,
#                     yielding the final stream shape (seq_len, heads, head_dim).
#   The three tensors Q, K, V are returned as StepTensors matching the required
#   output shapes [(64,16,32), (64,4,32), (64,4,32)].
def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------------
    # 1. Load raw inputs as single‑tile streams.
    # -----------------------------------------------------------------------
    seq_len = input_tensor.shape[0]          # 64
    hidden_dim = input_tensor.shape[1]       # 512
    head_dim = cos.shape[-1]                # 32 (from the supplied cosine tensor)

    # Load the input activation matrix.
    inp = offchip_load(
        input_tensor,
        stride=(0,),                # only one tile, stride irrelevant
        out_shape_tiled=(1,),       # single stream element
        tile_row=seq_len,
        tile_col=hidden_dim,
    )

    # Load projection weight matrices.
    q_w = offchip_load(
        q_proj,
        stride=(0,),
        out_shape_tiled=(1,),
        tile_row=hidden_dim,
        tile_col=q_proj.shape[1],
    )
    k_w = offchip_load(
        k_proj,
        stride=(0,),
        out_shape_tiled=(1,),
        tile_row=hidden_dim,
        tile_col=k_proj.shape[1],
    )
    v_w = offchip_load(
        v_proj,
        stride=(0,),
        out_shape_tiled=(1,),
        tile_row=hidden_dim,
        tile_col=v_proj.shape[1],
    )

    # -----------------------------------------------------------------------
    # 2. RMSNorm on the input.
    #    rms = x * rsqrt(mean(x^2) + eps)
    # -----------------------------------------------------------------------
    eps = 1e-6
    inv_hidden = 1.0 / hidden_dim

    # x^2
    sq = unary_square(inp)
    # sum over hidden dimension (row‑wise sum -> shape (seq_len, 1))
    sum_sq = unary_rowwise_sum(sq)
    # mean = sum / hidden_dim
    mean_sq = unary_mul_imm(sum_sq, constant=inv_hidden)
    # add epsilon
    mean_eps = unary_add_imm(mean_sq, constant=eps)
    # rsqrt -> scaling factor per token (shape (seq_len, 1))
    scale = unary_rsqrt(mean_eps)
    # broadcast multiply: normalized activation
    normed = binary_mul(inp, scale)

    # -----------------------------------------------------------------------
    # 3. Linear projections (matrix multiply).
    # -----------------------------------------------------------------------
    Q_raw = binary_matmul(normed, q_w)   # (1,1,seq_len, num_heads*head_dim)
    K_raw = binary_matmul(normed, k_w)   # (1,1,seq_len, num_kv_heads*head_dim)
    V_raw = binary_matmul(normed, v_w)   # (1,1,seq_len, num_kv_heads*head_dim)

    # -----------------------------------------------------------------------
    # Helper to reshape a projection from (seq_len, heads*head_dim) tile to
    # stream shape (seq_len) with tile (heads, head_dim).
    # -----------------------------------------------------------------------
    def reshape_proj(raw_proj, num_heads):
        # a) Move seq_len from tile rows into a stream dimension.
        step = retile_streamify(raw_proj, chunk=1, split_row=True)

        # b) Split the output column dimension (heads*head_dim) into heads.
        step = retile_streamify(step, chunk=head_dim, split_row=False)

        # c) Split the combined stream dimension (seq_len * num_heads) into
        #    (seq_len, num_heads).
        step = reshape_stream(step, chunk_size=num_heads, rank=0)

        # d) Absorb the innermost stream dim (num_heads) into tile rows.
        step = accum_retile_row(step, rank=1)

        # e) Collapse the leading singleton stream dim with seq_len.
        step = flatten(step, min_rank=0, max_rank=1)
        return step

    # Number of heads for Q and K/V.
    num_heads = q_proj.shape[1] // head_dim          # 16
    num_kv_heads = k_proj.shape[1] // head_dim       # 4

    # -----------------------------------------------------------------------
    # 4. Apply the reshaping pipeline to each projection.
    # -----------------------------------------------------------------------
    Q = reshape_proj(Q_raw, num_heads)
    K = reshape_proj(K_raw, num_kv_heads)
    V = reshape_proj(V_raw, num_kv_heads)

    # Return the three projected tensors.
    return Q, K, V