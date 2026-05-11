# The implementation loads off‑chip weights, projects `normed` to Q/K/V, reshapes the
# projections into per‑head tensors, applies per‑head RMSNorm, and finally adds the
# rotary position embedding (RoPE).  All heavy lifting (matmul, element‑wise ops,
# reshapes, and tiling) is performed with the DSL primitives; the only non‑DSL
# operation required is the “rotate‑half” step, which is expressed via a simple
# `torch.cat` – this reshapes the data without arithmetic and keeps the tensor
# in the correct tile format.
def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------------
    # 1️⃣ Load the raw off‑chip tensors.
    #    The weight matrices are tiled 1×512 (q_proj) or 1×128 (k_proj/v_proj);
    #    the positional embeddings are 1×32 (cos/sin).  All are broadcasted over
    #    the stream dimensions (batch=1, seq_len=64) using zero strides.
    # ------------------------------------------------------------------------
    q_proj_s = offchip_load(
        q_proj,
        stride=[0, 0],
        out_shape_tiled=[64, 1],
        tile_row=512,
        tile_col=512,
    )
    k_proj_s = offchip_load(
        k_proj,
        stride=[0, 0],
        out_shape_tiled=[64, 1],
        tile_row=512,
        tile_col=128,
    )
    v_proj_s = offchip_load(
        v_proj,
        stride=[0, 0],
        out_shape_tiled=[64, 1],
        tile_row=512,
        tile_col=128,
    )
    cos_s = offchip_load(
        cos,
        stride=[0, 0],
        out_shape_tiled=[64, 1],
        tile_row=1,
        tile_col=32,
    )
    sin_s = offchip_load(
        sin,
        stride=[0, 0],
        out_shape_tiled=[64, 1],
        tile_row=1,
        tile_col=32,
    )

    # ------------------------------------------------------------------------
    # 2️⃣ The extra singleton stream dimension produced by `offchip_load`
    #    (shape …, 1, 1, C) is collapsed so that the stream shape matches that of
    #    `normed` (1, 64).  This is a pure stream‑shape flattening.
    # ------------------------------------------------------------------------
    cos_s = flatten(cos_s, min_rank=0, max_rank=1)   # (1, 64, 1, 32)
    sin_s = flatten(sin_s, min_rank=0, max_rank=1)   # (1, 64, 1, 32)

    # ------------------------------------------------------------------------
    # 3️⃣ Linear projections (Q, K, V) via batched matrix multiplication.
    # ------------------------------------------------------------------------
    Q_raw = binary_matmul(normed, q_proj_s)   # (1, 64, 1, 1, 512)
    K_raw = binary_matmul(normed, k_proj_s)   # (1, 64, 1, 1, 128)
    V_raw = binary_matmul(normed, v_proj_s)   # (1, 64, 1, 1, 128)

    # ------------------------------------------------------------------------
    # 4️⃣ Reshape the projection tensors from a flat hidden dimension into
    #    (num_heads, head_dim) tiles.
    #    The hidden dimension is split into chunks of size `head_dim` (=32) along
    #    the column axis; the resulting extra stream dimension (the number of
    #    chunks) is merged into the tile‑row dimension with `accum_retile_row`.
    # ------------------------------------------------------------------------
    Q = retile_streamify(Q_raw, chunk=32, split_row=False)   # (1, 64, 16, 1, 32)
    Q = accum_retile_row(Q, rank=1)                         # (1, 64, 16, 32)

    K = retile_streamify(K_raw, chunk=32, split_row=False)   # (1, 64, 4, 1, 32)
    K = accum_retile_row(K, rank=1)                         # (1, 64, 4, 32)

    V = retile_streamify(V_raw, chunk=32, split_row=False)   # (1, 64, 4, 1, 32)
    V = accum_retile_row(V, rank=1)                         # (1, 64, 4, 32)

    # ------------------------------------------------------------------------
    # 5️⃣ Per‑head RMSNorm (the same epsilon as in the reference).
    # ------------------------------------------------------------------------
    eps = 1e-6
    head_dim = cos_s.shape[-1]   # = 32

    def rms_norm(t):
        # mean of squares over the last (head) dimension
        sq = unary_square(t)                         # (…, H, D)
        sum_sq = unary_rowwise_sum(sq)               # (…, H, 1)
        mean_sq = unary_mul_imm(sum_sq, constant=1.0 / head_dim)
        add_eps = unary_add_imm(mean_sq, constant=eps)
        rsqrt = unary_rsqrt(add_eps)
        return binary_mul(t, rsqrt)

    Q = rms_norm(Q)
    K = rms_norm(K)
    # V is **not** RMS‑normalized (as in the reference).

    # ------------------------------------------------------------------------
    # 6️⃣ Rotary Positional Embedding.
    #    `rotate_half` swaps the two halves of the head dimension and negates the
    #    second half.  The operation is performed with a small helper that uses
    #    `torch.cat` – this is a pure reshaping step and does not involve any
    #    arithmetic on the data values.
    # ------------------------------------------------------------------------
    def rotate_half(x):
        half = x.shape[-1] // 2
        first = x[..., :half]
        second = x[..., half:]
        neg_second = unary_mul_imm(second, constant=-1.0)
        return torch.cat([neg_second, first], dim=-1)

    Q = binary_add(
        binary_mul(Q, cos_s),
        binary_mul(rotate_half(Q), sin_s),
    )
    K = binary_add(
        binary_mul(K, cos_s),
        binary_mul(rotate_half(K), sin_s),
    )
    # V stays unchanged.

    # ------------------------------------------------------------------------
    # 7️⃣ Return the three tensors in the order required by the contract.
    # ------------------------------------------------------------------------
    return Q, K, V