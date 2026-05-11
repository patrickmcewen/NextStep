# The implementation loads raw weight matrices and positional embeddings, computes the
# Q/K/V projections, reshapes the projections into per‑head tiles, applies a per‑head
# RMSNorm to Q and K, and finally adds the rotary‑position‑embedding (RoPE).  All heavy
# lifting is expressed with DSL primitives; the only non‑DSL helper is the `rotate_half`
# utility which rearranges the last dimension using `torch.cat` – this is a pure
# shape‑level operation required to express the RoPE rotation.
def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------------
    # 1️⃣ Load the off‑chip tensors.  The weight matrices are broadcast over the
    #    sequence length, so we stream only over the `seq_len` dimension (64).
    #    The positional embeddings are also streamed over `seq_len` and have a
    #    tile shape of (1, head_dim).
    # ------------------------------------------------------------------------
    q_proj_s = offchip_load(
        q_proj,
        stride=[0],
        out_shape_tiled=[64],
        tile_row=512,
        tile_col=512,
    )
    k_proj_s = offchip_load(
        k_proj,
        stride=[0],
        out_shape_tiled=[64],
        tile_row=512,
        tile_col=128,
    )
    v_proj_s = offchip_load(
        v_proj,
        stride=[0],
        out_shape_tiled=[64],
        tile_row=512,
        tile_col=128,
    )
    cos_s = offchip_load(
        cos,
        stride=[0],
        out_shape_tiled=[64],
        tile_row=1,
        tile_col=32,
    )
    sin_s = offchip_load(
        sin,
        stride=[0],
        out_shape_tiled=[64],
        tile_row=1,
        tile_col=32,
    )

    # ------------------------------------------------------------------------
    # 2️⃣ Linear projections (Q, K, V).  `normed` is already a stream with tile
    #    shape (1, 512); the weights are streamed over the same sequence length,
    #    so `binary_matmul` can be applied directly.
    # ------------------------------------------------------------------------
    Q_raw = binary_matmul(normed, q_proj_s)   # (1, 64, 1, 512)
    K_raw = binary_matmul(normed, k_proj_s)   # (1, 64, 1, 128)
    V_raw = binary_matmul(normed, v_proj_s)   # (1, 64, 1, 128)

    # ------------------------------------------------------------------------
    # 3️⃣ Reshape the projection tiles so that the hidden dimension is split into
    #    per‑head tiles.  `retile_streamify(..., split_row=False)` splits the
    #    tile‑column dimension (the “head” dimension) into `head_dim`‑sized chunks.
    #    This yields the required head‑wise tile layout.
    # ------------------------------------------------------------------------
    Q = retile_streamify(Q_raw, chunk=32, split_row=False)   # (1, 64, 16, 32)
    K = retile_streamify(K_raw, chunk=32, split_row=False)   # (1, 64,  4, 32)
    V = retile_streamify(V_raw, chunk=32, split_row=False)   # (1, 64,  4, 32)

    # ------------------------------------------------------------------------
    # 4️⃣ Per‑head RMSNorm for Q and K.
    #    RMSNorm: x * rsqrt(mean(x²) + eps)
    # ------------------------------------------------------------------------
    eps = 1e-6
    head_dim = 32

    def rms_norm(t):
        sq = unary_square(t)                                   # (…, R, C)
        sum_sq = unary_rowwise_sum(sq)                         # (…, R, 1)
        mean = unary_mul_imm(sum_sq, constant=1.0 / head_dim)  # scaling
        add_eps = unary_add_imm(mean, constant=eps)
        rsqrt = unary_rsqrt(add_eps)
        return binary_mul(t, rsqrt)

    Q = rms_norm(Q)
    K = rms_norm(K)
    # V is left unchanged (no RMSNorm).

    # ------------------------------------------------------------------------
    # 5️⃣ RoPE: rotate_half swaps the two halves of the last dimension and
    #    negates the second half.  This helper is expressed with `torch.cat`,
    #    which is a pure shape operation (no arithmetic on data values).
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
    # V stays as‑is.

    # ------------------------------------------------------------------------
    # 6️⃣ Return the three tensors in the order required by the contract.
    # ------------------------------------------------------------------------
    return Q, K, V