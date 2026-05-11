# The implementation loads off‑chip tensors, performs the Q/K/V projections, splits the
# hidden dimension into per‑head tiles, applies a per‑head RMSNorm, and finally adds
# rotary‑position embeddings (RoPE).  All heavy lifting uses the DSL primitives; the
# only non‑DSL helper is the `rotate_half` utility, which reshapes the last dimension
# using `torch.cat` (a shape‑only operation).
def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------------
    # 1️⃣ Extract scalar dimensions (pure Python – allowed).
    # ------------------------------------------------------------------------
    seq_len = normed.shape[1]          # stream dimension of `normed`
    head_dim = cos.shape[-1]           # 32
    num_heads = q_proj.shape[1] // head_dim      # 512 // 32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim   # 128 // 32 = 4
    eps = 1e-6

    # ------------------------------------------------------------------------
    # 2️⃣ Load the off‑chip weight matrices (broadcast across the sequence) and
    #    the positional embeddings (broadcast across heads).
    # ------------------------------------------------------------------------
    q_proj_s = offchip_load(
        q_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=512,
        tile_col=512,
    )
    k_proj_s = offchip_load(
        k_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=512,
        tile_col=128,
    )
    v_proj_s = offchip_load(
        v_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=512,
        tile_col=128,
    )
    # Cos / sin for Q (num_heads)
    cos_q = offchip_load(
        cos,
        stride=[1, 0],
        out_shape_tiled=[seq_len, num_heads],
        tile_row=1,
        tile_col=head_dim,
    )
    sin_q = offchip_load(
        sin,
        stride=[1, 0],
        out_shape_tiled=[seq_len, num_heads],
        tile_row=1,
        tile_col=head_dim,
    )
    # Cos / sin for K (num_kv_heads)
    cos_k = offchip_load(
        cos,
        stride=[1, 0],
        out_shape_tiled=[seq_len, num_kv_heads],
        tile_row=1,
        tile_col=head_dim,
    )
    sin_k = offchip_load(
        sin,
        stride=[1, 0],
        out_shape_tiled=[seq_len, num_kv_heads],
        tile_row=1,
        tile_col=head_dim,
    )

    # ------------------------------------------------------------------------
    # 3️⃣ Flatten the (seq_len × heads) streams of the positional embeddings so
    #    they match the flattened token‑head streams produced after retile.
    # ------------------------------------------------------------------------
    cos_q = flatten(cos_q, min_rank=0, max_rank=1)   # (1, seq_len*num_heads, 1, head_dim)
    sin_q = flatten(sin_q, min_rank=0, max_rank=1)
    cos_k = flatten(cos_k, min_rank=0, max_rank=1)   # (1, seq_len*num_kv_heads, 1, head_dim)
    sin_k = flatten(sin_k, min_rank=0, max_rank=1)

    # ------------------------------------------------------------------------
    # 4️⃣ Linear projections (batched matmul).  `normed` is already a stream with
    #    tile shape (1, hidden_dim).  The weight tensors are broadcasted over the
    #    sequence dimension.
    # ------------------------------------------------------------------------
    Q_raw = binary_matmul(normed, q_proj_s)   # (1, seq_len, 1, hidden_dim)
    K_raw = binary_matmul(normed, k_proj_s)   # (1, seq_len, 1, hidden_dim)
    V_raw = binary_matmul(normed, v_proj_s)   # (1, seq_len, 1, hidden_dim)

    # ------------------------------------------------------------------------
    # 5️⃣ Split the hidden dimension into per‑head tiles (chunk = head_dim).
    #    This also merges the token and head dimensions into a single stream
    #    dimension (seq_len × heads).
    # ------------------------------------------------------------------------
    Q = retile_streamify(Q_raw, chunk=head_dim, split_row=False)   # (1, seq_len*num_heads, 1, head_dim)
    K = retile_streamify(K_raw, chunk=head_dim, split_row=False)   # (1, seq_len*num_kv_heads, 1, head_dim)
    V = retile_streamify(V_raw, chunk=head_dim, split_row=False)   # (1, seq_len*num_kv_heads, 1, head_dim)

    # ------------------------------------------------------------------------
    # 6️⃣ Per‑head RMSNorm (applied to Q and K only).
    # ------------------------------------------------------------------------
    def rms_norm(t):
        sq = unary_square(t)                                   # (…, H, D)
        sum_sq = unary_rowwise_sum(sq)                         # (…, H, 1)
        mean = unary_mul_imm(sum_sq, constant=1.0 / head_dim)  # mean of squares
        add_eps = unary_add_imm(mean, constant=eps)
        rsqrt = unary_rsqrt(add_eps)
        return binary_mul(t, rsqrt)

    Q = rms_norm(Q)
    K = rms_norm(K)
    # V is left unchanged.

    # ------------------------------------------------------------------------
    # 7️⃣ Helper to apply the rotary transform: swap halves of the last dimension
    #    and negate the second half.
    # ------------------------------------------------------------------------
    def rotate_half(x):
        half = x.shape[-1] // 2
        first = x[..., :half]
        second = x[..., half:]
        neg_second = unary_mul_imm(second, constant=-1.0)
        return torch.cat([neg_second, first], dim=-1)

    # ------------------------------------------------------------------------
    # 8️⃣ RoPE (Q and K only).  The positional tensors have been flattened to the
    #    same (seq_len × heads) stream shape as the corresponding Q/K tensors.
    # ------------------------------------------------------------------------
    Q = binary_add(
        binary_mul(Q, cos_q),
        binary_mul(rotate_half(Q), sin_q),
    )
    K = binary_add(
        binary_mul(K, cos_k),
        binary_mul(rotate_half(K), sin_k),
    )
    # V remains as‑is.

    # ------------------------------------------------------------------------
    # 9️⃣ Return the three tensors in the order required by the contract.
    # ------------------------------------------------------------------------
    return Q, K, V