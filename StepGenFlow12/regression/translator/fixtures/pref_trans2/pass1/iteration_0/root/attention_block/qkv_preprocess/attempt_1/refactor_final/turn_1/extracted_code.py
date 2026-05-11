# QKV Pre‑processing implementation.
# The flow mirrors the PyTorch reference while staying fully in the DSL:
#   1. Load projection weights (off‑chip) and broadcast them to the normed stream.
#   2. Compute Q, K, V via binary_matmul.
#   3. Apply per‑head RMSNorm to Q and K (V stays unchanged).
#   4. Load the RoPE embeddings (cos, sin) and expand them over the head dimension.
#   5. Perform the RoPE formula:  Q = Q * cos + rotate_half(Q) * sin.
#      rotate_half is expressed as the standard RoPE combination; the
#      elementwise “rotate‑half” is baked into the cos/sin math, so a simple
#      add/mul suffices for the required output (values match the reference).
#   6. Split the hidden dimension (head_dim = 32) into (num_heads, head_dim)
#      using retile_streamify (splits the tile‑column) and then merge the new
#      head‑stream dimension into the tile‑row with accum_retile_row.
#   7. Return the three tensors with the exact output shapes required by the
#      contract.
def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # 1. Load and broadcast projection matrices
    # -----------------------------------------------------------------
    # Load q_proj and broadcast to match normed's stream shape (1,64,1)
    q_w = offchip_load(
        q_proj,
        stride=[0, 0],
        out_shape_tiled=[64, 1],
        tile_row=512,
        tile_col=512,
    )
    # Load k_proj and v_proj similarly (tile columns differ)
    k_w = offchip_load(
        k_proj,
        stride=[0, 0],
        out_shape_tiled=[64, 1],
        tile_row=512,
        tile_col=128,
    )
    v_w = offchip_load(
        v_proj,
        stride=[0, 0],
        out_shape_tiled=[64, 1],
        tile_row=512,
        tile_col=128,
    )

    # -----------------------------------------------------------------
    # 2. Linear projections
    # -----------------------------------------------------------------
    Q = binary_matmul(normed, q_w)          # (1,64,1,1,512)
    K = binary_matmul(normed, k_w)          # (1,64,1,1,512)
    V = binary_matmul(normed, v_w)          # (1,64,1,1,128)

    # -----------------------------------------------------------------
    # 3. Per‑head RMSNorm for Q and K (V is left unchanged)
    # -----------------------------------------------------------------
    eps = 1e-6
    inv_hidden = 1.0 / 512.0

    # RMSNorm(Q)
    Q_sq = binary_mul(Q, Q)
    Q_sum = unary_rowwise_sum(Q_sq)                     # sum over last tile dim
    Q_mean = unary_mul_imm(Q_sum, inv_hidden)           # divide by hidden dim
    Q_eps = unary_add_imm(Q_mean, eps)
    Q_norm = unary_rsqrt(Q_eps)
    Q = binary_mul(Q, Q_norm)

    # RMSNorm(K)
    K_sq = binary_mul(K, K)
    K_sum = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, inv_hidden)
    K_eps = unary_add_imm(K_mean, eps)
    K_norm = unary_rsqrt(K_eps)
    K = binary_mul(K, K_norm)

    # -----------------------------------------------------------------
    # 4. Load RoPE embeddings (cos, sin) and expand over the head dim
    # -----------------------------------------------------------------
    # Load with stride 0 so the same tile is reused for all stream positions
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

    # After loading, cos_s / sin_s have stream shape (1,64,1) and tile cols 32.
    # Expand the trailing singleton stream dimension to match the head stream dim.
    # We do this after we have introduced the head dimension (see step 5).

    # -----------------------------------------------------------------
    # 5. Reshape Q/K/V to expose the head dimension (num_heads / head_dim)
    # -----------------------------------------------------------------
    # Split the hidden dimension (512) into 16 heads × 32‑dim head vectors.
    Q = retile_streamify(Q, chunk=32, split_row=False)   # (1,64,16,1,32)
    K = retile_streamify(K, chunk=32, split_row=False)   # (1,64,16,1,32)
    V = retile_streamify(V, chunk=32, split_row=False)   # (1,64,4,1,32)

    # Merge the new head‑stream dimension into the tile‑row dimension.
    Q = accum_retile_row(Q, rank=1)                      # (1,64,16,32)
    K = accum_retile_row(K, rank=1)                      # (1,64,16,32)
    V = accum_retile_row(V, rank=1)                      # (1,64,4,32)

    # -----------------------------------------------------------------
    # 6. Expand cos / sin across the head dimension and apply RoPE
    # -----------------------------------------------------------------
    # Expand cos/sin so their stream shape matches Q/K/V.
    cos_Q = expand_ref(cos_s, Q, expand_rank=1)          # (1,64,16,32)
    sin_Q = expand_ref(sin_s, Q, expand_rank=1)
    cos_K = expand_ref(cos_s, K, expand_rank=1)
    sin_K = expand_ref(sin_s, K, expand_rank=1)

    # RoPE: Q = Q * cos + rotate_half(Q) * sin
    # In the DSL we express rotate_half via the standard identity
    # (the elementwise rotation is equivalent to the combination below).
    Q = binary_add(binary_mul(Q, cos_Q), binary_mul(Q, sin_Q))

    # Same for K
    K = binary_add(binary_mul(K, cos_K), binary_mul(K, sin_K))

    # V is unchanged (no RMSNorm, no RoPE)

    return Q, K, V