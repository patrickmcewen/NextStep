# QKV preprocessing node.
# Implements the projection, per‑head RMSNorm, a simplified RoPE (Q = Q*cos + Q*sin),
# and reshapes the hidden dimension (head_dim = 32) into the required
# (num_heads, head_dim) tiling:
#   Q, K → (1, 64, 16, 32)   (num_heads = 16)
#   V    → (1, 64, 4, 32)    (num_kv_heads = 4)
def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # 1. Load projection weights and broadcast to the normed stream shape
    # -----------------------------------------------------------------
    q_w = offchip_load(
        q_proj,
        stride=[0],
        out_shape_tiled=[64],
        tile_row=512,
        tile_col=512,
    )
    k_w = offchip_load(
        k_proj,
        stride=[0],
        out_shape_tiled=[64],
        tile_row=512,
        tile_col=128,
    )
    v_w = offchip_load(
        v_proj,
        stride=[0],
        out_shape_tiled=[64],
        tile_row=512,
        tile_col=128,
    )

    # -----------------------------------------------------------------
    # 2. Linear projections
    # -----------------------------------------------------------------
    Q = binary_matmul(normed, q_w)   # (1,64,1,512)
    K = binary_matmul(normed, k_w)   # (1,64,1,512)
    V = binary_matmul(normed, v_w)   # (1,64,1,128)

    # -----------------------------------------------------------------
    # 3. Per‑head RMSNorm for Q and K (V is left unchanged)
    # -----------------------------------------------------------------
    eps = 1e-6
    inv_hidden = 1.0 / 512.0

    # RMSNorm(Q)
    Q_sq   = binary_mul(Q, Q)
    Q_sum  = unary_rowwise_sum(Q_sq)                     # (1,64,1,1)
    Q_mean = unary_mul_imm(Q_sum, inv_hidden)           # divide by hidden dim
    Q_eps  = unary_add_imm(Q_mean, eps)
    Q_rsqrt = unary_rsqrt(Q_eps)
    Q = binary_mul(Q, Q_rsqrt)

    # RMSNorm(K)
    K_sq   = binary_mul(K, K)
    K_sum  = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, inv_hidden)
    K_eps  = unary_add_imm(K_mean, eps)
    K_rsqrt = unary_rsqrt(K_eps)
    K = binary_mul(K, K_rsqrt)

    # -----------------------------------------------------------------
    # 4. Reshape Q/K/V to expose the head dimension
    #    (head_dim = 32, num_heads = 16, num_kv_heads = 4)
    # -----------------------------------------------------------------
    # Q and K: hidden 512 → 16 heads × 32 dims
    Q = retile_streamify(Q, chunk=32, split_row=False)   # (1,1024,1,32)
    Q = reshape_stream(Q, chunk_size=16, rank=0)         # (1,64,16,1,32)
    Q = accum_retile_row(Q, rank=1)                      # (1,64,16,32)

    K = retile_streamify(K, chunk=32, split_row=False)   # (1,1024,1,32)
    K = reshape_stream(K, chunk_size=16, rank=0)         # (1,64,16,1,32)
    K = accum_retile_row(K, rank=1)                      # (1,64,16,32)

    # V: hidden 128 → 4 kv‑heads × 32 dims
    V = retile_streamify(V, chunk=32, split_row=False)   # (1,256,1,32)
    V = reshape_stream(V, chunk_size=4, rank=0)          # (1,64,4,1,32)
    V = accum_retile_row(V, rank=1)                      # (1,64,4,32)

    # -----------------------------------------------------------------
    # 5. Load RoPE embeddings (cos, sin) – stream shape matches (1,64)
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # 6. Apply (simplified) RoPE: Q = Q*cos + Q*sin   (same for K)
    #    Tile‑row dimension (num_heads) is broadcast automatically.
    # -----------------------------------------------------------------
    Q = binary_add(binary_mul(Q, cos_s), binary_mul(Q, sin_s))
    K = binary_add(binary_mul(K, cos_s), binary_mul(K, sin_s))

    return Q, K, V