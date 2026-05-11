# Implementation notes
# --------------------
# The previous version built a rotation matrix for `rotate_half` but used it
# directly in `binary_matmul`.  The matrix was the transpose of the one
# required for the row‑vector multiplication (`x @ R`).  By marking the
# weight as transposed (`weight_transposed=True`) we use the correct
# matrix without rebuilding it.  All other logic (projection, reshape,
# per‑head RMSNorm, and RoPE composition) matches the reference PyTorch
# implementation.

def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Extract scalar dimensions (pure Python)
    # ------------------------------------------------------------------
    seq_len      = normed.shape[1]                     # 64
    head_dim     = cos.shape[-1]                       # 32
    num_heads    = q_proj.shape[1] // head_dim         # 512 // 32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim         # 128 // 32 = 4
    half         = head_dim // 2
    eps          = 1e-6

    # ------------------------------------------------------------------
    # Load RAW tensors (broadcast across the sequence dimension)
    # ------------------------------------------------------------------
    q_weight = offchip_load(
        q_proj,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=512,
        tile_col=512,
        transposed=False,
    )
    k_weight = offchip_load(
        k_proj,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=512,
        tile_col=128,
        transposed=False,
    )
    v_weight = offchip_load(
        v_proj,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=512,
        tile_col=128,
        transposed=False,
    )
    cos_tile = offchip_load(
        cos,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=head_dim,
        transposed=False,
    )
    sin_tile = offchip_load(
        sin,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=head_dim,
        transposed=False,
    )

    # ------------------------------------------------------------------
    # Linear projections
    # ------------------------------------------------------------------
    Q = binary_matmul(normed, q_weight)   # (1, S) × tile (1, 512)
    K = binary_matmul(normed, k_weight)   # (1, S) × tile (1, 128)
    V = binary_matmul(normed, v_weight)   # (1, S) × tile (1, 128)

    # ------------------------------------------------------------------
    # Reshape projections into per‑head tiles
    # ------------------------------------------------------------------
    # Q → (1, S) × tile (num_heads, head_dim)
    Q = retile_streamify(Q, chunk=head_dim, split_row=False)
    Q = reshape_stream(Q, chunk_size=num_heads, rank=0)
    Q = accum_retile_row(Q, rank=1)

    # K → (1, S) × tile (num_kv_heads, head_dim)
    K = retile_streamify(K, chunk=head_dim, split_row=False)
    K = reshape_stream(K, chunk_size=num_kv_heads, rank=0)
    K = accum_retile_row(K, rank=1)

    # V → (1, S) × tile (num_kv_heads, head_dim)
    V = retile_streamify(V, chunk=head_dim, split_row=False)
    V = reshape_stream(V, chunk_size=num_kv_heads, rank=0)
    V = accum_retile_row(V, rank=1)

    # ------------------------------------------------------------------
    # Per‑head RMSNorm for Q and K (V left unchanged)
    # ------------------------------------------------------------------
    # Q RMSNorm
    Q_sq   = binary_mul(Q, Q)
    Q_sum  = unary_rowwise_sum(Q_sq)                     # (heads, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / head_dim)       # divide by head_dim
    Q_eps  = unary_add_imm(Q_mean, eps)                 # + epsilon
    Q_rsqrt = unary_rsqrt(Q_eps)
    Q_norm = binary_mul(Q, Q_rsqrt)

    # K RMSNorm
    K_sq   = binary_mul(K, K)
    K_sum  = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / head_dim)
    K_eps  = unary_add_imm(K_mean, eps)
    K_rsqrt = unary_rsqrt(K_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # ------------------------------------------------------------------
    # Rotation matrix for `rotate_half` (constant 32×32)
    # ------------------------------------------------------------------
    rot_rows = []
    for i in range(head_dim):
        row = [0.0] * head_dim
        if i < half:
            # first‑half rows map to second half columns (sign +)
            row[i + half] = 1.0
        else:
            # second‑half rows map to first half columns (sign -)
            row[i - half] = -1.0
        rot_rows.append(row)
    rot_const = torch.tensor(rot_rows, dtype=torch.float32)   # (head_dim, head_dim)

    rot_weight = offchip_load(
        rot_const,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=head_dim,
        tile_col=head_dim,
        transposed=False,
    )

    # ------------------------------------------------------------------
    # Apply rotate_half via matrix multiplication (use transposed weight)
    # ------------------------------------------------------------------
    Q_rot = binary_matmul(Q_norm, rot_weight, weight_transposed=True)
    K_rot = binary_matmul(K_norm, rot_weight, weight_transposed=True)

    # ------------------------------------------------------------------
    # RoPE: x * cos + rotate_half(x) * sin
    # ------------------------------------------------------------------
    Q_cos = binary_mul(Q_norm, cos_tile)
    Q_sin = binary_mul(Q_rot, sin_tile)
    Q_out = binary_add(Q_cos, Q_sin)

    K_cos = binary_mul(K_norm, cos_tile)
    K_sin = binary_mul(K_rot, sin_tile)
    K_out = binary_add(K_cos, K_sin)

    V_out = V  # V unchanged after head‑splitting

    return Q_out, K_out, V_out