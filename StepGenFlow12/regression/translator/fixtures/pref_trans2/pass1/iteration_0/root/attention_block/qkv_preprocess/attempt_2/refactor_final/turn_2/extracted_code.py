# Implementation notes
# --------------------
# 1. Load all RAW tensors (q_proj, k_proj, v_proj, cos, sin) with `offchip_load`.
#    The stride (0,) and `out_shape_tiled=(seq_len,)` broadcast each weight /
#    positional embedding across the whole sequence.
#
# 2. Perform the three linear projections with `binary_matmul`.  The result
#    tensors have tile shape (1, hidden) and are reshaped into per‑head tiles:
#       – split the column dimension into `head_dim`‑wide chunks with
#         `retile_streamify(..., split_row=False)`;
#       – reshape the added stream dimension back to (seq_len, num_heads) via
#         `reshape_stream`;
#       – merge the per‑token head count into the tile‑row dimension with
#         `accum_retile_row`.
#
# 3. Apply per‑head RMSNorm to Q and K.  The sequence of DSL ops reproduces
#    `x * rsqrt(mean(x²) + eps)`:
#       * square the tensor (`binary_mul`);
#       * sum over the head dimension (`unary_rowwise_sum`);
#       * scale by 1/head_dim (`unary_mul_imm`);
#       * add epsilon (`unary_add_imm`);
#       * rsqrt (`unary_rsqrt`);
#       * multiply the original tensor by the rsqrt (`binary_mul`).
#
# 4. RoPE – the rotate‑half operation is expressed as a matrix multiplication
#    with a constant 32×32 rotation matrix.  The matrix implements
#    `rotate_half(x) = [-x[..., half:], x[..., :half]]`.  It is materialised
#    as a constant tensor (allowed as a small literal) and loaded off‑chip so
#    that `binary_matmul` can apply it.
#
#    The final RoPE formula is
#        out = Q_norm * cos + rotate_half(Q_norm) * sin
#    applied separately to Q and K.
#
# 5. V is returned unchanged after the head‑splitting step.
#
# All operations are pure DSL calls; the only Python arithmetic is scalar
# dimension extraction and constant construction of the rotation matrix.

def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Extract scalar dimensions (pure Python)
    # ------------------------------------------------------------------
    seq_len = normed.shape[1]                     # 64
    head_dim = cos.shape[-1]                      # 32
    num_heads = q_proj.shape[1] // head_dim       # 512 // 32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim    # 128 // 32 = 4
    half = head_dim // 2
    eps = 1e-6

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
    # Split projection results into per‑head tiles
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
    # Per‑head RMSNorm for Q and K (V is left unchanged)
    # ------------------------------------------------------------------
    # Q RMSNorm
    Q_sq = binary_mul(Q, Q)
    Q_sum = unary_rowwise_sum(Q_sq)                     # sum over head_dim → (heads, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / head_dim)       # divide by head_dim
    Q_eps = unary_add_imm(Q_mean, eps)                  # add epsilon
    Q_rsqrt = unary_rsqrt(Q_eps)
    Q_norm = binary_mul(Q, Q_rsqrt)                     # scale

    # K RMSNorm
    K_sq = binary_mul(K, K)
    K_sum = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / head_dim)
    K_eps = unary_add_imm(K_mean, eps)
    K_rsqrt = unary_rsqrt(K_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # ------------------------------------------------------------------
    # Rotation matrix for `rotate_half` (constant 32×32)
    # ------------------------------------------------------------------
    # Build matrix R such that x @ R == rotate_half(x)
    rot_rows = []
    for i in range(head_dim):
        row = [0.0] * head_dim
        if i < half:
            # output index i takes -x[i+half]
            row[i + half] = -1.0
        else:
            # output index i takes  x[i-half]
            row[i - half] = 1.0
        rot_rows.append(row)
    rot_const = torch.tensor(rot_rows, dtype=torch.float32)  # shape (32,32)

    rot_weight = offchip_load(
        rot_const,
        stride=(0,),
        out_shape_tiled=(seq_len,),
        tile_row=head_dim,
        tile_col=head_dim,
        transposed=False,
    )

    # ------------------------------------------------------------------
    # Apply rotate_half via matrix multiplication
    # ------------------------------------------------------------------
    Q_rot = binary_matmul(Q_norm, rot_weight)   # (1, S) × tile (num_heads, head_dim)
    K_rot = binary_matmul(K_norm, rot_weight)   # (1, S) × tile (num_kv_heads, head_dim)

    # ------------------------------------------------------------------
    # RoPE: x * cos + rotate_half(x) * sin
    # ------------------------------------------------------------------
    Q_cos = binary_mul(Q_norm, cos_tile)
    Q_sin = binary_mul(Q_rot, sin_tile)
    Q_out = binary_add(Q_cos, Q_sin)

    K_cos = binary_mul(K_norm, cos_tile)
    K_sin = binary_mul(K_rot, sin_tile)
    K_out = binary_add(K_cos, K_sin)

    V_out = V  # V unchanged after head splitting

    return Q_out, K_out, V_out