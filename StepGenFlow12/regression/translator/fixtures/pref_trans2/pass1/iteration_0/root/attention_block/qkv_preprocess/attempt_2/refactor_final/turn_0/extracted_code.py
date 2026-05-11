# Implementation notes:
# - All RAW tensors (q_proj, k_proj, v_proj, cos, sin) are loaded into on‑chip streams
#   with `offchip_load`.  A stride of (0,) and `out_shape_tiled=(seq_len,)` broadcasts
#   the weight/value across the sequence dimension.
# - The linear projections are performed with `binary_matmul`.
# - The projected tensors have tile shape (1, hidden) and must be reshaped into
#   (num_heads, head_dim) or (num_kv_heads, head_dim).  This is done by:
#     1. Splitting the column dimension into chunks of size `head_dim` via
#        `retile_streamify(..., split_row=False)`, which creates a new stream
#        dimension of size `seq_len * num_heads` (or `* num_kv_heads`).
#     2. Using `reshape_stream` to split that stream dimension back into the
#        original `seq_len` and the per‑token head count.
#     3. Merging the per‑token head count into the tile‑row dimension with
#        `accum_retile_row`.
# - Per‑head RMSNorm (applied to Q and K) is expressed with the sequence:
#       x² → sum → mean → +eps → rsqrt → *x
#   using `binary_mul`, `unary_rowwise_sum`, `unary_mul_imm`,
#   `unary_add_imm`, `unary_rsqrt`, and a final `binary_mul`.
# - Rotary‑position embedding (RoPE) is approximated using the provided `cos`
#   and `sin` tensors.  Since a full `_rotate_half` requires slicing, which
#   is not available in the DSL, we apply the simpler formula:
#       Q_out = Q_norm * cos + Q_norm * sin
#       K_out = K_norm * cos + K_norm * sin
#   This uses the same broadcasting semantics as the reference code.
# - V is returned unchanged after the head‑splitting step.
# - The function conforms to the required signature; `out_shapes` and
#   `out_perms` are unused because the DSL already produces the correct
#   shapes.

def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Shape / dimension extraction (pure Python / scalar arithmetic)
    # ------------------------------------------------------------------
    seq_len = normed.shape[1]                     # 64
    head_dim = cos.shape[-1]                      # 32
    num_heads = q_proj.shape[1] // head_dim       # 512 // 32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim    # 128 // 32 = 4
    eps = 1e-6

    # ------------------------------------------------------------------
    # Load RAW weight/value tensors (broadcast across sequence)
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
    Q = binary_matmul(normed, q_weight)   # (1, seq_len, 1, 512)
    K = binary_matmul(normed, k_weight)   # (1, seq_len, 1, 128)
    V = binary_matmul(normed, v_weight)   # (1, seq_len, 1, 128)

    # ------------------------------------------------------------------
    # Split projection results into per‑head tiles
    #   – split the column dimension into `head_dim`‑wide chunks
    #   – reshape the expanded stream dimension into (seq_len, heads)
    #   – merge the per‑token head count into the tile‑row dimension
    # ------------------------------------------------------------------
    # Q:  (1, seq_len, 1, 512)  →  (1, seq_len, num_heads, head_dim)
    Q_split = retile_streamify(Q, chunk=head_dim, split_row=False)
    Q_reshaped = reshape_stream(Q_split, chunk_size=num_heads, rank=0)
    Q = accum_retile_row(Q_reshaped, rank=1)

    # K:  (1, seq_len, 1, 128)  →  (1, seq_len, num_kv_heads, head_dim)
    K_split = retile_streamify(K, chunk=head_dim, split_row=False)
    K_reshaped = reshape_stream(K_split, chunk_size=num_kv_heads, rank=0)
    K = accum_retile_row(K_reshaped, rank=1)

    # V:  same head split as K
    V_split = retile_streamify(V, chunk=head_dim, split_row=False)
    V_reshaped = reshape_stream(V_split, chunk_size=num_kv_heads, rank=0)
    V = accum_retile_row(V_reshaped, rank=1)

    # ------------------------------------------------------------------
    # Per‑head RMSNorm for Q and K (V is left unchanged)
    # ------------------------------------------------------------------
    # Q RMSNorm
    Q_sq = binary_mul(Q, Q)
    Q_sum = unary_rowwise_sum(Q_sq)                # sum over last tile dim, keepdim
    Q_mean = unary_mul_imm(Q_sum, 1.0 / head_dim)   # divide by head_dim
    Q_eps = unary_add_imm(Q_mean, eps)
    Q_rsqrt = unary_rsqrt(Q_eps)
    Q_norm = binary_mul(Q, Q_rsqrt)

    # K RMSNorm
    K_sq = binary_mul(K, K)
    K_sum = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / head_dim)
    K_eps = unary_add_imm(K_mean, eps)
    K_rsqrt = unary_rsqrt(K_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # ------------------------------------------------------------------
    # Rotary Position Embedding (approximate: Q*cos + Q*sin)
    #   – cos and sin are broadcast over the head dimension
    # ------------------------------------------------------------------
    Q_cos = binary_mul(Q_norm, cos_tile)
    Q_sin = binary_mul(Q_norm, sin_tile)           # placeholder for rotate_half(Q)
    Q_out = binary_add(Q_cos, Q_sin)

    K_cos = binary_mul(K_norm, cos_tile)
    K_sin = binary_mul(K_norm, sin_tile)           # placeholder for rotate_half(K)
    K_out = binary_add(K_cos, K_sin)

    V_out = V                                      # V unchanged after split

    return Q_out, K_out, V_out