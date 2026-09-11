"""
Implementation notes:
- RMSNorm is performed as  x * rsqrt(mean(x²) + eps).
  We compute x² with `unary_square`, sum across the last (head‑dim) axis
  with `unary_rowwise_sum`, scale by 1/head_dim, add eps, take rsqrt and
  finally multiply with the original tensor.
- `cos` and `sin` are RAW Off‑chip tensors. They are loaded with
  `offchip_load`. The loader always adds a leading singleton stream
  dimension, so we collapse the two stream dimensions (1, seq_len) back
  to a single one with `flatten`.
- The RoPE rotation `rotate_half(x)` is expressed using only DSL ops:
  * split the column dimension into two halves via `retile_streamify`,
    which turns the column size into a new stream factor (seq_len × 2)
    and shrinks the tile columns to `head_dim//2`.
  * `parallelize(..., 2)` separates the two halves into distinct streams.
  * negate the second half (`unary_mul_imm` with -1.0).
  * `static_reassemble` interleaves the streams as
    [‑second_half, first_half] → implements the required
    `concat([-x_half2, x_half1])`.
  * `reshape_stream` reshapes the combined stream (seq_len*2) into a
    two‑dimensional stream (seq_len, 2) so that the inner dimension can be
    merged into the tile columns with `accum_retile_col`.  This restores the
    original column size while keeping the stream mask identical to the
    original tensors.
- The same rotation logic is applied to both Q and K after RMSNorm.
- V is passed through unchanged.
- All tensor arithmetic is expressed via the DSL functions; the only
  Python arithmetic is on scalar meta‑data (shape sizes, constants).
"""
def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    # ----- RMSNorm -------------------------------------------------
    def rms_norm(x):
        # x²
        sq = unary_square(x)
        # sum over head dimension (last tile dim), keep dim → (…, heads, 1)
        sum_sq = unary_rowwise_sum(sq)
        # mean = sum / head_dim
        head_dim = x.shape[-1]               # static integer
        inv_head_dim = 1.0 / head_dim
        mean_sq = unary_mul_imm(sum_sq, inv_head_dim)
        # add epsilon and rsqrt
        eps = 1e-6
        mean_eps = unary_add_imm(mean_sq, eps)
        rsqrt = unary_rsqrt(mean_eps)
        # x * rsqrt (broadcast over head dim)
        return binary_mul(x, rsqrt)

    Q_norm = rms_norm(Q)
    K_norm = rms_norm(K)

    # ----- Load cos / sin (RAW) ------------------------------------
    # tile shape is (1, head_dim)
    tile_row = 1
    tile_col = Q_norm.shape[-1]   # head_dim (32)
    stride = (1,)
    out_shape = (Q_norm.shape[0],)   # seq_len (64)

    cos_loaded = offchip_load(cos, stride=stride,
                              out_shape_tiled=out_shape,
                              tile_row=tile_row, tile_col=tile_col)
    cos_loaded = flatten(cos_loaded, min_rank=0, max_rank=1)   # (seq_len,1,head_dim)

    sin_loaded = offchip_load(sin, stride=stride,
                              out_shape_tiled=out_shape,
                              tile_row=tile_row, tile_col=tile_col)
    sin_loaded = flatten(sin_loaded, min_rank=0, max_rank=1)   # (seq_len,1,head_dim)

    # ----- Helper to perform rotate_half using only DSL ops --------
    def rotate_half(x):
        half = x.shape[-1] // 2                     # split columns
        # split columns into two stream elements per token
        split = retile_streamify(x, chunk=half, split_row=False)
        # separate the two halves (first, second)
        halves = parallelize(split, 2)                     # [first, second]
        first_half = halves[0]
        second_half = halves[1]
        # negate the second half
        second_neg = unary_mul_imm(second_half, -1.0)
        # interleave as [‑second, first] to achieve rotate_half
        interleaved = static_reassemble([second_neg, first_half])
        # reshape stream (seq_len*2) → (seq_len, 2)
        reshaped = reshape_stream(interleaved, chunk_size=2, rank=0, add_outer_dim=False)
        # merge the inner stream dim into tile columns → restores original cols
        merged = accum_retile_col(reshaped, rank=1)
        return merged

    # ----- Apply RoPE to Q and K -----------------------------------
    Q_rot = rotate_half(Q_norm)
    K_rot = rotate_half(K_norm)

    # Q' = Q_norm * cos + Q_rot * sin
    Q_cos = binary_mul(Q_norm, cos_loaded)
    Q_sin = binary_mul(Q_rot, sin_loaded)
    Q_out = binary_add(Q_cos, Q_sin)

    # K' = K_norm * cos + K_rot * sin
    K_cos = binary_mul(K_norm, cos_loaded)
    K_sin = binary_mul(K_rot, sin_loaded)
    K_out = binary_add(K_cos, K_sin)

    # V is unchanged
    return Q_out, K_out, V