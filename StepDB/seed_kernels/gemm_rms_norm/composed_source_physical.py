"""gemm_rms_norm @ PHYSICAL TILE granularity for the matmul body.

Mirrors composed_source_logical.py for the rsqrt-computation prelude
(which is cheap and unaffected by tile choice), but downtiles A and B
to (sub, sub) = (16, 16) immediately after the matmul offchip_loads
and runs the entire matmul body in physical tiles — including the
rsqrt application via broadcasting (sub, 1) rsqrt tiles into the
(sub, sub) A/B streams.

The output is reassembled back to (big, big) via accum_retile_col +
accum_retile_row + accum_add (same K-reduction pattern as
gemm_parallel/composed_source.py), so the offchip_store sees one
(big, big) tile per (m_l, n_l) and produces (M, N) directly.

Contrast with composed_source_logical.py:
- Logical version keeps tile = (big, big) for matmul, broadcast, etc.
  The compiler's downtiler is expected to scale each op to HTS=16
  internally, allocating its own PMU bank per IR op.
- This version forces tile = (sub, sub) explicitly in the DSL for
  every heavy op, so the lowered IR never sees a "(big, big) matmul
  followed by a (big, big) broadcast-mul" pattern — it sees a stream
  of (sub, sub) ops the downtiler scales 1:1.
"""


def downtile(x, sub_r, sub_c):
    """Same helper as gemm_parallel/composed_source.py."""
    tile_r, tile_c = int(x.shape[-2]), int(x.shape[-1])
    assert tile_r % sub_r == 0 and tile_c % sub_c == 0, (
        f"downtile: ({tile_r},{tile_c}) not divisible by ({sub_r},{sub_c})"
    )
    n_row = tile_r // sub_r
    n_col = tile_c // sub_c
    y = retile_streamify(x, chunk=sub_r, split_row=True)
    y = retile_streamify(y, chunk=sub_c, split_row=False)
    y = reshape_stream(y, chunk_size=n_col, rank=0)
    y = reshape_stream(y, chunk_size=n_row, rank=1)
    return y


def tiled_reference(dims, tensors):
    big = 256
    sub = 16
    n_row = n_col = big // sub                # 16

    M, K, N = dims["M"], dims["K"], dims["N"]
    eps = tensors["eps"]

    M_tiles = M // big
    K_tiles = K // big
    N_tiles = N // big

    # =================================================================
    # rsqrt prelude — identical to composed_source_logical.py. The rsqrt
    # output is per-row scalar, so even at "logical tile" the work is
    # small; the comparison is meant to live in the matmul body below.
    # =================================================================
    A_rsqrt_load = offchip_load(
        tensors["A"],
        stride=(K_tiles, 1),
        out_shape_tiled=(M_tiles, K_tiles),
        tile_row=big, tile_col=big,
    )
    A_sq = unary_square(A_rsqrt_load)
    A_sq_row = unary_rowwise_sum(A_sq)
    A_sq_total = accum_add(A_sq_row, rank=1)
    A_mean = unary_mul_imm(A_sq_total, constant=1.0 / K)
    A_eps = unary_add_imm(A_mean, constant=eps)
    rsqrt_A = unary_rsqrt(A_eps)
    # rsqrt_A: stream (1, M_tiles), tile (big, 1).

    B_rsqrt_load = offchip_load(
        tensors["B"],
        stride=(N_tiles, 1),
        out_shape_tiled=(K_tiles, N_tiles),
        tile_row=big, tile_col=big,
    )
    B_sq = unary_square(B_rsqrt_load)
    B_sq_row = unary_rowwise_sum(B_sq)
    B_sq_total = accum_add(B_sq_row, rank=1)
    B_mean = unary_mul_imm(B_sq_total, constant=1.0 / N)
    B_eps = unary_add_imm(B_mean, constant=eps)
    rsqrt_B = unary_rsqrt(B_eps)
    # rsqrt_B: stream (1, K_tiles), tile (big, 1).

    # =================================================================
    # Downtile rsqrt_A and rsqrt_B to (sub, 1) sub-tiles so they
    # broadcast cleanly into the (sub, sub) matmul streams.
    # =================================================================
    # retile_streamify multiplies the innermost stream dim rather than
    # adding a new one — pair with reshape_stream to split (M_tiles*n_row)
    # back into (M_tiles, n_row) so bufferize can fold both as buffer dims.
    rsqrt_A_sub = retile_streamify(rsqrt_A, chunk=sub, split_row=True)
    rsqrt_A_sub = reshape_stream(rsqrt_A_sub, chunk_size=n_row, rank=0)
    # Stream (1, M_tiles, n_row), tile (sub, 1).
    rsqrt_A_buf = bufferize(rsqrt_A_sub, rank=2)
    # Buffer (M_tiles, n_row), tile (sub, 1).
    rsqrt_A_stream = streamify(
        rsqrt_A_buf,
        # Walk: (m_l, n_l, k_l, sub_r, sub_n, sub_c) → buffer[m_l, sub_r]
        # linear = m_l * n_row + sub_r → strides (n_row, 0, 0, 1, 0, 0).
        stride=(n_row, 0, 0, 1, 0, 0),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles, n_row, n_col, n_col),
    )
    # Stream (1, M_tiles, N_tiles, K_tiles, n_row, n_col, n_col), tile (sub, 1).

    rsqrt_B_sub = retile_streamify(rsqrt_B, chunk=sub, split_row=True)
    rsqrt_B_sub = reshape_stream(rsqrt_B_sub, chunk_size=n_row, rank=0)
    # Stream (1, K_tiles, n_row), tile (sub, 1) — n_row here splits the
    # K-rows within each big tile (B's tile has K on its row axis).
    rsqrt_B_buf = bufferize(rsqrt_B_sub, rank=2)
    # Buffer (K_tiles, n_row), tile (sub, 1).
    rsqrt_B_stream = streamify(
        rsqrt_B_buf,
        # B's stream order at sub-tile level: sub_c indexes the K-row
        # within a big tile (see gemm_parallel's docstring). So we need
        # rsqrt_B[k_l, sub_c]. linear = k_l * n_row + sub_c.
        # output walks (m_l, n_l, k_l, sub_r, sub_n, sub_c).
        stride=(0, 0, n_row, 0, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles, n_row, n_col, n_col),
    )
    # Stream (1, M_tiles, N_tiles, K_tiles, n_row, n_col, n_col), tile (sub, 1).

    # =================================================================
    # Matmul body — exactly the gemm_parallel pattern, with rsqrt
    # applied per-sub-tile via binary_mul before binary_map_accum.
    # =================================================================
    A_load = offchip_load(
        tensors["A"],
        stride=(K_tiles, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=big, tile_col=big,
    )
    A_buf = bufferize(downtile(A_load, sub, sub), rank=2)
    A_stream = streamify(
        A_buf,
        stride=(n_col, 0, 1),
        out_shape_tiled=(n_row, n_col, n_col),
    )
    # Stream (1, M_tiles, N_tiles, K_tiles, n_row, n_col, n_col), tile (sub, sub).

    B_load = offchip_load(
        tensors["B"],
        stride=(0, 1, N_tiles),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=big, tile_col=big,
    )
    B_buf = bufferize(downtile(B_load, sub, sub), rank=2)
    B_stream = streamify(
        B_buf,
        stride=(0, 1, n_col),
        out_shape_tiled=(n_row, n_col, n_col),
    )
    # Stream matches A_stream.

    # Apply rsqrt per sub-tile: tile (sub, sub) * tile (sub, 1) broadcasts
    # the rsqrt value across all sub_c (resp. sub_n) columns of the row.
    A_normed = binary_mul(A_stream, rsqrt_A_stream)
    B_normed = binary_mul(B_stream, rsqrt_B_stream)
    # Both: stream (1, M_tiles, N_tiles, K_tiles, n_row, n_col, n_col),
    # tile (sub, sub).

    # Stage 1: matmul + reduce sub_c (inner K).
    C_partial = binary_map_accum(A_normed, B_normed, rank=1)
    # Stream (1, M_tiles, N_tiles, K_tiles, n_row, n_col), tile (sub, sub).

    # Stage 2: reassemble big tile in-stream, reduce k_l on (big, big).
    C_col = accum_retile_col(C_partial, rank=1)
    # tile (sub, big), stream drops n_col.
    C_full = accum_retile_row(C_col, rank=1)
    # tile (big, big), stream drops n_row.
    C_red = accum_add(C_full, rank=1)
    # tile (big, big), reduces k_l. Stream (1, M_tiles, N_tiles).

    return offchip_store(C_red)
