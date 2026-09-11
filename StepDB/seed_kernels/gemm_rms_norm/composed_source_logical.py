"""gemm_rms_norm @ LOGICAL TILE granularity.

Every DSL op here operates on (big, big) tiles. There is no explicit
downtile in this kernel — the hwsim compiler's downtiler pass is left
responsible for lowering every op (UnaryMap, BinaryMap, Accum,
BinaryMapAccum, ...) to the hardware tile size (HTS=16x16).

The intended contrast is with composed_source_physical.py, which
downtiles once at the top and then lives in (16, 16) tiles for every
op except the final offchip_store.

This file is what you would write if you trusted hwsim to schedule
the lowering for you. Per the downtiler analysis, that means every
IR op boundary becomes its own PMU bank backed by a Bufferize with
storage_limit=10, with no fusion across nodes.
"""


def tiled_reference(dims, tensors):
    big = 256
    M, K, N = dims["M"], dims["K"], dims["N"]
    eps = tensors["eps"]

    M_tiles = M // big
    K_tiles = K // big
    N_tiles = N // big

    # ----- rms_norm(A): per-row rsqrt along K -----
    # Separate offchip_load (no broadcast) — matches the existing rms_norm
    # step_impl pattern. Stream (1, M_tiles, K_tiles), tile (big, big).
    A_rsqrt_load = offchip_load(
        tensors["A"],
        stride=(K_tiles, 1),
        out_shape_tiled=(M_tiles, K_tiles),
        tile_row=big, tile_col=big,
    )
    A_sq = unary_square(A_rsqrt_load)
    A_sq_row = unary_rowwise_sum(A_sq)                # tile (big, 1)
    # Reduce across K_tiles (innermost stream dim).
    A_sq_total = accum_add(A_sq_row, rank=1)
    # Stream (1, M_tiles), tile (big, 1).
    A_mean = unary_mul_imm(A_sq_total, constant=1.0 / K)
    A_eps = unary_add_imm(A_mean, constant=eps)
    rsqrt_A = unary_rsqrt(A_eps)
    # rsqrt_A: stream (1, M_tiles), tile (big, 1).

    # ----- rms_norm(B): per-K-row rsqrt along N -----
    # Stream (1, K_tiles, N_tiles), tile (big, big). N_tiles is innermost
    # so accum_add(rank=1) reduces along N naturally.
    B_rsqrt_load = offchip_load(
        tensors["B"],
        stride=(N_tiles, 1),
        out_shape_tiled=(K_tiles, N_tiles),
        tile_row=big, tile_col=big,
    )
    B_sq = unary_square(B_rsqrt_load)
    B_sq_row = unary_rowwise_sum(B_sq)                # tile (big, 1)
    B_sq_total = accum_add(B_sq_row, rank=1)
    # Stream (1, K_tiles), tile (big, 1).
    B_mean = unary_mul_imm(B_sq_total, constant=1.0 / N)
    B_eps = unary_add_imm(B_mean, constant=eps)
    rsqrt_B = unary_rsqrt(B_eps)
    # rsqrt_B: stream (1, K_tiles), tile (big, 1).

    # ----- Matmul with N/M broadcast on A/B respectively -----
    # Reload A with N broadcast: each A big tile re-fetched N_tiles times.
    A_load = offchip_load(
        tensors["A"],
        stride=(K_tiles, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=big, tile_col=big,
    )
    # Stream (1, M_tiles, N_tiles, K_tiles), tile (big, big).

    B_load = offchip_load(
        tensors["B"],
        stride=(0, 1, N_tiles),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=big, tile_col=big,
    )
    # Stream (1, M_tiles, N_tiles, K_tiles), tile (big, big).

    # Align rsqrt_A with A_load. rsqrt_A is (1, M_tiles); A_load is
    # (1, M_tiles, N_tiles, K_tiles). repeat_static appends an innermost
    # stream dim, so two applications give (1, M_tiles, N_tiles, K_tiles).
    rsqrt_A_aligned = repeat_static(rsqrt_A, factor=N_tiles)
    rsqrt_A_aligned = repeat_static(rsqrt_A_aligned, factor=K_tiles)
    # Stream (1, M_tiles, N_tiles, K_tiles), tile (big, 1).
    # binary_mul broadcasts (big, big) * (big, 1) over the tile via torch.
    A_normed = binary_mul(A_load, rsqrt_A_aligned)
    # Tile (big, big), stream (1, M_tiles, N_tiles, K_tiles).

    # Align rsqrt_B with B_load. rsqrt_B is (1, K_tiles); we need to
    # broadcast it across (M_tiles, N_tiles) while keeping K_tiles
    # innermost. Bufferize the small rsqrt_B stream and restream with
    # zero strides on (m_l, n_l) — buffer cost = K * 4 bytes.
    rsqrt_B_buf = bufferize(rsqrt_B, rank=1)
    rsqrt_B_aligned = streamify(
        rsqrt_B_buf,
        stride=(0, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
    )
    # Stream (1, M_tiles, N_tiles, K_tiles), tile (big, 1).
    B_normed = binary_mul(B_load, rsqrt_B_aligned)
    # Tile (big, big), stream (1, M_tiles, N_tiles, K_tiles).

    # Matmul on big tiles, reducing k_l.
    C = binary_map_accum(A_normed, B_normed, rank=1)
    # Stream (1, M_tiles, N_tiles), tile (big, big).

    return offchip_store(C)
