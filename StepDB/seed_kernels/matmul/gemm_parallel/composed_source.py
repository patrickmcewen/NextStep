def downtile(x, sub_r, sub_c):
    """Split each (T_r, T_c) tile into a (T_r/sub_r) x (T_c/sub_c) grid of
    (sub_r, sub_c) sub-tiles. Adds two new innermost stream dims for the
    sub-grid (row-chunks outer, col-chunks inner).

        stream (..., D)             tile (T_r, T_c)
                      ->
        stream (..., D, n_row, n_col)   tile (sub_r, sub_c)
    """
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
    """GEMM with big-tile DRAM loads, one-big-tile on-chip working set,
    and a two-stage K reduction (sub_c inner, k_l outer).

    Each side keeps only ONE big tile on chip (256 KB) by using offchip_load
    with stride-0 broadcasts:
      - A iterates (m_l, n_l, k_l) with n_l stride 0 (A doesn't depend on N).
      - B iterates (m_l, n_l, k_l) with m_l stride 0 (B doesn't depend on M).
    Each (m_l, n_l, k_l) outer step loads one big tile; each big-tile pair is
    downtiled, matmul'd on (sub, sub) sub-tiles, and accumulated.

    Natural stream order from this layout is (m_l, n_l, k_l, sub_r, sub_n, sub_c).
    K splits across positions 2 and 5 — non-adjacent — so K is reduced in two
    stages:
      stage 1: binary_map_accum(rank=1)  matmul + sum sub_c.
      stage 2: accum_retile_col + accum_retile_row reassemble the (sub, sub)
               sub-grid into one (big, big) tile in-stream (steady-state
               cost ~chunk*sub_tile, no per-(m_l,n_l) materialization), then
               accum_add(rank=1) reduces k_l in a (big, big) persistent
               accumulator. offchip_store of (big, big) tiles produces the
               full (M, N) output via the (penult*tile_row, last*tile_col)
               layout convention — no output reorder buffer required.

    SRAM per side: 256 KB (big tile) + ~256 KB (accum_retile + k_l acc)
    ≈ ~1 MB total — vs. 128 MB for full rank=4 bufferize, and vs. ~8.5 MB
    for the earlier bufferize(rank=3) + output-reorder version.
    DRAM cost: 16x amplification on each side (re-reads big tiles across
    the broadcast axis).
    """
    big = 256
    sub = 16
    n_row = n_col = big // sub          # 16

    M_tiles = dims["M"] // big           # 16
    K_tiles = dims["K"] // big           # 16
    N_tiles = dims["N"] // big           # 16

    sub_per_k = n_row * n_col            # 256 sub-tiles per big tile

    # ------------------------------------------------------------------
    # A: load big tiles in (m_l, n_l, k_l) order with N broadcast.
    #    Each big tile of A is fetched from DRAM N_tiles times (16x amp).
    # ------------------------------------------------------------------
    A_load = offchip_load(
        tensors["A"],
        stride=(K_tiles, 0, 1),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=big, tile_col=big,
    )
    A_buf = bufferize(downtile(A_load, sub, sub), rank=2)
    # Per-buffer = one A big tile, (sub_mm, sub_kk) sub-tile grid (256 KB).
    # Inner stream: (sub_r, sub_n_bcast, sub_c). A doesn't index sub_n.
    A_stream = streamify(
        A_buf,
        stride=(n_col, 0, 1),
        out_shape_tiled=(n_row, n_col, n_col),
    )
    # Combined: (1, M_tiles, N_tiles, K_tiles, n_row, n_col, n_col)
    # Order:    (m_l,  n_l,    k_l,     sub_r, sub_n, sub_c)

    # ------------------------------------------------------------------
    # B: load big tiles in (m_l, n_l, k_l) order with M broadcast.
    #    Each big tile of B is fetched M_tiles times (16x amp).
    # ------------------------------------------------------------------
    B_load = offchip_load(
        tensors["B"],
        stride=(0, 1, N_tiles),
        out_shape_tiled=(M_tiles, N_tiles, K_tiles),
        tile_row=big, tile_col=big,
    )
    B_buf = bufferize(downtile(B_load, sub, sub), rank=2)
    # Per-buffer = one B big tile, (sub_kk, sub_nn) sub-tile grid (256 KB).
    # Inner stream: (sub_r_bcast, sub_n, sub_c). B doesn't index sub_r.
    B_stream = streamify(
        B_buf,
        stride=(0, 1, n_col),
        out_shape_tiled=(n_row, n_col, n_col),
    )
    # Combined: matches A's 7-dim shape and order.

    # ------------------------------------------------------------------
    # Stage 1: per-big-tile matmul + reduce sub_c (inner K).
    # ------------------------------------------------------------------
    C_partial = binary_map_accum(A_stream, B_stream, rank=1)
    # Stream: (1, M_tiles, N_tiles, K_tiles, n_row, n_col), tile (sub, sub).

    # ------------------------------------------------------------------
    # Stage 2: reassemble big tile in-stream via accum_retile, then reduce
    # k_l at big-tile granularity. This avoids the 4 MB sub-tile reorder
    # buffer the bufferize(rank=3) version needed — accum_retile is a
    # streaming op whose steady-state cost is just `chunk * sub_tile`.
    # No output reorder needed: offchip_store of (big,big) tiles uses the
    # (penult*tile_row, last*tile_col) convention to produce (M, N).
    # ------------------------------------------------------------------
    C_col = accum_retile_col(C_partial, rank=1)
    # Absorbs n_col into tile_c: tile (sub, big),
    # stream (1, M_tiles, N_tiles, K_tiles, n_row).
    C_full = accum_retile_row(C_col, rank=1)
    # Absorbs n_row into tile_r: tile (big, big),
    # stream (1, M_tiles, N_tiles, K_tiles).
    C_red = accum_add(C_full, rank=1)
    # Reduces k_l in a (big, big) persistent accumulator (~256 KB).
    # Stream: (1, M_tiles, N_tiles), tile (big, big).

    return offchip_store(C_red)
