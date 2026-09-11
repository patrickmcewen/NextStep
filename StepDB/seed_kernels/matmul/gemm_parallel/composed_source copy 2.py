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
    """Independent-parallelism GEMM with off-chip big-tile loads + on-chip
    downtiling to the compute tile size.

    Each side does a single bufferize+streamify whose stride pattern both
    (a) permutes (m_l, k_l, sub_r, sub_c) -> (m_l, sub_r, k_l, sub_c) so
    that flattening recovers the natural M_small/K_small layout, and
    (b) inserts the cross-axis broadcast (N for A, M for B). Three flattens
    collapse the resulting 7-dim stream into (M_small, N_small, K_small)
    and `parallelize` slices the outer M_small dim for par_factor consumers.
    The loop body is just matmul + K-reduce.
    """
    big = 256
    sub = 16
    n_row = n_col = big // sub          # 16

    M_tiles = dims["M"] // big           # 16
    K_tiles = dims["K"] // big           # 16
    N_tiles = dims["N"] // big           # 16
    M_small = M_tiles * n_row            # 256
    K_small = K_tiles * n_col            # 256
    N_small = N_tiles * n_col            # 256

    par_factor = dims.get("par_factor", 8)
    assert par_factor >= 2, "par_factor must be at least 2"
    assert M_small % par_factor == 0, (
        f"par_factor {par_factor} must divide M_small={M_small}"
    )

    # Constants for the strided streamify below. A's per-M-row buffer is
    # laid out as (K_tiles, n_row, n_col), so buf_linear(k_l, sub_r, sub_c)
    # = k_l*sub_per_k + sub_r*n_col + sub_c. B's full buffer prepends
    # k_l*N_per_k + n_l*sub_per_k.
    sub_per_k = n_row * n_col            # tiles per (m_l, k_l) cell
    N_per_k = N_tiles * sub_per_k        # B: tiles per k_l row

    # ------------------------------------------------------------------
    # A: out stream (M_small, N_small, K_small) with N broadcast.
    #    Bufferize(rank=3) folds (K_tiles, n_row, n_col) so each on-chip
    #    buffer is one M-row of A's sub-tile grid (4 MB) instead of the
    #    full M*K tensor (64 MB). M_tiles moves to the outer stream; the
    #    inner 5-dim streamify still emits in
    #    (sub_r, n_l, sub_n, k_l, sub_c) order, so the combined 7-dim
    #    stream is identical to the rank=4 version.
    # ------------------------------------------------------------------
    A_load = offchip_load(
        tensors["A"], (K_tiles, 1), (M_tiles, K_tiles), big, big,
    )
    A_buf = bufferize(downtile(A_load, sub, sub), rank=3)
    A_stream = streamify(
        A_buf,
        stride=(n_col, 0, 0, sub_per_k, 1),
        out_shape_tiled=(n_row, N_tiles, n_col, K_tiles, n_col),
    )
    A_stream = flatten(A_stream, min_rank=0, max_rank=1)   # (K_tiles, n_col) -> K_small
    A_stream = flatten(A_stream, min_rank=1, max_rank=2)   # (N_tiles, n_col) -> N_small
    A_stream = flatten(A_stream, min_rank=2, max_rank=4)   # (1, M_tiles, n_row) -> M_small
    # stream (M_small, N_small, K_small), tile (sub, sub)

    # ------------------------------------------------------------------
    # B: out stream (M_small, N_small, K_small) with M broadcast.
    #    stride positions: (m_l, sub_r, n_l, sub_n, k_l, sub_k)
    # ------------------------------------------------------------------
    B_load = offchip_load(
        tensors["B"], (N_tiles, 1), (K_tiles, N_tiles), big, big,
    )
    B_buf = bufferize(downtile(B_load, sub, sub), rank=4)
    B_stream = streamify(
        B_buf,
        stride=(0, 0, sub_per_k, 1, N_per_k, n_col),
        out_shape_tiled=(M_tiles, n_row, N_tiles, n_col, K_tiles, n_col),
    )
    B_stream = flatten(B_stream, min_rank=0, max_rank=1)
    B_stream = flatten(B_stream, min_rank=1, max_rank=2)
    B_stream = flatten(B_stream, min_rank=2, max_rank=4)
    # stream (M_small, N_small, K_small), tile (sub, sub)

    A_par = parallelize(A_stream, par_factor)
    B_par = parallelize(B_stream, par_factor)

    partials = []
    for i in range(par_factor):
        C_i = binary_map_accum(A_par[i], B_par[i], rank=1)
        partials.append(C_i)

    # parallelize's strided slice on M_small is inverted by static_reassemble.
    return offchip_store(static_reassemble(partials))
