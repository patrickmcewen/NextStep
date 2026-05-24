"""Independent M-axis parallelism for GEMM (HBM-side parallelism).

The single LinearOffChipLoad for A is replaced by par_factor separate
LinearOffChipLoads — one per consumer — each reading its own M-tile rows
via start_tile_idx and a par_factor× stride. B is replicated by every
consumer's offchip_load (its M-stride is already 0), so HBM B-traffic
grows by par_factor; the trade-off is full DRAM-channel parallelism on A.
"""


def tiled_reference(dims, tensors):
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m, tile_k, tile_n = dims["tile_m"], dims["tile_k"], dims["tile_n"]
    par_factor = dims["par_factor"]
    m_tiles, n_tiles, k_tiles = M // tile_m, N // tile_n, K // tile_k
    assert m_tiles % par_factor == 0, (
        f"m_tiles={m_tiles} not divisible by par_factor={par_factor}"
    )
    m_tiles_per = m_tiles // par_factor

    partials = []
    for i in range(par_factor):
        # Consumer i sees M-tile rows {i, par_factor+i, 2*par_factor+i, ...}.
        # start_tile_idx=i*k_tiles shifts A's base by i M-tile rows; the
        # par_factor*k_tiles stride along the M axis then steps by par_factor
        # M-rows between consumed tiles. B is loaded in full per consumer.
        A_i = offchip_load(
            tensors["A"],
            stride=(par_factor * k_tiles, 0, 1),
            out_shape_tiled=(m_tiles_per, n_tiles, k_tiles),
            tile_row=tile_m, tile_col=tile_k,
            start_tile_idx=i * k_tiles,
        )
        B_i = offchip_load(
            tensors["B"],
            stride=(0, 1, n_tiles),
            out_shape_tiled=(m_tiles_per, n_tiles, k_tiles),
            tile_row=tile_k, tile_col=tile_n,
        )
        A_flat = flatten(A_i, min_rank=2, max_rank=3)
        B_flat = flatten(B_i, min_rank=2, max_rank=3)
        partials.append(binary_map_accum(A_flat, B_flat, rank=1))

    return offchip_store(static_reassemble(partials))
