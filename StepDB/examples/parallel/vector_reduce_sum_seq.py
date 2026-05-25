"""Sequential baseline for vector_reduce_sum: out[m, :] = sum over k-tiles of A[m, :].

Loads (M_tiles, K_tiles) tiles of A and accumulates along the K-tile axis with
accum_add(rank=1). The output stream is (M_tiles,) with tile (tile_m, tile_k).
"""


def tiled_reference(dims, tensors):
    M, K = dims["M"], dims["K"]
    tile_m, tile_k = dims["tile_m"], dims["tile_k"]
    m_tiles, k_tiles = M // tile_m, K // tile_k

    A = offchip_load(
        tensors["A"],
        stride=(k_tiles, 1),
        out_shape_tiled=(m_tiles, k_tiles),
        tile_row=tile_m, tile_col=tile_k,
    )
    reduced = accum_add(A, rank=1)
    return offchip_store(reduced)
