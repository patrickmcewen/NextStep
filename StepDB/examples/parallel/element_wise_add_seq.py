"""Sequential baseline for element_wise_add: out = A + B.

Two off-chip loads feeding one binary_add, then store. The lowered graph
has no parallelism — single producer at every stage. Used as the "before"
side of the shared-mode parallelism example.
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
    B = offchip_load(
        tensors["B"],
        stride=(k_tiles, 1),
        out_shape_tiled=(m_tiles, k_tiles),
        tile_row=tile_m, tile_col=tile_k,
    )
    out = binary_add(A, B)
    return offchip_store(out)
