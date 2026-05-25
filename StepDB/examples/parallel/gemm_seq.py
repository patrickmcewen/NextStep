"""Sequential baseline for GEMM: out = A @ B.

Mirrors the few-shot's "before" code so the new shared-mode example can be
read as a drop-in replacement for the M-axis independent transform.
"""


def tiled_reference(dims, tensors):
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m, tile_k, tile_n = dims["tile_m"], dims["tile_k"], dims["tile_n"]
    m_tiles, n_tiles, k_tiles = M // tile_m, N // tile_n, K // tile_k

    A = offchip_load(
        tensors["A"],
        stride=(k_tiles, 0, 1),
        out_shape_tiled=(m_tiles, n_tiles, k_tiles),
        tile_row=tile_m, tile_col=tile_k,
    )
    B = offchip_load(
        tensors["B"],
        stride=(0, 1, n_tiles),
        out_shape_tiled=(m_tiles, n_tiles, k_tiles),
        tile_row=tile_k, tile_col=tile_n,
    )
    A_flat = flatten(A, min_rank=2, max_rank=3)
    B_flat = flatten(B, min_rank=2, max_rank=3)
    out = binary_map_accum(A_flat, B_flat, rank=1)
    return offchip_store(out)
