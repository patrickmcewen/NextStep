"""Sequential baseline for chained_unary: rsqrt(silu(exp(x^2))).

A four-stage unary pipeline. Each stage operates element-wise on its
predecessor's stream and emits the same stream shape — so the lowered graph
is one Load -> 4 UnaryMaps -> Store dataflow chain.
"""


def tiled_reference(dims, tensors):
    M, K = dims["M"], dims["K"]
    tile_m, tile_k = dims["tile_m"], dims["tile_k"]
    m_tiles, k_tiles = M // tile_m, K // tile_k

    x = offchip_load(
        tensors["input"],
        stride=(k_tiles, 1),
        out_shape_tiled=(m_tiles, k_tiles),
        tile_row=tile_m, tile_col=tile_k,
    )
    x = unary_square(x)
    x = unary_exp(x)
    x = unary_silu(x)
    x = unary_rsqrt(x)
    return offchip_store(x)
