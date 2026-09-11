"""Shared-mode M-axis parallelism for vector_reduce_sum.

The reduction is along K-tiles, which is *inside* each consumer's tile-stream
— so M-axis parallelism leaves the math untouched. Each consumer reduces its
own M-tile rows independently, and static_reassemble interleaves the per-row
reduced tiles back into the original M order.

Stream-shape bookkeeping (offchip_load adds a leading singleton on the way in;
accum_add drops the K-tiles dim on the way out, so we have to add the singleton
back before offchip_store):

    offchip_load:                  (1,        m_tiles,     k_tiles)
    flatten(min=1, max=2):         (m_tiles,  k_tiles)
    parallelize(par_factor):  [each (m_tiles/par_factor, k_tiles)]
    accum_add(rank=1):        [each (m_tiles/par_factor,)]
    static_reassemble:             (m_tiles,)
    promote_outer:                 (1, m_tiles)
    offchip_store: ok (>=2 stream dims).
"""


def tiled_reference(dims, tensors):
    M, K = dims["M"], dims["K"]
    tile_m, tile_k = dims["tile_m"], dims["tile_k"]
    par_factor = dims["par_factor"]
    m_tiles, k_tiles = M // tile_m, K // tile_k
    assert m_tiles % par_factor == 0, (
        f"m_tiles={m_tiles} not divisible by par_factor={par_factor}"
    )

    A = offchip_load(
        tensors["A"],
        stride=(k_tiles, 1),
        out_shape_tiled=(m_tiles, k_tiles),
        tile_row=tile_m, tile_col=tile_k,
    )
    A = flatten(A, min_rank=1, max_rank=2)

    A_par = parallelize(A, par_factor)
    partials = [accum_add(A_par[i], rank=1) for i in range(par_factor)]

    merged = static_reassemble(partials)
    merged = promote_outer(merged)
    return offchip_store(merged)
