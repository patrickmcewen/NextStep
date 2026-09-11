"""Shared-mode M-axis parallelism for element_wise_add.

Both off-chip loads stay as a single LinearOffChipLoad each — `parallelize`
inserts an IR Parallelize node downstream of each load that round-robin
routes M-tile rows to `par_factor` consumers. Each consumer runs its own
binary_add; static_reassemble interleaves the outputs back into the
unparallelized tile order so offchip_store sees the original (m_tiles,
k_tiles) stream.

This is shared mode: a single upstream LinearOffChipLoad still has to feed
all par_factor consumers through the Parallelize op's round-robin dispatch.
HBM traffic is unchanged (no replicated reads).
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
    B = offchip_load(
        tensors["B"],
        stride=(k_tiles, 1),
        out_shape_tiled=(m_tiles, k_tiles),
        tile_row=tile_m, tile_col=tile_k,
    )
    # offchip_load emits a leading singleton stream dim. Absorb it into the
    # M tile-count so parallelize() slices the actual M axis (rank-indexed
    # from innermost: 0 = k_tiles, 1 = m_tiles, 2 = leading 1).
    A = flatten(A, min_rank=1, max_rank=2)
    B = flatten(B, min_rank=1, max_rank=2)

    # parallelize() returns par_factor streams round-robin-sliced along the
    # outermost stream dim (here: M). Both A and B must be sliced in lockstep
    # so the i-th consumer pairs A's i-th M-slice with B's i-th M-slice.
    A_par = parallelize(A, par_factor)
    B_par = parallelize(B, par_factor)
    partials = [binary_add(A_par[i], B_par[i]) for i in range(par_factor)]

    out = static_reassemble(partials)
    return offchip_store(out)
