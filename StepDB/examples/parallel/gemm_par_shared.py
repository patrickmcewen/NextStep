"""Shared-mode M-axis parallelism for GEMM.

Contrast this with the few-shot's M-axis *independent* example: same kernel,
same axis, different mechanism.

  - Independent mode emits par_factor separate LinearOffChipLoads for A
    (one per consumer), each reading its own M-slice. B is replicated by
    every consumer's offchip_load, multiplying HBM B-traffic by par_factor.
  - Shared mode keeps a single LinearOffChipLoad for A and inserts an IR
    Parallelize op downstream that round-robin dispatches M-tile rows to
    par_factor consumers. HBM traffic is unchanged (no replicated loads),
    but the single source's token rate is now a ceiling on aggregate
    consumer throughput.

Pick shared when consumer-side compute is the bottleneck and the source
has slack; pick independent when the source itself is the bottleneck (and
the broadcast operand is small enough to tolerate par_factor× HBM cost).
"""


def tiled_reference(dims, tensors):
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m, tile_k, tile_n = dims["tile_m"], dims["tile_k"], dims["tile_n"]
    par_factor = dims["par_factor"]
    m_tiles, n_tiles, k_tiles = M // tile_m, N // tile_n, K // tile_k
    assert m_tiles % par_factor == 0, (
        f"m_tiles={m_tiles} not divisible by par_factor={par_factor}"
    )

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
    # Absorb the leading singleton from offchip_load into the m_tiles dim so
    # the outermost stream dim is M (what we want to parallelize on).
    A_flat = flatten(A, min_rank=2, max_rank=3)
    B_flat = flatten(B, min_rank=2, max_rank=3)

    # SHARED parallelism: single A/B load, single IR Parallelize node fans
    # M-tile rows out to par_factor consumers. Both A and B are sliced in
    # lockstep so consumer i pairs A's i-th M-slice with B's i-th M-slice
    # (B's M stride is 0, so all par_factor B-slices are byte-equal copies
    # of the same underlying — the Parallelize op still routes them).
    A_par = parallelize(A_flat, par_factor)
    B_par = parallelize(B_flat, par_factor)
    partials = [
        binary_map_accum(A_par[i], B_par[i], rank=1)
        for i in range(par_factor)
    ]

    out = static_reassemble(partials)
    return offchip_store(out)
