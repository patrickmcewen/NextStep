"""Shared-mode M-axis parallelism for chained_unary (a deep unary pipeline).

The chain Square -> Exp -> Silu -> Rsqrt is element-wise and per-row
independent, so we can parallelize on the M-tile axis without changing the
math. Because each consumer replicates the entire chain (4 UnaryMap ops),
shared mode amortizes the per-stage start-up latency across par_factor
independent pipelines rather than just one.

Note: parallelize() goes *before* the chain (right after the single
LinearOffChipLoad and after absorbing offchip_load's leading singleton).
That way the IR has one Parallelize node feeding par_factor replicated
chains, instead of par_factor Parallelize nodes (one per stage) — same
math, far fewer routing ops.
"""


def tiled_reference(dims, tensors):
    M, K = dims["M"], dims["K"]
    tile_m, tile_k = dims["tile_m"], dims["tile_k"]
    par_factor = dims["par_factor"]
    m_tiles, k_tiles = M // tile_m, K // tile_k
    assert m_tiles % par_factor == 0, (
        f"m_tiles={m_tiles} not divisible by par_factor={par_factor}"
    )

    x = offchip_load(
        tensors["input"],
        stride=(k_tiles, 1),
        out_shape_tiled=(m_tiles, k_tiles),
        tile_row=tile_m, tile_col=tile_k,
    )
    x = flatten(x, min_rank=1, max_rank=2)

    # Single parallelize before the chain — one IR Parallelize op fans the
    # M-tile rows out to par_factor consumers, each running the full chain.
    x_par = parallelize(x, par_factor)
    partials = []
    for i in range(par_factor):
        y = unary_square(x_par[i])
        y = unary_exp(y)
        y = unary_silu(y)
        y = unary_rsqrt(y)
        partials.append(y)

    out = static_reassemble(partials)
    return offchip_store(out)
