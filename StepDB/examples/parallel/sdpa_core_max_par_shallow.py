"""M-axis parallelism for sdpa_core_max with **shallow** per-branch buffers (3b).

Each load is split on the M-tile axis via `parallelize`, then each split
stream passes through a `bufferize(rank=1) + streamify` that stages exactly
one M-tile-row of tiles (the innermost N-tile dim). This decouples the
Parallelize dispatcher from each per-consumer pipeline: even if consumer 0
hasn't drained its current M-row, the dispatcher can put consumer 1's next
M-row tiles into consumer 1's shallow buffer and keep moving.

Why shallow (rank=1) rather than deep (rank=2)? n_tiles tiles per M-row is
usually enough to absorb the row-max -> subtract back-pressure that lives
INSIDE the per-consumer body. We pay one tile-row's worth of SRAM per
branch — par_factor * n_tiles * tile_m * tile_n bytes total — instead of
the whole M-slice. See sdpa_core_max_par_deep.py for the deeper buffer.
"""


def _identity_replay(x, n_tiles):
    """bufferize the innermost N stream dim, then streamify it back identically."""
    buf = bufferize(x, rank=1)
    return streamify(buf, stride=(1,), out_shape_tiled=(n_tiles,))


def tiled_reference(dims, tensors):
    M, N, D = dims["M"], dims["N"], dims["D"]
    tile_m, tile_n = dims["tile_m"], dims["tile_n"]
    par_factor = dims["par_factor"]
    m_tiles = M // tile_m
    n_tiles = N // tile_n
    assert m_tiles % par_factor == 0, (
        f"m_tiles={m_tiles} not divisible by par_factor={par_factor}"
    )
    m_tiles_per = m_tiles // par_factor

    Q = offchip_load(
        tensors["Q"],
        stride=(1,),
        out_shape_tiled=(m_tiles,),
        tile_row=tile_m, tile_col=D,
    )
    K = offchip_load(
        tensors["K"],
        stride=(0, 1),
        out_shape_tiled=(m_tiles, n_tiles),
        tile_row=tile_n, tile_col=D,
    )
    V = offchip_load(
        tensors["V"],
        stride=(0, 1),
        out_shape_tiled=(m_tiles, n_tiles),
        tile_row=tile_n, tile_col=D,
    )
    Q_rep = repeat_static(Q, factor=n_tiles)

    # Absorb offchip_load's leading singleton so the M-tile axis is outermost
    # for parallelize(). Stream rank is 3 (leading 1, m_tiles, n_tiles), so
    # merge ranks 1 and 2.
    Q_rep = flatten(Q_rep, min_rank=1, max_rank=2)   # (m_tiles, n_tiles)
    K = flatten(K, min_rank=1, max_rank=2)
    V = flatten(V, min_rank=1, max_rank=2)

    Q_par = parallelize(Q_rep, par_factor)
    K_par = parallelize(K, par_factor)
    V_par = parallelize(V, par_factor)

    partials = []
    for i in range(par_factor):
        # Shallow buffer per branch: rank=1 absorbs the innermost N-tile dim
        # into a Buffer; streamify replays identically. The buffer's depth is
        # n_tiles tiles — exactly one M-tile-row worth.
        Q_i = _identity_replay(Q_par[i], n_tiles)
        K_i = _identity_replay(K_par[i], n_tiles)
        V_i = _identity_replay(V_par[i], n_tiles)

        qkt = binary_matmul(Q_i, K_i, weight_transposed=True)
        qkt_a, qkt_b = broadcast(qkt, 2)

        qkt_split = retile_streamify(qkt_a, chunk=1, split_row=False)
        row_max = accum_max(qkt_split, rank=1)
        neg_row_max = unary_mul_imm(row_max, constant=-1.0)
        neg_row_max_rep = repeat_static(neg_row_max, factor=n_tiles)

        # The IR-side bufferize between qkt_b and the subtract is the same
        # row-max-vs-subtract decoupling buffer that the sequential code uses;
        # it's unrelated to the parallelize-side buffers above.
        qkt_buf = bufferize(qkt_b, rank=1)
        qkt_replay = streamify(qkt_buf, stride=(1,), out_shape_tiled=(n_tiles,))

        qkt_shifted = binary_add(qkt_replay, neg_row_max_rep)
        exp_qkt = unary_exp(qkt_shifted)
        exp_a, exp_b = broadcast(exp_qkt, 2)

        numerator = binary_map_accum(exp_a, V_i, rank=1)
        tile_rowsum = accum_add(exp_b, rank=1)
        denom = unary_rowwise_sum(tile_rowsum)
        partials.append(binary_div(numerator, denom))

    merged = static_reassemble(partials)
    merged = promote_outer(merged)
    return offchip_store(merged)
