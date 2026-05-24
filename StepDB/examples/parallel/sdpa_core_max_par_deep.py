"""M-axis parallelism for sdpa_core_max with **deep** per-branch buffers (3a).

Same parallelize() placement as the shallow variant, but each split stream
is buffered across its ENTIRE per-consumer footprint before any compute
starts. The deep buffer (rank=2 over a promoted 3-D stream) holds every
M-tile-row × N-tile that consumer is going to see — so the Parallelize
dispatcher only needs to walk each branch's full slice once, then the
downstream pipeline runs purely from on-chip SRAM with zero further
back-pressure from the dispatcher.

Compared to shallow:
  - SRAM cost grows from `n_tiles` to `m_tiles_per * n_tiles` tiles per
    branch — i.e. par_factor× larger total buffer footprint.
  - Pays off only when per-branch latency variance is large enough that
    the shallow (one-M-row) buffer keeps stalling. For long-context SDPA
    (large N), shallow is usually enough; for short context with
    irregular per-row reductions, deep can help.
"""


def _deep_replay(x, m_tiles_per, n_tiles):
    """Buffer the WHOLE per-consumer stream then replay identically.

    Requires the input stream to have shape (m_tiles_per, n_tiles); we
    promote_outer to lift it to (1, m_tiles_per, n_tiles), bufferize over the
    two inner dims into a (m_tiles_per, n_tiles) buffer, then streamify with
    row-major stride to re-emit it. A trailing flatten drops the singleton
    so the per-branch body sees the same (m_tiles_per, n_tiles) stream as
    the shallow version.
    """
    y = promote_outer(x)
    buf = bufferize(y, rank=2)
    y = streamify(buf, stride=(n_tiles, 1), out_shape_tiled=(m_tiles_per, n_tiles))
    return flatten(y, min_rank=1, max_rank=2)


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

    Q_rep = flatten(Q_rep, min_rank=1, max_rank=2)
    K = flatten(K, min_rank=1, max_rank=2)
    V = flatten(V, min_rank=1, max_rank=2)

    Q_par = parallelize(Q_rep, par_factor)
    K_par = parallelize(K, par_factor)
    V_par = parallelize(V, par_factor)

    partials = []
    for i in range(par_factor):
        Q_i = _deep_replay(Q_par[i], m_tiles_per, n_tiles)
        K_i = _deep_replay(K_par[i], m_tiles_per, n_tiles)
        V_i = _deep_replay(V_par[i], m_tiles_per, n_tiles)

        qkt = binary_matmul(Q_i, K_i, weight_transposed=True)
        qkt_a, qkt_b = broadcast(qkt, 2)

        qkt_split = retile_streamify(qkt_a, chunk=1, split_row=False)
        row_max = accum_max(qkt_split, rank=1)
        neg_row_max = unary_mul_imm(row_max, constant=-1.0)
        neg_row_max_rep = repeat_static(neg_row_max, factor=n_tiles)

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
