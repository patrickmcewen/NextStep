"""Sequential baseline for sdpa_core_max: safe (max-subtracted) softmax(Q@K^T) @ V.

Mirrors the IR step_impl exactly: Q is loaded once and repeat_static'd over N
to align with the K/V stream shape; QK^T fans out to a row-max reduction
(via retile_streamify + accum_max) and a delayed subtract path (via
bufferize + streamify identity replay) so the same scores feed both paths
with no recomputation; exp scores then fan out to the V-matmul and the
denominator's row-sum.
"""


def tiled_reference(dims, tensors):
    M, N, D = dims["M"], dims["N"], dims["D"]
    tile_m, tile_n = dims["tile_m"], dims["tile_n"]
    m_tiles = M // tile_m
    n_tiles = N // tile_n

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
    # Q has stream (1, m_tiles); repeat_static appends an inner stream dim so
    # Q's stream becomes (1, m_tiles, n_tiles) to match K's.
    Q_rep = repeat_static(Q, factor=n_tiles)

    # Q @ K^T: (tile_m, D) @ (tile_n, D)^T -> (tile_m, tile_n).
    qkt = binary_matmul(Q_rep, K, weight_transposed=True)

    qkt_a, qkt_b = broadcast(qkt, 2)

    # Row-max branch: each (tile_m, tile_n) tile is column-split into tile_n
    # tiles of (tile_m, 1); accum_max(rank=1) reduces across n_tiles*tile_n
    # entries per M-tile-row.
    qkt_split = retile_streamify(qkt_a, chunk=1, split_row=False)
    row_max = accum_max(qkt_split, rank=1)
    neg_row_max = unary_mul_imm(row_max, constant=-1.0)
    neg_row_max_rep = repeat_static(neg_row_max, factor=n_tiles)

    # Subtract branch: bufferize qkt's N stream dim then re-emit identically.
    # Without this, the subtract op would consume qkt tiles at the same rate
    # the row-max path does — but the row_max scalar isn't ready until the
    # full N-stream has been reduced. The buffer absorbs the entire N window
    # per M-tile-row so the subtract has its operand ready when row_max lands.
    qkt_buf = bufferize(qkt_b, rank=1)
    qkt_replay = streamify(qkt_buf, stride=(1,), out_shape_tiled=(n_tiles,))

    qkt_shifted = binary_add(qkt_replay, neg_row_max_rep)
    exp_qkt = unary_exp(qkt_shifted)

    exp_a, exp_b = broadcast(exp_qkt, 2)

    # Numerator: exp(scores) @ V, accumulated over N. Each tile contributes
    # (tile_m, tile_n) @ (tile_n, D) -> (tile_m, D); rank=1 sums across N.
    numerator = binary_map_accum(exp_a, V, rank=1)

    # Denominator: sum exp across N, then row-wise sum within each tile.
    tile_rowsum = accum_add(exp_b, rank=1)
    denom = unary_rowwise_sum(tile_rowsum)

    softmax_out = binary_div(numerator, denom)
    return offchip_store(softmax_out)
