"""STeP implementation: SDPA core compute with safe (max-subtracted) softmax.

Mirrors `seed_kernels/attention/sdpa_core/step_impl.py` but does the standard
numerical-stabilization step:
  Q @ K^T  ->  scores - row_max(scores)  ->  exp  ->  @V (accumulated)  ->  / sum(exp)

There is no `RowWiseMax` map_fn in the STeP IR, only a streaming `accum_fn.Max`
that reduces tiles element-wise across an outer stream dim. To compute the
full per-row max we use `RetileStreamify(chunk=1, split_row=False)` to
re-expose each tile's tile_n columns by multiplying the innermost stream dim,
then a single `Accum(Max, accum_rank=1)` reduces all (N//tile_n)*tile_n = N
score elements per row in one pass:
  qkt: (M//tile_m, N//tile_n) tiles [tile_m, tile_n]
       --RetileStreamify(chunk=1, split_row=False)-->
       (M//tile_m, N//tile_n * tile_n) tiles [tile_m, 1]
       --Accum(Max, accum_rank=1)-->
       (M//tile_m,) tiles [tile_m, 1]

The original scores are kept around for the subtract via Bufferize/Streamify
(rank=1 over the N stream dim), so the same Q@K^T result feeds both the max
reduction and the subtract — no recomputation.

After subtract+Exp the rest of the pipeline is identical to sdpa_core:
exp @ V (BinaryMapAccum), tile-wise rowsum (Accum + RowWiseSum), divide.

Tiling strategy:
  Q: (M // tile_m, N // tile_n) tiles of [tile_m, D]       (repeated over N)
  K: (M // tile_m, N // tile_n) tiles of [tile_n, D]
  V: (M // tile_m, N // tile_n) tiles of [tile_n, D]
  QK^T: (M // tile_m, N // tile_n) tiles of [tile_m, tile_n]
  row_max: (M // tile_m,) tiles of [tile_m, 1]
  exp@V:  (M // tile_m,) tiles of [tile_m, D]              (accumulated over N)
  sum_exp:(M // tile_m,) tiles of [tile_m, 1]              (accumulated + RowWiseSum)
  output: (M // tile_m,) tiles of [tile_m, D]
"""

SEED = 42


def build_graph(dims):
    M, N, D = dims["M"], dims["N"], dims["D"]
    tile_m = dims.get("tile_m", M)
    tile_n = dims.get("tile_n", N)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    torch.manual_seed(SEED)
    Q_data = torch.randn(M, D)
    K_data = torch.randn(N, D)
    V_data = torch.randn(N, D)

    step_graph = Graph()

    # --- Load Q, K, V ---
    load_q = LinearOffChipLoad(
        underlying=Q_data,
        stride=(1,),
        out_shape_tiled=(M // tile_m,),
        tile_row=tile_m,
        tile_col=D,
        par_dispatch=4,
    )
    q_repeated = RepeatStatic(
        graph=step_graph,
        input=load_q,
        repeat_factor=N // tile_n,
    )
    # q_repeated: (M // tile_m, N // tile_n) tiles of [tile_m, D]

    load_k = LinearOffChipLoad(
        underlying=K_data,
        stride=(0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )

    load_v = LinearOffChipLoad(
        underlying=V_data,
        stride=(0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )

    # --- Stage 8: QK^T ---
    qkt = BinaryMap(
        graph=step_graph,
        in1=q_repeated,
        in2=load_k,
        fn=Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=1024,
    )
    # qkt: (M // tile_m, N // tile_n) tiles of [tile_m, tile_n]

    # Fan out qkt: one copy feeds the row-max reduction, one is bufferized for
    # later subtract.
    qkt_branches = Broadcast(step_graph, qkt, 2)

    # --- Branch A: row_max(qkt) per M-tile ---
    # Split each [tile_m, tile_n] tile column-wise into tile_n separate
    # [tile_m, 1] tiles, multiplying the innermost stream dim:
    # (M//tile_m, N//tile_n) [tile_m, tile_n]
    #     -> (M//tile_m, N//tile_n * tile_n) [tile_m, 1]
    qkt_split = RetileStreamify(
        graph=step_graph,
        input=(qkt_branches, 0),
        chunk=1,
        split_row=False,
    )
    # Reduce across the (now N-element-wide) innermost stream dim element-wise.
    # (M//tile_m, N) [tile_m, 1]  ->  (M//tile_m,) [tile_m, 1]
    row_max = Accum(
        graph=step_graph,
        input=qkt_split,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_m, 1)),
        fn=accum_fn.Max(),
        init_fn=Empty(shape=(tile_m, 1), dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )
    # Negate so the subtract becomes an Add (no Sub MapFn exists).
    neg_row_max = UnaryMap(
        graph=step_graph,
        input=row_max,
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False,
        compute_bw=1024,
    )
    # Replicate across N tiles to align with the qkt stream shape.
    # (M//tile_m,) [tile_m, 1]  ->  (M//tile_m, N//tile_n) [tile_m, 1]
    neg_row_max_repeated = RepeatStatic(
        graph=step_graph,
        input=neg_row_max,
        repeat_factor=N // tile_n,
    )

    # --- Branch B: bufferize qkt for replay alongside row_max ---
    # Bufferize collapses the innermost stream dim (N) into a buffer per
    # M-tile; Streamify with empty repeat_factor replays it identity. This
    # delays branch B by exactly the time branch A takes to produce row_max.
    qkt_buff = Bufferize(
        graph=step_graph,
        input=(qkt_branches, 1),
        rank=1,
    )
    qkt_replayed = Streamify(
        graph=step_graph,
        input=qkt_buff,
        repeat_factor=[],
        rank=1,
    )

    # --- Stage 8.5: scores - row_max  (Add with broadcast on cols) ---
    qkt_shifted = BinaryMap(
        graph=step_graph,
        in1=qkt_replayed,
        in2=neg_row_max_repeated,
        fn=Add(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # --- Stage 9: exp(QK^T - row_max) ---
    exp_qkt = UnaryMap(
        graph=step_graph,
        input=qkt_shifted,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Broadcast exp for two consumers: V-matmul and row-sum
    exp_broadcast = Broadcast(step_graph, exp_qkt, 2)

    # --- Stage 10: exp @ V, accumulated over N ---
    mult_v = BinaryMapAccum(
        graph=step_graph,
        in1=(exp_broadcast, 0),
        in2=load_v,
        fn=MapAccumMatmul(),
        init_fn=Zero(shape=(tile_m, D), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    # --- Stage 11: Softmax normalization ---
    tile_shape_exp = (tile_m, tile_n)
    tile_wise_rowsum = Accum(
        graph=step_graph,
        input=(exp_broadcast, 1),
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=tile_shape_exp),
        fn=AccumAdd(),
        init_fn=Zero(shape=tile_shape_exp, dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    intra_tile_rowsum = UnaryMap(
        graph=step_graph,
        input=tile_wise_rowsum,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    softmax_out = BinaryMap(
        graph=step_graph,
        in1=mult_v,
        in2=intra_tile_rowsum,
        fn=Div(),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=softmax_out,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
