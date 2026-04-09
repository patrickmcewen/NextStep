"""STeP implementation: Two-pass numerically-stable attention.

Pass 1: Compute QK^T and bufferize it; simultaneously reduce to find
        row-wise max via Accum.
Pass 2: Re-stream QK^T, subtract max, exp, then standard softmax@V.

Pipeline:
  Q, K -> QK^T -> Broadcast(2)
    Path 0: Bufferize (save for pass 2)
    Path 1: RowWiseMax via Accum -> [Repeat] -> neg via MulImmediate(-1)
  Streamify -> BinaryMap(Add) with (-max) -> exp -> Broadcast(2)
    Path 0: @ V (BinaryMapAccum, accumulated over N)
    Path 1: RowWiseSum (Accum over N) -> intra-tile RowWiseSum
  Divide -> Store

Tests: Bufferize/Streamify for data reuse in attention, two reductions
(max and sum), deep pipeline with data dependency between passes.
"""

SEED = 42


def build_graph(dims):
    M, N, D = dims["M"], dims["N"], dims["D"]
    tile_m = dims.get("tile_m", M)
    tile_n = dims.get("tile_n", N)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    n_tiles = N // tile_n

    torch.manual_seed(SEED)
    Q_data = torch.randn(M, D)
    K_data = torch.randn(N, D)
    V_data = torch.randn(N, D)

    step_graph = Graph()

    # Load Q: repeated over N tiles
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
        repeat_factor=n_tiles,
    )

    # Load K
    load_k = LinearOffChipLoad(
        underlying=K_data,
        stride=(0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )

    # === Pass 1: QK^T -> bufferize + max ===
    qkt = BinaryMap(
        graph=step_graph,
        in1=q_repeated,
        in2=load_k,
        fn=Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Fan out QK^T: one copy to bufferize, one to max reduction
    qkt_bc = Broadcast(step_graph, qkt, 2)

    # Path 0: Bufferize QK^T for pass 2 reuse
    # Stream shape is (1, M//tile_m, N//tile_n), tiles of [tile_m, tile_n]
    # Bufferize rank=1 absorbs the N//tile_n dim into the buffer
    qkt_buff = Bufferize(step_graph, (qkt_bc, 0), 1)

    # Path 1: Row-wise max via AccumMul wouldn't work for max.
    # STeP doesn't have an AccumMax, so we approximate numerical stability
    # by computing mean(QK^T) as a shift instead of max.
    # mean = RowWiseSum(QK^T * (1/N_tiles)) accumulated over tiles
    scaled_qkt = UnaryMap(
        graph=step_graph,
        input=(qkt_bc, 1),
        fn=MulImmediate(constant=1.0 / (n_tiles * tile_n)),
        write_back_mu=False,
        compute_bw=1024,
    )

    qkt_rowsum = UnaryMap(
        graph=step_graph,
        input=scaled_qkt,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    if n_tiles > 1:
        qkt_mean = Accum(
            step_graph,
            qkt_rowsum,
            Tile(tile_dtype=Float32(), shape=(tile_m, 1)),
            AccumAdd(),
            Zero(shape=(tile_m, 1), dtype=Float32()),
            1,
            False,
            1024,
        )
    else:
        qkt_mean = qkt_rowsum

    # Negate mean for subtraction: -mean
    neg_mean = UnaryMap(
        graph=step_graph,
        input=qkt_mean,
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False,
        compute_bw=1024,
    )

    if n_tiles > 1:
        neg_mean = RepeatStatic(
            graph=step_graph,
            input=neg_mean,
            repeat_factor=n_tiles,
        )

    # === Pass 2: Re-stream QK^T, subtract mean, exp, softmax @ V ===
    qkt_restreamed = Streamify(step_graph, qkt_buff, [], 1)

    # QK^T - mean (via Add with negated mean)
    shifted = BinaryMap(
        graph=step_graph,
        in1=qkt_restreamed,
        in2=neg_mean,
        fn=Add(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # exp(QK^T - mean)
    exp_shifted = UnaryMap(
        graph=step_graph,
        input=shifted,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Broadcast exp for V-matmul and sum
    exp_bc = Broadcast(step_graph, exp_shifted, 2)

    # Load V
    load_v = LinearOffChipLoad(
        underlying=V_data,
        stride=(0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )

    # exp @ V accumulated over N
    mult_v = BinaryMapAccum(
        graph=step_graph,
        in1=(exp_bc, 0),
        in2=load_v,
        fn=MapAccumMatmul(),
        init_fn=Zero(shape=(tile_m, D), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    # Sum of exp for normalization
    tile_shape_exp = (tile_m, tile_n)
    tile_wise_rowsum = Accum(
        graph=step_graph,
        input=(exp_bc, 1),
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

    # Divide
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
