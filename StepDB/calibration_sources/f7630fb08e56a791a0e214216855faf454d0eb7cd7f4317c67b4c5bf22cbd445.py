"""STeP implementation: Scaled Dot-Product Attention (core compute).

Implements the attention compute pipeline from flashattn.py stages 8-11:
  Q @ K^T  ->  exp  ->  @V (accumulated)  ->  / sum(exp)

Tiling strategy:
  Q: (M // tile_m, N // tile_n) tiles of [tile_m, D]       (repeated over N)
  K: (M // tile_m, N // tile_n) tiles of [tile_n, D]       (repeated over M)
  V: (M // tile_m, N // tile_n) tiles of [tile_n, D]       (repeated over M)
  QK^T:   (M // tile_m, N // tile_n) tiles of [tile_m, tile_n]
  exp@V:  (M // tile_m,) tiles of [tile_m, D]              (accumulated over N)
  sum_exp:(M // tile_m,) tiles of [tile_m, 1]              (accumulated + RowWiseSum)
  output: (M // tile_m,) tiles of [tile_m, D]

Extracted from step_tl/end_to_end/attention/flashattn.py and
step_tl/dynamic_par/flashattn.py.
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

    # --- Load Q: (M // tile_m,) tiles of [tile_m, D], then repeat for N tiles ---
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

    # --- Load K: (M // tile_m, N // tile_n) tiles of [tile_n, D] ---
    # stride=(0, 1): outer dim (M) doesn't advance in K; inner dim (N) does
    load_k = LinearOffChipLoad(
        underlying=K_data,
        stride=(0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )

    # --- Stage 8: QK^T ---
    # [tile_m, D] @ [tile_n, D]^T -> [tile_m, tile_n]
    qkt = BinaryMap(
        graph=step_graph,
        in1=q_repeated,
        in2=load_k,
        fn=Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=1024,
    )

    # --- Stage 9: exp(QK^T) ---
    exp_qkt = UnaryMap(
        graph=step_graph,
        input=qkt,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Broadcast exp for two consumers: V-matmul and row-sum
    exp_broadcast = Broadcast(step_graph, exp_qkt, 2)

    # --- Load V: (M // tile_m, N // tile_n) tiles of [tile_n, D] ---
    load_v = LinearOffChipLoad(
        underlying=V_data,
        stride=(0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )

    # --- Stage 10: exp(QK^T) @ V, accumulated over N ---
    # [tile_m, tile_n] @ [tile_n, D] -> accumulate -> [tile_m, D]
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
    # Accumulate exp tiles over N: (M // tile_m, N // tile_n) -> (M // tile_m,)
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

    # Intra-tile row sum: [tile_m, tile_n] -> [tile_m, 1]
    intra_tile_rowsum = UnaryMap(
        graph=step_graph,
        input=tile_wise_rowsum,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Divide: context / sum_exp
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
