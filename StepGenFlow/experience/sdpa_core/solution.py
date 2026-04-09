def build_graph(dims):
    # --------------------------------------------------------------
    # 1.  Create the same random inputs as the reference model.
    # --------------------------------------------------------------
    torch.manual_seed(42)                     # match the reference SEED
    M, N, D = dims["M"], dims["N"], dims["D"]

    Q = torch.randn(M, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)

    # --------------------------------------------------------------
    # 2.  Initialise the graph and load the three matrices.
    # --------------------------------------------------------------
    g = Graph()

    # Q : [M, D] – load as a single tile
    q = LinearOffChipLoad(
        underlying=Q,
        stride=(1, 1),                 # (C//tile_col, 1) – whole matrix is one tile
        out_shape_tiled=(1, 1),        # (M//M, D//D)
        tile_row=M,
        tile_col=D,
        par_dispatch=1,
        transposed=False,
    )
    g.add_node(q)

    # K : [N, D] – load as a single tile
    k = LinearOffChipLoad(
        underlying=K,
        stride=(1, 1),
        out_shape_tiled=(1, 1),
        tile_row=N,
        tile_col=D,
        par_dispatch=1,
        transposed=False,
    )
    g.add_node(k)

    # V : [N, D] – load as a single tile
    v = LinearOffChipLoad(
        underlying=V,
        stride=(1, 1),
        out_shape_tiled=(1, 1),
        tile_row=N,
        tile_col=D,
        par_dispatch=1,
        transposed=False,
    )
    g.add_node(v)

    # --------------------------------------------------------------
    # 3.  Q @ Kᵀ → scores   (BinaryMapAccum, reduction over inner D)
    # --------------------------------------------------------------
    scores = BinaryMapAccum(
        g,
        in1=q,
        in2=k,
        fn=map_accum_fn.Matmul(weight_transposed=True),   # Kᵀ
        init_fn=init_fn.Zero(shape=(M, N), dtype=Float32()),
        rank=1,                     # reduce over the shared D dimension
        write_back_mu=False,
        compute_bw=1,
    )
    g.add_node(scores)

    # --------------------------------------------------------------
    # 4.  exp(scores)
    # --------------------------------------------------------------
    exp_scores = UnaryMap(
        g,
        input=scores,
        fn=map_fn.Exp(),
        write_back_mu=False,
        compute_bw=1,
    )
    g.add_node(exp_scores)

    # --------------------------------------------------------------
    # 5.  exp_scores @ V → context   (BinaryMapAccum)
    # --------------------------------------------------------------
    context = BinaryMapAccum(
        g,
        in1=exp_scores,
        in2=v,
        fn=map_accum_fn.Matmul(weight_transposed=False),   # V is not transposed
        init_fn=init_fn.Zero(shape=(M, D), dtype=Float32()),
        rank=1,                     # sum over the inner N dimension
        write_back_mu=False,
        compute_bw=1,
    )
    g.add_node(context)

    # --------------------------------------------------------------
    # 6.  norm = Σ_j exp(scores_{i,j})   (RowWiseSum on exp_scores)
    # --------------------------------------------------------------
    norm = UnaryMap(
        g,
        input=exp_scores,
        fn=map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=1,
    )
    g.add_node(norm)          # shape (M, 1) in tile dimensions

    # --------------------------------------------------------------
    # 7.  Divide each row of `context` by the corresponding `norm`.
    #     BinaryMap performs element‑wise division; broadcasting of the
    #     trailing singleton tile column (size 1) to D is handled by
    #     PyTorch broadcasting inside the emulator.
    # --------------------------------------------------------------
    output = BinaryMap(
        g,
        in1=context,
        in2=norm,
        fn=map_fn.Div(),
        write_back_mu=False,
        compute_bw=1,
    )
    g.add_node(output)

    # --------------------------------------------------------------
    # 8.  Store the final result.
    # --------------------------------------------------------------
    store = OffChipStore(
        g,
        input=output,
        par_dispatch=1,
        store_file_name="output",
    )
    g.add_node(store)

    # --------------------------------------------------------------
    # 9.  Finalise and return.
    # --------------------------------------------------------------
    g = infer_broadcast(g)
    return g, store