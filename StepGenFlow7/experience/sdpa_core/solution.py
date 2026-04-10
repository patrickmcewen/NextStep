def build_graph(dims: dict, tensors: dict):
    # --------------------------------------------------------------
    # 1️⃣  Dimensions & tile sizes
    # --------------------------------------------------------------
    M = dims["M"]
    N = dims["N"]
    D = dims["D"]
    tile_m = dims["tile_m"]
    tile_n = dims["tile_n"]

    M_grid = M // tile_m        # 2
    N_grid = N // tile_n        # 8

    # --------------------------------------------------------------
    # 2️⃣  Graph + LinearOffChipLoad nodes (adds leading singleton)
    # --------------------------------------------------------------
    graph = Graph()

    # Q : (M, D) → tiled (1, M_grid, N_grid, tile_m, D)
    #   – stride (1, 0) broadcasts Q across the N_grid dimension.
    Q_load = LinearOffChipLoad(
        underlying=tensors["Q"],
        stride=(1, 0),                     # broadcast over N_grid
        out_shape_tiled=(M_grid, N_grid),
        tile_row=tile_m,
        tile_col=D,
        par_dispatch=4,
        transposed=False,
    )
    graph.add_node(Q_load)

    # K : (N, D) → tiled (1, M_grid, N_grid, D, tile_n)  (tiles transposed)
    #   – stride (0, 1) broadcasts K across the M_grid dimension.
    K_load = LinearOffChipLoad(
        underlying=tensors["K"],
        stride=(0, 1),                     # broadcast over M_grid
        out_shape_tiled=(M_grid, N_grid),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
        transposed=True,                   # produce (D, tile_n) tiles
    )
    graph.add_node(K_load)

    # V : (N, D) → tiled (1, M_grid, N_grid, tile_n, D)  (no transposition)
    V_load = LinearOffChipLoad(
        underlying=tensors["V"],
        stride=(0, 1),                     # broadcast over M_grid
        out_shape_tiled=(M_grid, N_grid),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
        transposed=False,
    )
    graph.add_node(V_load)

    # --------------------------------------------------------------
    # 3️⃣  Stream‑aligned tensors (no extra repeat/expand needed)
    # --------------------------------------------------------------
    Q_grid = Q_load          # (1, M_grid, N_grid, tile_m, D)
    K_grid = K_load          # (1, M_grid, N_grid, D, tile_n)
    V_grid = V_load          # (1, M_grid, N_grid, tile_n, D)

    # --------------------------------------------------------------
    # 4️⃣  scores = Q @ Kᵀ   (tile‑level matmul)
    # --------------------------------------------------------------
    scores = BinaryMap(
        graph,
        Q_grid,
        K_grid,
        map_fn.Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=1024,
    )                       # (1, M_grid, N_grid, tile_m, tile_n)

    # --------------------------------------------------------------
    # 5️⃣  exp_scores = exp(scores)
    # --------------------------------------------------------------
    exp_scores = UnaryMap(
        graph,
        scores,
        map_fn.Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )                       # (1, M_grid, N_grid, tile_m, tile_n)

    # --------------------------------------------------------------
    # 6️⃣  context = exp_scores @ V   (matmul + reduction over N_grid)
    # --------------------------------------------------------------
    context_tile = BinaryMap(
        graph,
        exp_scores,
        V_grid,
        map_fn.Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=1024,
    )                       # (1, M_grid, N_grid, tile_m, D)

    # Reduce over N_grid (rank=1)
    context_sum = Accum(
        graph,
        context_tile,
        output_stream_dtype=Tile(Float32(), shape=(tile_m, D)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(tile_m, D), dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )                       # (1, M_grid, tile_m, D)

    # --------------------------------------------------------------
    # 7️⃣  norm = sum(exp_scores) over N_grid (and tile_n)
    # --------------------------------------------------------------
    row_sum = UnaryMap(
        graph,
        exp_scores,
        map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )                       # (1, M_grid, N_grid, tile_m, 1)

    norm = Accum(
        graph,
        row_sum,
        output_stream_dtype=Tile(Float32(), shape=(tile_m, 1)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(tile_m, 1), dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )                       # (1, M_grid, tile_m, 1)

    # --------------------------------------------------------------
    # 8️⃣  output = context / norm   (broadcast norm over D)
    # --------------------------------------------------------------
    output_tiled = BinaryMap(
        graph,
        context_sum,
        norm,
        map_fn.Div(),
        write_back_mu=False,
        compute_bw=1024,
    )                       # (1, M_grid, tile_m, D)

    # --------------------------------------------------------------
    # 9️⃣  Untile back to flat (M, D) and return
    # --------------------------------------------------------------
    output_store = OffChipStore(
        graph,
        output_tiled,
        par_dispatch=4,
        store_file_name="output",
    )

    # --------------------------------------------------------------
    # Infer broadcast & finish
    # --------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, output_store