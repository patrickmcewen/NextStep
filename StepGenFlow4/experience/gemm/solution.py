def build_graph(dims: dict, tensors: dict):
    # --------------------------------------------------------------
    # 1️⃣  Extract dimensions and raw tensors
    # --------------------------------------------------------------
    M, K, N = dims["M"], dims["K"], dims["N"]
    tm, tk, tn = dims["tile_m"], dims["tile_k"], dims["tile_n"]

    grid_m = M // tm          # number of M‑tiles
    grid_k = K // tk          # number of K‑tiles
    grid_n = N // tn          # number of N‑tiles

    # --------------------------------------------------------------
    # 2️⃣  Load tiled tensors with broadcasting on the missing axes
    # --------------------------------------------------------------
    graph = Graph()

    # A : (M, K) → (grid_m, grid_n, grid_k, tm, tk)
    # broadcast across grid_n (stride 0 on that dim)
    A_load = LinearOffChipLoad(
        underlying=tensors["A"],
        stride=(K // tk, 0, 1),                # (grid_k, 0, 1)
        out_shape_tiled=(grid_m, grid_n, grid_k),
        tile_row=tm,
        tile_col=tk,
        par_dispatch=4,
    )
    graph.add_node(A_load)

    # B : (K, N) → (grid_m, grid_n, grid_k, tk, tn)
    # broadcast across grid_m (stride 0 on that dim)
    B_load = LinearOffChipLoad(
        underlying=tensors["B"],
        stride=(0, 1, N // tn),                # (0, 1, grid_n)
        out_shape_tiled=(grid_m, grid_n, grid_k),
        tile_row=tk,
        tile_col=tn,
        par_dispatch=4,
    )
    graph.add_node(B_load)

    # --------------------------------------------------------------
    # 3️⃣  Tile‑level GEMM with accumulation over the K‑stream dimension
    # --------------------------------------------------------------
    C_acc = BinaryMapAccum(
        graph,
        A_load,
        B_load,
        map_accum_fn.Matmul(weight_transposed=False),
        init_fn.Zero(shape=(tm, tn), dtype=Float32()),
        rank=1,                     # reduce over the K‑stream (last) dimension
        write_back_mu=False,
        compute_bw=1024,
    )

    # --------------------------------------------------------------
    # 4️⃣  Store the final result (the emulator will untile it)
    # --------------------------------------------------------------
    output = OffChipStore(
        graph,
        C_acc,
        par_dispatch=4,
        store_file_name="output",
    )

    # --------------------------------------------------------------
    # 5️⃣  Finalise graph
    # --------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, output