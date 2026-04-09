def build_graph(dims):
    # ------------------------------------------------------------------
    # 1️⃣  Reset the random seed (must match the PyTorch reference)
    # ------------------------------------------------------------------
    torch.manual_seed(42)                     # same seed as the reference code
    graph = Graph()

    # ------------------------------------------------------------------
    # 2️⃣  Problem dimensions and tiling parameters
    # ------------------------------------------------------------------
    M, K = dims["M"], dims["K"]
    tm, tk = dims["tile_m"], dims["tile_k"]

    # Number of tiles in each dimension
    out_shape_tiled = (M // tm, K // tk)

    # Stride for a (M, K) matrix tiled with (tm, tk):
    #   (tiles_along_K, 1) → (K//tk, 1)
    stride = (K // tk, 1)

    # ------------------------------------------------------------------
    # 3️⃣  Load A and B from off‑chip (the random tensors)
    # ------------------------------------------------------------------
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(M, K, dtype=torch.float32)

    # LinearOffChipLoad is a source node – it does NOT take the graph as an argument.
    A_load = LinearOffChipLoad(
        underlying=A,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=tm,
        tile_col=tk,
        par_dispatch=1,
        transposed=False,
    )

    B_load = LinearOffChipLoad(
        underlying=B,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=tm,
        tile_col=tk,
        par_dispatch=1,
        transposed=False,
    )

    # ------------------------------------------------------------------
    # 4️⃣  Element‑wise addition (BinaryMap with Add)
    # ------------------------------------------------------------------
    add = BinaryMap(
        graph,
        in1=A_load,
        in2=B_load,
        fn=map_fn.Add(),
        write_back_mu=False,
        compute_bw=1,
    )

    # ------------------------------------------------------------------
    # 5️⃣  Store the result (sink node)
    # ------------------------------------------------------------------
    store = OffChipStore(graph, add, par_dispatch=1)

    # ------------------------------------------------------------------
    # 6️⃣  Broadcast inference & return
    # ------------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, store