def build_graph(dims, tensors):
    """
    Build a STeP graph for C = A @ B using compute nodes instead of PyTorch ops.
    """
    # ------------------------------------------------------------------
    # 1. Extract problem sizes and tile counts
    # ------------------------------------------------------------------
    M = dims["M"]
    K = dims["K"]
    N = dims["N"]
    tm = dims["tile_m"]
    tk = dims["tile_k"]
    tn = dims["tile_n"]

    M_grid = M // tm      # number of M‑tiles
    K_grid = K // tk      # number of K‑tiles
    N_grid = N // tn      # number of N‑tiles

    # ------------------------------------------------------------------
    # 2. Create the graph and add the two LinearOffChipLoad sources
    # ------------------------------------------------------------------
    graph = Graph()

    # A : (M, K) → tiled stream (M_grid, N_grid, K_grid, tm, tk)
    A_load = LinearOffChipLoad(
        underlying=tensors["A"],          # (M, K)
        stride=(K_grid, 0, 1),            # broadcast across N_grid
        out_shape_tiled=(M_grid, N_grid, K_grid),
        tile_row=tm,
        tile_col=tk,
        par_dispatch=4,
    )
    graph.add_node(A_load)

    # B : (K, N) → tiled stream (M_grid, N_grid, K_grid, tk, tn)
    B_load = LinearOffChipLoad(
        underlying=tensors["B"],          # (K, N)
        stride=(0, 1, N_grid),            # broadcast across M_grid
        out_shape_tiled=(M_grid, N_grid, K_grid),
        tile_row=tk,
        tile_col=tn,
        par_dispatch=4,
    )
    graph.add_node(B_load)

    # ------------------------------------------------------------------
    # 3. Tile‑level GEMM with accumulation over the K‑stream dimension
    # ------------------------------------------------------------------
    # BinaryMapAccum performs per‑tile matmul and then reduces over the
    # innermost stream dimension (rank=1) → summation over K_grid.
    C_node = BinaryMapAccum(
        graph,
        A_load,
        B_load,
        map_accum_fn.Matmul(weight_transposed=False),
        init_fn.Zero(shape=(tm, tn), dtype=Float32()),
        1,                # rank: reduce over K_grid
        False,
        1024,
    )

    # ------------------------------------------------------------------
    # 4. Store the tiled result back to off‑chip memory
    # ------------------------------------------------------------------
    store_node = OffChipStore(graph, C_node, par_dispatch=4)

    # ------------------------------------------------------------------
    # 5. Finalise the graph (broadcast inference) and return
    # ------------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, store_node