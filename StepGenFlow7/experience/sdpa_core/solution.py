def build_graph(dims, tensors):
    # ------------------------------------------------------------
    # Dimensions & tiling parameters
    # ------------------------------------------------------------
    M = dims["M"]
    N = dims["N"]
    D = dims["D"]
    tile_m = dims["tile_m"]          # rows of Q / output tiles
    tile_n = dims["tile_n"]          # rows of K/V tiles (also N‑tile size)

    # Number of tiles along each logical dimension
    grid_m = M // tile_m              # = 2  (tiles over M)
    grid_n = N // tile_n              # = 8  (tiles over N)

    # ------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------
    graph = Graph()

    # ------------------------------------------------------------
    # Load dense tensors into tiled stream form
    # ------------------------------------------------------------
    Q = LinearOffChipLoad(
        tensors["Q"],
        stride=(1, 0),                     # advance 1 tile per grid_m step, broadcast over grid_n
        out_shape_tiled=(grid_m, grid_n),
        tile_row=tile_m,
        tile_col=D,
        par_dispatch=1,
    )
    graph.add_node(Q)

    K = LinearOffChipLoad(
        tensors["K"],
        stride=(0, 1),                     # broadcast over grid_m, advance 1 tile per grid_n step
        out_shape_tiled=(grid_m, grid_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=1,
    )
    graph.add_node(K)

    V = LinearOffChipLoad(
        tensors["V"],
        stride=(0, 1),                     # same broadcasting pattern as K
        out_shape_tiled=(grid_m, grid_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=1,
    )
    graph.add_node(V)

    # ------------------------------------------------------------
    # Scores = Q @ Kᵀ   (tiled matmul, weight_transposed=True)
    # ------------------------------------------------------------
    scores = BinaryMap(
        graph,
        Q,
        K,
        map_fn.Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------
    # exp_scores = exp(scores)
    # ------------------------------------------------------------
    exp_scores = UnaryMap(
        graph,
        scores,
        map_fn.Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------
    # context = exp_scores @ V   (tiled matmul, no transpose)
    # ------------------------------------------------------------
    context_tiles = BinaryMap(
        graph,
        exp_scores,
        V,
        map_fn.Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------
    # Reduce over the N‑tiles (grid_n) to obtain per‑query results
    # ------------------------------------------------------------
    # 1) sum over the streaming dimension that represents grid_n
    context_sum = Accum(
        graph,
        context_tiles,
        Tile(Float32(), shape=(tile_m, D)),
        accum_fn.Add(),
        init_fn=None,
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    # 2) merge that stream dim into the tiled‑row dimension → full M dimension
    context = Accum(
        graph,
        context_sum,
        Tile(Float32(), shape=(tile_m, D)),
        accum_fn.RetileRow(),
        init_fn=None,
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------
    # Compute the normalisation term   norm = Σₙ exp(QKᵀ)
    # ------------------------------------------------------------
    # Sum over N‑tiles (grid_n) first
    exp_sum_stream = Accum(
        graph,
        exp_scores,
        Tile(Float32(), shape=(tile_m, tile_n)),
        accum_fn.Add(),
        init_fn=None,
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    # Then sum inside each tile across the N‑tile columns
    norm_tile = UnaryMap(
        graph,
        exp_sum_stream,
        map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Merge stream dim into rows to get (1, M, 1)
    norm = Accum(
        graph,
        norm_tile,
        Tile(Float32(), shape=(tile_m, 1)),
        accum_fn.RetileRow(),
        init_fn=None,
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------
    # Final output = context / norm   (broadcast division)
    # ------------------------------------------------------------
    out_tiled = BinaryMap(
        graph,
        context,
        norm,
        map_fn.Div(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------
    # Convert tiled tensor back to dense matrix (M × D)
    # ------------------------------------------------------------
    out = OffChipStore(
        graph,
        out_tiled,
        par_dispatch=1,
    )

    # ------------------------------------------------------------
    # Finalise graph
    # ------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, out