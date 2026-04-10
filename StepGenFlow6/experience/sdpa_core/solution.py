def build_graph(dims, tensors):
    # --------------------------------------------------------------
    # 1️⃣ dimensions & tile sizes
    # --------------------------------------------------------------
    M = dims["M"]          # number of queries
    N = dims["N"]          # number of keys / values
    D = dims["D"]          # feature dimension

    tm = dims["tile_m"]    # tile rows for M
    tn = dims["tile_n"]    # tile cols for N

    Mg = M // tm           # number of M‑tiles
    Ng = N // tn           # number of N‑tiles

    # --------------------------------------------------------------
    # 2️⃣ create graph and stream‑load the three inputs
    # --------------------------------------------------------------
    graph = Graph()

    Q_load = LinearOffChipLoad(
        underlying=tensors["Q"],          # (M, D)
        stride=(1, 0),                    # advance across Mg, broadcast over Ng
        out_shape_tiled=(Mg, Ng),
        tile_row=tm,
        tile_col=D,
        par_dispatch=4,
    )
    graph.add_node(Q_load)

    K_load = LinearOffChipLoad(
        underlying=tensors["K"],          # (N, D)
        stride=(0, 1),                    # broadcast over Mg, advance across Ng
        out_shape_tiled=(Mg, Ng),
        tile_row=tn,
        tile_col=D,
        par_dispatch=4,
    )
    graph.add_node(K_load)

    V_load = LinearOffChipLoad(
        underlying=tensors["V"],          # (N, D)
        stride=(0, 1),                    # broadcast over Mg, advance across Ng
        out_shape_tiled=(Mg, Ng),
        tile_row=tn,
        tile_col=D,
        par_dispatch=4,
    )
    graph.add_node(V_load)

    # --------------------------------------------------------------
    # 3️⃣ scores = Q @ Kᵀ   → (1, Mg, Ng, tm, tn)
    # --------------------------------------------------------------
    scores = BinaryMap(
        graph,
        Q_load,
        K_load,
        map_fn.Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=1024,
    )  # (1, Mg, Ng, tm, tn)

    # --------------------------------------------------------------
    # 4️⃣ exponentiate the scores
    # --------------------------------------------------------------
    exp_scores = UnaryMap(
        graph,
        scores,
        map_fn.Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )  # (1, Mg, Ng, tm, tn)

    # --------------------------------------------------------------
    # 5️⃣ context = exp_scores @ V   → (1, Mg, tm, D) after Ng reduction
    # --------------------------------------------------------------
    ctx_partial = BinaryMap(
        graph,
        exp_scores,
        V_load,
        map_fn.Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=1024,
    )  # (1, Mg, Ng, tm, D)

    # sum over Ng (rank=1)
    context = Accum(
        graph,
        ctx_partial,
        Tile(Float32(), shape=(tm, D)),
        accum_fn.Add(),
        init_fn.Empty(shape=(tm, D), dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )  # (1, Mg, tm, D)

    # --------------------------------------------------------------
    # 6️⃣ normaliser = exp_scores summed over tn (keepdim) → (1, Mg, tm, 1)
    # --------------------------------------------------------------
    norm_tile = UnaryMap(
        graph,
        exp_scores,
        map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )  # (1, Mg, Ng, tm, 1)

    norm = Accum(
        graph,
        norm_tile,
        Tile(Float32(), shape=(tm, 1)),
        accum_fn.Add(),
        init_fn.Empty(shape=(tm, 1), dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )  # (1, Mg, tm, 1)

    # --------------------------------------------------------------
    # 7️⃣ final output = context / norm   (norm broadcast over D)
    # --------------------------------------------------------------
    output_tiled = BinaryMap(
        graph,
        context,
        norm,
        map_fn.Div(),
        write_back_mu=False,
        compute_bw=1024,
    )  # (1, Mg, tm, D)

    # --------------------------------------------------------------
    # 8️⃣ store back to off‑chip memory
    # --------------------------------------------------------------
    out_op = OffChipStore(
        graph,
        output_tiled,
        par_dispatch=4,
    )

    # --------------------------------------------------------------
    # 9️⃣ finalize graph
    # --------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, out_op