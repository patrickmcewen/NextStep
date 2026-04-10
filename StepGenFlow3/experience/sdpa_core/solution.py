def build_graph(dims):
    # ------------------------------------------------------------
    #  Geometry
    # ------------------------------------------------------------
    M = dims["M"]
    N = dims["N"]
    D = dims["D"]
    tile_m = dims["tile_m"]          # rows per Q‑tile
    tile_n = dims["tile_n"]          # rows per K/V‑tile (also cols of V)

    M_tiles = M // tile_m            # = 2
    N_tiles = N // tile_n            # = 8

    # ------------------------------------------------------------
    #  Deterministic random inputs (same seed order as reference)
    # ------------------------------------------------------------
    torch.manual_seed(SEED)
    Q = torch.randn(M, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)

    # ------------------------------------------------------------
    #  Build the graph
    # ------------------------------------------------------------
    graph = Graph()

    # ---------- Loads ----------
    # Q : broadcast across the N dimension (same Q‑tile for every N‑tile)
    q_load = LinearOffChipLoad(
        underlying=Q,
        stride=(1, 0),                     # advance one tile when M changes, N is broadcast
        out_shape_tiled=(M_tiles, N_tiles),
        tile_row=tile_m,
        tile_col=D,
        par_dispatch=1,
        transposed=False,
    )

    # K : broadcast across the M dimension, will be used transposed
    k_load = LinearOffChipLoad(
        underlying=K,
        stride=(0, 1),                     # M broadcast, N advances by one tile
        out_shape_tiled=(M_tiles, N_tiles),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=1,
        transposed=False,
    )

    # V : same tiling / broadcasting as K
    v_load = LinearOffChipLoad(
        underlying=V,
        stride=(0, 1),
        out_shape_tiled=(M_tiles, N_tiles),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=1,
        transposed=False,
    )

    # ---------- Q × Kᵀ (scores) ----------
    scores = BinaryMap(
        graph,
        q_load,                     # (tile_m, D)
        k_load,                     # (tile_n, D)
        fn=Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=1,
    )                               # → (tile_m, tile_n) per (M_tile, N_tile)

    # ---------- exp(scores) ----------
    exp_scores = UnaryMap(
        graph,
        scores,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1,
    )                               # → (tile_m, tile_n)

    # ---------- normaliser: sum_{j} exp(scores_{ij}) ----------
    # 1) row‑wise sum inside each tile (produces (tile_m,1))
    row_sum = UnaryMap(
        graph,
        exp_scores,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1,
    )                               # → (tile_m,1) per (M_tile, N_tile)

    # 2) sum across the outer N‑tile dimension
    norm = Accum(
        graph,
        row_sum,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_m, 1)),
        fn=AccumAdd(),
        init_fn=Zero(shape=(tile_m, 1), dtype=Float32()),
        accum_rank=1,                # reduce over N‑tiles
        write_back_mu=False,
        compute_bw=1,
    )                               # → (1, M_tiles, tile_m, 1)

    # ---------- weighted context: exp(scores) @ V ----------
    ctx_tile = BinaryMap(
        graph,
        exp_scores,                 # (tile_m, tile_n)
        v_load,                     # (tile_n, D)
        fn=Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=1,
    )                               # → (tile_m, D) per N‑tile

    # ---------- sum over N‑tiles to get final context ----------
    context = Accum(
        graph,
        ctx_tile,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_m, D)),
        fn=AccumAdd(),
        init_fn=Zero(shape=(tile_m, D), dtype=Float32()),
        accum_rank=1,                # reduce over N‑tiles
        write_back_mu=False,
        compute_bw=1,
    )                               # → (1, M_tiles, tile_m, D)

    # ---------- final division (softmax) ----------
    out = BinaryMap(
        graph,
        context,                    # (tile_m, D)
        norm,                       # (tile_m, 1) – broadcasted along D
        fn=Div(),
        write_back_mu=False,
        compute_bw=1,
    )                               # → (tile_m, D)

    # ---------- Store ----------
    store = OffChipStore(
        graph,
        out,
        par_dispatch=1,
        store_file_name="output",
    )

    # ------------------------------------------------------------
    #  Resolve any implicit broadcasts and return
    # ------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, store