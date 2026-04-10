def build_graph(dims: dict):
    """
    Complete STeP graph for the scaled‑dot‑product attention core.
    All tiled arithmetic, stream‑dim reductions and the final division are
    expressed as STeP nodes, ending with an OffChipStore.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Extract dimensions and stream grid sizes
    # ------------------------------------------------------------------
    M = dims["M"]
    N = dims["N"]
    D = dims["D"]
    tile_m = dims["tile_m"]          # rows of Q / output tile
    tile_n = dims["tile_n"]          # columns of Kᵀ / V tile

    grid_m = M // tile_m              # number of Q‑tiles  (stream dim 0)
    grid_n = N // tile_n              # number of K/V‑tiles (stream dim 1)

    # ------------------------------------------------------------------
    # 2️⃣  Generate deterministic inputs (same seed as reference)
    # ------------------------------------------------------------------
    torch.manual_seed(42)                     # SEED from the reference
    Q_tensor = torch.randn(M, D)               # [M, D]
    K_tensor = torch.randn(N, D)               # [N, D]
    V_tensor = torch.randn(N, D)               # [N, D]

    # ------------------------------------------------------------------
    # 3️⃣  Build the graph: off‑chip loads
    # ------------------------------------------------------------------
    g = Graph()

    # Q → tiles (tile_m, D), broadcast over the N‑stream dimension
    Q_load = LinearOffChipLoad(
        underlying=Q_tensor,
        stride=(1, 0),                         # step in grid_m, broadcast over grid_n
        out_shape_tiled=(grid_m, grid_n),
        tile_row=tile_m,
        tile_col=D,
        par_dispatch=1,
    )
    g.add_node(Q_load)

    # Kᵀ → tiles (D, tile_n), broadcast over the M‑stream dimension
    K_load = LinearOffChipLoad(
        underlying=K_tensor.t(),               # shape (D, N)
        stride=(0, 1),                         # broadcast over grid_m, step in grid_n
        out_shape_tiled=(grid_m, grid_n),
        tile_row=D,                            # full D dimension
        tile_col=tile_n,                       # tile over N dimension
        par_dispatch=1,
    )
    g.add_node(K_load)

    # V → tiles (tile_n, D), broadcast over the M‑stream dimension
    V_load = LinearOffChipLoad(
        underlying=V_tensor,
        stride=(0, 1),                         # broadcast across grid_m, step in grid_n
        out_shape_tiled=(grid_m, grid_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=1,
    )
    g.add_node(V_load)

    # ------------------------------------------------------------------
    # 4️⃣  Tiled arithmetic (all STeP compute nodes)
    # ------------------------------------------------------------------

    # scores = Q @ Kᵀ   → (grid_m, grid_n, tile_m, tile_n)
    scores_node = BinaryMap(
        graph=g,
        in1=Q_load,
        in2=K_load,
        fn=map_fn.Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=1024,
    )

    # exp_scores = exp(scores)
    exp_node = UnaryMap(
        graph=g,
        input=scores_node,
        fn=map_fn.Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # context_partial = exp_scores @ V   → (grid_m, grid_n, tile_m, D)
    context_node = BinaryMap(
        graph=g,
        in1=exp_node,
        in2=V_load,
        fn=map_fn.Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=1024,
    )

    # norm_tile = row‑wise sum of exp_scores over tile_n → (grid_m, grid_n, tile_m, 1)
    norm_tile_node = UnaryMap(
        graph=g,
        input=exp_node,
        fn=map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 5️⃣  Reduce across the N‑stream dimension (grid_n) with Accum
    # ------------------------------------------------------------------

    # Reduce context_partial over grid_n
    context_accum = Accum(
        graph=g,
        input=context_node,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_m, D)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(tile_m, D), dtype=Float32()),
        accum_rank=1,                     # reduce 1 stream dim (grid_n)
        write_back_mu=False,
        compute_bw=1024,
    )

    # Reduce norm_tile over grid_n
    norm_accum = Accum(
        graph=g,
        input=norm_tile_node,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_m, 1)),
        fn=accum_fn.Add(),
        init_fn=init_fn.Zero(shape=(tile_m, 1), dtype=Float32()),
        accum_rank=1,                     # reduce 1 stream dim (grid_n)
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 6️⃣  Final per‑tile division (context / norm)
    # ------------------------------------------------------------------
    out_tile = BinaryMap(
        graph=g,
        in1=context_accum,
        in2=norm_accum,
        fn=map_fn.Div(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # ------------------------------------------------------------------
    # 7️⃣  Store the final (M, D) result
    # ------------------------------------------------------------------
    output = OffChipStore(
        graph=g,
        input=out_tile,
        par_dispatch=4,
        store_file_name="output",
    )

    # ------------------------------------------------------------------
    # 8️⃣  Broadcast inference and return
    # ------------------------------------------------------------------
    g = infer_broadcast(g)
    return g, output