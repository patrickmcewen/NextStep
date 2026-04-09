def build_graph(dims):
    """
    STeP graph for the core of scaled‑dot‑product attention:

        out = (exp(Q @ Kᵀ) @ V) / exp(Q @ Kᵀ).sum(-1, keepdim=True)

    The tiled algorithm follows the reference in the problem statement.
    Random tensors are generated inside the builder (seeded with SEED) so the
    functional emulator can compare the result with the PyTorch reference.
    """
    graph = Graph()

    # ------------------------------------------------------------------
    # 1️⃣  Problem sizes & tiling parameters
    # ------------------------------------------------------------------
    M, N, D = dims["M"], dims["N"], dims["D"]
    tm, tn = dims["tile_m"], dims["tile_n"]

    # ------------------------------------------------------------------
    # 2️⃣  Random inputs (same seed as the reference)
    # ------------------------------------------------------------------
    torch.manual_seed(SEED)
    Q = torch.randn(M, D)          # [M, D]
    K = torch.randn(N, D)          # [N, D]
    V = torch.randn(N, D)          # [N, D]

    # ------------------------------------------------------------------
    # 3️⃣  Load & tile the inputs.
    #    Stride (tiles_per_dim, 0) or (0, tiles_per_dim) broadcasts the operand
    #    across the dimension it does not vary.
    # ------------------------------------------------------------------
    # Q : (M, D) → stream shape (M//tm, N//tn) with broadcast on N
    Q_load = LinearOffChipLoad(
        Q,
        stride=(D // tn, 0),                     # advance on M‑tiles, broadcast N
        out_shape_tiled=(M // tm, N // tn),
        tile_row=tm,
        tile_col=tn,
        par_dispatch=1,
    )

    # K : (N, D) → same stream shape, broadcast on M
    K_load = LinearOffChipLoad(
        K,
        stride=(0, D // tn),                     # broadcast M, advance on N‑tiles
        out_shape_tiled=(M // tm, N // tn),
        tile_row=tn,
        tile_col=tn,
        par_dispatch=1,
    )

    # V : (N, D) → same broadcast pattern as K
    V_load = LinearOffChipLoad(
        V,
        stride=(0, D // tn),
        out_shape_tiled=(M // tm, N // tn),
        tile_row=tn,
        tile_col=tn,
        par_dispatch=1,
    )

    # ------------------------------------------------------------------
    # 4️⃣  scores = Q @ Kᵀ   (K is transposed)
    # ------------------------------------------------------------------
    scores = BinaryMap(
        graph,
        Q_load,
        K_load,
        fn=Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=0,
    )

    # ------------------------------------------------------------------
    # 5️⃣  exp_scores = exp(scores)
    # ------------------------------------------------------------------
    exp_scores = UnaryMap(
        graph,
        scores,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=0,
    )

    # ------------------------------------------------------------------
    # 6️⃣  context_tile = exp_scores @ V   (V not transposed)
    # ------------------------------------------------------------------
    context_tile = BinaryMap(
        graph,
        exp_scores,
        V_load,
        fn=Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=0,
    )

    # ------------------------------------------------------------------
    # 7️⃣  context_sum = Σ_N context_tile   (reduce over streamed N dimension)
    # ------------------------------------------------------------------
    # After reduction we have stream shape (M_tile,) and tile shape (tm, tn)
    context_sum = Accum(
        graph,
        context_tile,
        output_stream_dtype=Tile(Float32(), (tm, tn)),
        fn=AccumAdd(),
        init_fn=Zero((tm, tn), dtype=Float32()),
        accum_rank=1,               # reduce over the N_tile dimension
        write_back_mu=False,
        compute_bw=0,
    )   # shape -> (1, M_tile, tm, tn)

    # ------------------------------------------------------------------
    # 8️⃣  norm = Σ_col exp_scores → Σ_N Σ_col
    # ------------------------------------------------------------------
    # a) sum over tile‑col (tn) → (M_tile, N_tile, tm, 1)
    row_sum_tile = UnaryMap(
        graph,
        exp_scores,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=0,
    )
    # b) sum over streamed N dimension → (M_tile, tm, 1)
    norm_tile = Accum(
        graph,
        row_sum_tile,
        output_stream_dtype=Tile(Float32(), (tm, 1)),
        fn=AccumAdd(),
        init_fn=Zero((tm, 1), dtype=Float32()),
        accum_rank=1,               # reduce over N_tile
        write_back_mu=False,
        compute_bw=0,
    )   # shape -> (1, M_tile, tm, 1)

    # ------------------------------------------------------------------
    # 9️⃣  out = context_sum / norm_tile   (broadcast over the last tile dim)
    # ------------------------------------------------------------------
    out = BinaryMap(
        graph,
        context_sum,
        norm_tile,
        fn=Div(),
        write_back_mu=False,
        compute_bw=0,
    )   # final shape -> (1, M_tile, tm, tn)

    # ------------------------------------------------------------------
    # 10️⃣  Store the result (untile inside the emulator).
    # ------------------------------------------------------------------
    store = OffChipStore(graph, out, par_dispatch=1)

    # ------------------------------------------------------------------
    # 11️⃣  Infer broadcasting (if any) and return.
    # ------------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, store