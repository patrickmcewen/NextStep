def build_graph(dims):
    """
    Build a STeP graph that computes C = A @ B using tiled GEMM.

    The implementation follows the tiled reference in the problem statement:
    1. Load A (M×K) and B (K×N) as tiled streams, broadcasting each over the
       opposite outer dimension.
    2. Perform a tiled matmul on each (tm × tk) × (tk × tn) tile pair.
    3. Accumulate (sum) over the K‑tile dimension.
    4. Store the result, letting the runtime untile to shape (M, N).

    All data‑path work (matmul, reduction) is expressed with STeP operators.
    """
    # ------------------------------------------------------------------ #
    # 0.  Setup – retrieve dimensions and seed the RNG to match PyTorch
    # ------------------------------------------------------------------ #
    torch.manual_seed(42)

    M, K, N = dims["M"], dims["K"], dims["N"]
    tm, tk, tn = dims["tile_m"], dims["tile_k"], dims["tile_n"]

    # Number of tiles in each dimension
    M_t = M // tm
    K_t = K // tk
    N_t = N // tn

    # ------------------------------------------------------------------ #
    # 1.  Load the tiled operands
    # ------------------------------------------------------------------ #
    graph = Graph()

    # A : (M, K) → tiles (tm, tk) broadcast over N‑tiles
    A_load = LinearOffChipLoad(
        underlying=torch.randn(M, K),               # same RNG as reference
        stride=(K_t, 0, 1),                         # (K_tile, 0, 1) → broadcast over N
        out_shape_tiled=(M_t, N_t, K_t),            # stream dims (M_tile, N_tile, K_tile)
        tile_row=tm,
        tile_col=tk,
        par_dispatch=1,
        transposed=False,
    )

    # B : (K, N) → tiles (tk, tn) broadcast over M‑tiles
    B_load = LinearOffChipLoad(
        underlying=torch.randn(K, N),
        stride=(0, 1, N_t),                         # (0, 1, N_tile) → broadcast over M
        out_shape_tiled=(M_t, N_t, K_t),
        tile_row=tk,
        tile_col=tn,
        par_dispatch=1,
        transposed=False,
    )

    # ------------------------------------------------------------------ #
    # 2.  Tile‑level matmul + reduction over the K‑stream dimension
    # ------------------------------------------------------------------ #
    # BinaryMapAccum performs: tile‑matmul → sum‑reduce over the last
    # `rank` stream dimension (here rank=1 → the K_tile axis).
    C_grid = BinaryMapAccum(
        graph,
        A_load,
        B_load,
        fn=MapAccumMatmul(weight_transposed=False),
        init_fn=Zero(shape=(tm, tn), dtype=Float32()),
        rank=1,                 # reduce over the K‑tile dimension
        write_back_mu=False,
        compute_bw=1,
    )

    # ------------------------------------------------------------------ #
    # 3.  Store – the runtime will untile back to (M, N)
    # ------------------------------------------------------------------ #
    out = OffChipStore(
        graph,
        C_grid,
        par_dispatch=1,
        store_file_name="output",
    )

    # ------------------------------------------------------------------ #
    # 4.  Broadcast inference (required boiler‑plate)
    # ------------------------------------------------------------------ #
    graph = infer_broadcast(graph)
    return graph, out