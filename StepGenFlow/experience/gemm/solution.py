def build_graph(dims):
    """
    Build a STeP graph that computes C = A @ B.
    The reference uses a fixed random seed, so we do the same.
    For simplicity we load each whole matrix as a single tile and perform a
    single Matmul via BinaryMap (no reduction needed).
    """
    torch.manual_seed(42)

    # Concrete input tensors (identical to the PyTorch reference)
    A = torch.randn(dims["M"], dims["K"])
    B = torch.randn(dims["K"], dims["N"])

    graph = Graph()

    # -------------------------------------------------------------
    # Load A as one tile covering the full (M, K) matrix.
    # -------------------------------------------------------------
    a = LinearOffChipLoad(
        underlying=A,
        stride=(1, 1),                 # only one tile → trivial stride
        out_shape_tiled=(1, 1),        # one tile in each stream dimension
        tile_row=dims["M"],            # tile spans the whole M dimension
        tile_col=dims["K"],            # tile spans the whole K dimension
        par_dispatch=1,
        transposed=False,
    )

    # -------------------------------------------------------------
    # Load B as one tile covering the full (K, N) matrix.
    # -------------------------------------------------------------
    b = LinearOffChipLoad(
        underlying=B,
        stride=(1, 1),
        out_shape_tiled=(1, 1),
        tile_row=dims["K"],
        tile_col=dims["N"],
        par_dispatch=1,
        transposed=False,
    )

    # -------------------------------------------------------------
    # GEMM – single Matmul (no reduction needed because each operand is one tile).
    # Use BinaryMap with the ordinary Matmul map function.
    # -------------------------------------------------------------
    gemm = BinaryMap(
        graph,
        a,
        b,
        fn=Matmul(weight_transposed=False),   # map_fn.Matmul
        write_back_mu=False,
        compute_bw=0,
    )

    # -------------------------------------------------------------
    # Store – the functional emulator will untile this node to produce the
    # dense (M, N) result.
    # -------------------------------------------------------------
    out = OffChipStore(graph, gemm, par_dispatch=1, store_file_name="output")

    # -------------------------------------------------------------
    # Finalise the graph.
    # -------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, out