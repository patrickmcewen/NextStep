"""STeP implementation: GEMM with TileMK tiling via the Linear kernel."""

SEED = 42


def build_graph(dims):
    """Weight-stationary GEMM: C = A @ B."""
    M, K, N = dims["M"], dims["K"], dims["N"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)
    tile_n = dims.get("tile_n", 16)
    par_dispatch = 4
    compute_bw = 4096

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    graph = Graph()

    torch.manual_seed(SEED)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    a_load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=par_dispatch,
    )

    b_load = LinearOffChipLoad(
        underlying=B,
        stride=(0, 1, N // tile_n),
        out_shape_tiled=(M // tile_m, N // tile_n, K // tile_k),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=par_dispatch,
    )

    matmul = BinaryMapAccum(
        graph=graph,
        in1=a_load,
        in2=b_load,
        fn=map_accum_fn.Matmul(),
        init_fn=init_fn.Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,
        write_back_mu=True,
        compute_bw=compute_bw,
    )

    output_op = OffChipStore(
        graph=graph,
        input=matmul,
        par_dispatch=par_dispatch,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output_op
