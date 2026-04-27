"""STeP implementation: Double bufferize/streamify chain.

Load -> Square -> Bufferize -> Streamify -> Silu -> Bufferize -> Streamify -> Store

Tests multiple PMU write/read cycles in sequence. Validates write_back_mu
timing and Bufferize/Streamify functional correctness through repeated
round-trips with compute between them.
"""

SEED = 42


def build_graph(dims):
    M, K = dims["M"], dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    torch.manual_seed(SEED)
    A = torch.randn(M, K)

    step_graph = Graph()

    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # Compute 1: Square
    squared = UnaryMap(
        graph=step_graph,
        input=load,
        fn=Square(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # First bufferize/streamify round-trip
    buff1 = Bufferize(step_graph, squared, 1)
    stream1 = Streamify(step_graph, buff1, [], 1)

    # Compute 2: Silu
    silued = UnaryMap(
        graph=step_graph,
        input=stream1,
        fn=Silu(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Second bufferize/streamify round-trip
    buff2 = Bufferize(step_graph, silued, 1)
    stream2 = Streamify(step_graph, buff2, [], 1)

    output = OffChipStore(
        graph=step_graph,
        input=stream2,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
