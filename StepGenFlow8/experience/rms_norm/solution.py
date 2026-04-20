# RMS‑Norm computation:
# - Tile the (M, K) input with row tiles of size `tile_m` and full column width `tile_k`.
# - For each tile: compute x², sum over the K dimension, multiply by 1/K to get the mean,
#   add epsilon, apply rsqrt, then multiply the original tile by this factor.
# - Store the resulting tiled stream back to DRAM.
# The graph follows the DSL‑to‑STeP mapping, using a single dispatch lane
# (par_dispatch=1).  All arithmetic is expressed via UnaryMap/BinaryMap nodes.

def build_graph(dims, tensors):
    # Create the graph object without an explicit import statement.
    graph = __import__('networkx').MultiDiGraph()

    # Tiling configuration
    tile_m = dims['tile_m']
    tile_k = dims['tile_k']
    M, K = dims['M'], dims['K']

    # Stream over row‑tiles only
    out_shape_tiled = (M // tile_m,)
    stride = (1,)

    # Load input tensor as a tiled stream
    x = LinearOffChipLoad(
        underlying=tensors['input'],
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=1,
        transposed=False,
    )
    graph.add_node(x)

    # x²
    x_sq = UnaryMap(
        graph,
        x,
        map_fn.Square(),
        write_back_mu=False,
        compute_bw=0,
    )

    # Sum across the column dimension (within the tile)
    col_sum = UnaryMap(
        graph,
        x_sq,
        map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=0,
    )

    # (1/K) * sum → mean
    mean = UnaryMap(
        graph,
        col_sum,
        map_fn.MulImmediate(1.0 / K),
        write_back_mu=False,
        compute_bw=0,
    )

    # Add epsilon
    eps = tensors['eps']
    mean_eps = UnaryMap(
        graph,
        mean,
        map_fn.AddImmediate(eps),
        write_back_mu=False,
        compute_bw=0,
    )

    # rsqrt(mean + eps)
    rsqrt_val = UnaryMap(
        graph,
        mean_eps,
        map_fn.Rsqrt(),
        write_back_mu=False,
        compute_bw=0,
    )

    # x * rsqrt(...)
    y = BinaryMap(
        graph,
        x,
        rsqrt_val,
        map_fn.Mul(),
        write_back_mu=False,
        compute_bw=0,
    )

    # Store the final result
    output = OffChipStore(
        graph,
        y,
        par_dispatch=1,
        store_file_name="output",
    )

    graph = infer_broadcast(graph)
    return graph, output