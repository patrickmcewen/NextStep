def build_graph(dims):
    """
    Build a STeP graph that computes A + B element‑wise.

    Parameters
    ----------
    dims : dict
        Must contain the keys ``M``, ``K``, ``tile_m`` and ``tile_k``.

    Returns
    -------
    (graph, output_op)
        *graph* – the constructed MultiDiGraph
        *output_op* – the final OffChipStore node whose value is the result.
    """
    ########################################################################
    # 1️⃣  Create the concrete input tensors (same seed as the reference)
    ########################################################################
    torch.manual_seed(42)                     # match the PyTorch reference
    M, K = dims["M"], dims["K"]
    A_tensor = torch.randn(M, K)
    B_tensor = torch.randn(M, K)

    ########################################################################
    # 2️⃣  Tile configuration
    ########################################################################
    tile_m = dims["tile_m"]
    tile_k = dims["tile_k"]

    tiles_m = M // tile_m          # number of row tiles
    tiles_k = K // tile_k          # number of column tiles

    # stride for LinearOffChipLoad: (cols_per_tile, 1)
    stride = (K // tile_k, 1)

    ########################################################################
    # 3️⃣  Assemble the graph
    ########################################################################
    graph = Graph()

    # Load the two operands from off‑chip memory
    a_load = LinearOffChipLoad(
        underlying=A_tensor,
        stride=stride,
        out_shape_tiled=(tiles_m, tiles_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=1,
        transposed=False,
    )

    b_load = LinearOffChipLoad(
        underlying=B_tensor,
        stride=stride,
        out_shape_tiled=(tiles_m, tiles_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=1,
        transposed=False,
    )

    # Element‑wise addition
    add = BinaryMap(
        graph,
        in1=a_load,
        in2=b_load,
        fn=map_fn.Add(),
        write_back_mu=False,
        compute_bw=0,
    )

    # Store the result (final observable node)
    store = OffChipStore(
        graph,
        add,               # positional input – no `_input` keyword
        par_dispatch=1,
        store_file_name="output",
    )

    ########################################################################
    # 4️⃣  Broadcast inference and return
    ########################################################################
    graph = infer_broadcast(graph)
    return graph, store