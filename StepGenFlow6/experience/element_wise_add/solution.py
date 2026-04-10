def build_graph(dims):
    """
    Build a STeP graph that computes C = A + B on tiled tensors.

    dims dict contains:
        M, K          – matrix dimensions
        tile_m, tile_k – tile sizes (must divide M and K)
    """
    # ------------------------------------------------------------------
    # 0. Boiler‑plate
    # ------------------------------------------------------------------
    graph = Graph()

    # ------------------------------------------------------------------
    # 1. Extract dimensions and create the same random inputs as the reference
    # ------------------------------------------------------------------
    torch.manual_seed(42)                     # match the reference seed
    M, K = dims["M"], dims["K"]
    tile_m, tile_k = dims["tile_m"], dims["tile_k"]

    # Tile counts (stream dimensions)
    tm = M // tile_m
    tk = K // tile_k

    # ------------------------------------------------------------------
    # 2. Off‑chip loads – tiled streams of shape (1, tm, tk, tile_m, tile_k)
    # ------------------------------------------------------------------
    # stride for a 2‑D tensor: (K//tile_k, 1)
    stride = (K // tile_k, 1)
    out_shape_tiled = (tm, tk)

    # Random inputs (same as reference)
    A = torch.randn(M, K)
    B = torch.randn(M, K)

    # Load A
    a_load = LinearOffChipLoad(
        underlying=A,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=1,
        transposed=False,
    )
    graph.add_node(a_load)

    # Load B
    b_load = LinearOffChipLoad(
        underlying=B,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=1,
        transposed=False,
    )
    graph.add_node(b_load)

    # ------------------------------------------------------------------
    # 3. Element‑wise addition (BinaryMap)
    # ------------------------------------------------------------------
    c_add = BinaryMap(
        graph,
        a_load,
        b_load,
        fn=Add(),
        write_back_mu=False,
        compute_bw=0,
    )

    # ------------------------------------------------------------------
    # 4. Store the result back to off‑chip memory
    # ------------------------------------------------------------------
    store = OffChipStore(
        graph,
        c_add,
        par_dispatch=1,
        store_file_name="output",
    )

    # ------------------------------------------------------------------
    # 5. Finalise broadcast information and return the graph + output node
    # ------------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, store