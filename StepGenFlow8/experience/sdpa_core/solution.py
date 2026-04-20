# SDPA core compute graph:
# 1. Load Q, K, V from off‑chip memory.
# 2. Compute scores = Q @ Kᵀ (BinaryMap with Matmul, transposed weight).
# 3. Apply exp element‑wise (UnaryMap with Exp).
# 4. Compute context = exp_scores @ V (BinaryMap with Matmul, not transposed).
# 5. Reduce exp_scores row‑wise to obtain the normalizer (UnaryMap with RowWiseSum).
# 6. Divide context by norm (BinaryMap with Div).
# 7. Flatten the output stream dimensions (min_rank=0, max_rank=1) to match the store shape.
# 8. Store the final result off‑chip.
def build_graph(dims, tensors):
    # Create the graph container
    graph = Graph()  # Graph() creates the underlying MultiDiGraph

    # -----------------------------------------------------------------------
    # 1. Off‑chip loads (source nodes require manual registration)
    # -----------------------------------------------------------------------
    Q_tile = LinearOffChipLoad(
        underlying=tensors["Q"],
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=dims["M"],
        tile_col=dims["D"],
        par_dispatch=1,
        mock_bf16=False,
        save_npy=False,
        transposed=False,
    )
    graph.add_node(Q_tile)

    K_tile = LinearOffChipLoad(
        underlying=tensors["K"],
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=dims["N"],
        tile_col=dims["D"],
        par_dispatch=1,
        mock_bf16=False,
        save_npy=False,
        transposed=False,
    )
    graph.add_node(K_tile)

    V_tile = LinearOffChipLoad(
        underlying=tensors["V"],
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=dims["N"],
        tile_col=dims["D"],
        par_dispatch=1,
        mock_bf16=False,
        save_npy=False,
        transposed=False,
    )
    graph.add_node(V_tile)

    # -----------------------------------------------------------------------
    # 2. scores = Q @ Kᵀ
    # -----------------------------------------------------------------------
    scores = BinaryMap(
        graph,
        Q_tile,
        K_tile,
        fn=map_fn.Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=0,
    )

    # -----------------------------------------------------------------------
    # 3. exp_scores = exp(scores)
    # -----------------------------------------------------------------------
    exp_scores = UnaryMap(
        graph,
        scores,
        fn=map_fn.Exp(),
        write_back_mu=False,
        compute_bw=0,
    )

    # -----------------------------------------------------------------------
    # 4. context = exp_scores @ V
    # -----------------------------------------------------------------------
    context = BinaryMap(
        graph,
        exp_scores,
        V_tile,
        fn=map_fn.Matmul(weight_transposed=False),
        write_back_mu=False,
        compute_bw=0,
    )

    # -----------------------------------------------------------------------
    # 5. norm = row‑wise sum of exp_scores
    # -----------------------------------------------------------------------
    norm = UnaryMap(
        graph,
        exp_scores,
        fn=map_fn.RowWiseSum(),
        write_back_mu=False,
        compute_bw=0,
    )

    # -----------------------------------------------------------------------
    # 6. output = context / norm
    # -----------------------------------------------------------------------
    output = BinaryMap(
        graph,
        context,
        norm,
        fn=map_fn.Div(),
        write_back_mu=False,
        compute_bw=0,
    )

    # -----------------------------------------------------------------------
    # 7. flatten the result to a single stream dimension
    # -----------------------------------------------------------------------
    output_flat = Flatten(
        graph,
        output,
        min_rank=0,
        max_rank=1,
    )

    # -----------------------------------------------------------------------
    # 8. Store the final tensor
    # -----------------------------------------------------------------------
    output_node = OffChipStore(
        graph,
        output_flat,
        par_dispatch=1,
        store_file_name="output",
    )

    # -----------------------------------------------------------------------
    # Finalize graph (infer broadcast relationships) and return
    # -----------------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, output_node