def build_graph(dims, tensors):
    # Build a STeP graph that implements the routed MoE kernel.
    # All DSL calls have been replaced with their corresponding STeP node constructors.
    # The graph follows the algorithm:
    #   * Load the input tensor and weight matrices.
    #   * For each expert compute gate, up, SiLU activation, projection, and down.
    #   * Apply the per‑expert mask (from expert_onehot) and its soft‑maxed weight,
    #     sum over the active‑expert dimension, and accumulate the contribution.
    #   * Store the final result.
    # Minimal values are used for parameters like `par_dispatch`, `write_back_mu`,
    # and `compute_bw` because they do not affect functional correctness.

    # -----------------------------------------------------------------------
    # Helper to add source nodes (they do not take the graph as a parameter)
    # -----------------------------------------------------------------------
    def _add_source(node):
        graph.add_node(node)
        return node

    # -----------------------------------------------------------------------
    # Create the graph and essential parameters
    # -----------------------------------------------------------------------
    graph = Graph()                     # empty graph instance
    tile_n = dims["tile_n"]
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    # -----------------------------------------------------------------------
    # Load the main input tensor X
    # -----------------------------------------------------------------------
    x = _add_source(
        LinearOffChipLoad(
            underlying=tensors["x"],
            stride=(1,),
            out_shape_tiled=(B,),
            tile_row=1,
            tile_col=D,
            par_dispatch=1,
        )
    )

    # -----------------------------------------------------------------------
    # Metadata nodes for routing information (constants)
    # -----------------------------------------------------------------------
    expert_onehot_f = _add_source(MetadataGen(tensors["expert_onehot"]))
    expert_weights_f = _add_source(MetadataGen(tensors["expert_weights"]))

    # -----------------------------------------------------------------------
    # Initialise output Y to zeros via Multiply‑Immediate with 0.0
    # -----------------------------------------------------------------------
    y = UnaryMap(
        graph,
        x,
        fn=map_fn.MulImmediate(0.0),
        write_back_mu=False,
        compute_bw=0,
    )

    # -----------------------------------------------------------------------
    # Loop over experts and accumulate their contributions
    # -----------------------------------------------------------------------
    for exp_idx in range(n_experts):
        # Load per‑expert weight matrices
        gate_w = _add_source(
            LinearOffChipLoad(
                underlying=tensors["gate_weights"][exp_idx],
                stride=(0,),
                out_shape_tiled=(B,),
                tile_row=D,
                tile_col=F,
                par_dispatch=1,
            )
        )
        up_w = _add_source(
            LinearOffChipLoad(
                underlying=tensors["up_weights"][exp_idx],
                stride=(0,),
                out_shape_tiled=(B,),
                tile_row=D,
                tile_col=F,
                par_dispatch=1,
            )
        )
        down_w = _add_source(
            LinearOffChipLoad(
                underlying=tensors["down_weights"][exp_idx],
                stride=(0,),
                out_shape_tiled=(B,),
                tile_row=F,
                tile_col=D,
                par_dispatch=1,
            )
        )

        # Compute gate and up projections
        gate_out = BinaryMap(
            graph,
            x,
            gate_w,
            fn=map_fn.Matmul(),
            write_back_mu=False,
            compute_bw=0,
        )
        up_out = BinaryMap(
            graph,
            x,
            up_w,
            fn=map_fn.Matmul(),
            write_back_mu=False,
            compute_bw=0,
        )

        # SiLU activation on gate output
        gate_act = UnaryMap(
            graph,
            gate_out,
            fn=map_fn.Silu(),
            write_back_mu=False,
            compute_bw=0,
        )

        # Element‑wise multiplication (gate activation * up projection)
        proj = BinaryMap(
            graph,
            gate_act,
            up_out,
            fn=map_fn.Mul(),
            write_back_mu=False,
            compute_bw=0,
        )

        # Down projection
        down_out = BinaryMap(
            graph,
            proj,
            down_w,
            fn=map_fn.Matmul(),
            write_back_mu=False,
            compute_bw=0,
        )

        # -------------------------------------------------------------------
        # Per‑expert mask: slice the constant expert_onehot tensor for this expert.
        # The slicing is performed on the raw torch tensor (allowed) before
        # creating a MetadataGen source node.
        # -------------------------------------------------------------------
        mask_tensor = tensors["expert_onehot"][:, :, exp_idx]  # shape (B, n_active)
        mask_i = _add_source(MetadataGen(mask_tensor))

        # Apply the soft‑maxed expert weight to the mask (element‑wise multiply)
        weighted_mask = BinaryMap(
            graph,
            expert_weights_f,
            mask_i,
            fn=map_fn.Mul(),
            write_back_mu=False,
            compute_bw=0,
        )

        # Sum over the active‑expert dimension (rank=1)
        weight_sum = Accum(
            graph,
            weighted_mask,
            weighted_mask.stream.stream_dtype,
            fn=accum_fn.Add(),
            init_fn=None,
            accum_rank=1,
            write_back_mu=False,
            compute_bw=0,
        )

        # Multiply down projection by the summed weight and add to the running output
        weighted_contrib = BinaryMap(
            graph,
            down_out,
            weight_sum,
            fn=map_fn.Mul(),
            write_back_mu=False,
            compute_bw=0,
        )
        y = BinaryMap(
            graph,
            y,
            weighted_contrib,
            fn=map_fn.Add(),
            write_back_mu=False,
            compute_bw=0,
        )

    # -----------------------------------------------------------------------
    # Store the final result
    # -----------------------------------------------------------------------
    output = OffChipStore(graph, y, par_dispatch=1)

    # Insert any missing broadcast nodes and return the graph
    graph = infer_broadcast(graph)
    return graph, output