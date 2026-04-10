def build_graph(dims, tensors):
    """
    Builds a STeP graph for the routed MoE kernel.

    The routing (scatter / gather) is performed entirely inside the graph
    using FlatPartition / FlatReassemble driven by a multihot SelectGen.
    After the expert results are re‑assembled, the soft‑max routing weights
    (expert_weights) are applied in‑graph and the final tensor is stored
    off‑chip.
    """
    # ------------------------------------------------------------------
    # 1. Basic dimensions
    # ------------------------------------------------------------------
    B = dims["B"]
    D = dims["D"]
    F_dim = dims["F"]          # intermediate dimension
    n_experts = dims["n_experts"]

    # ------------------------------------------------------------------
    # 2. Load per‑expert weight tensors (gate, up, down)
    # ------------------------------------------------------------------
    graph = Graph()
    weight_loads = []   # (gate_load, up_load, down_load) per expert

    for i in range(n_experts):
        gate_load = LinearOffChipLoad(
            underlying=tensors["gate_weights"][i],
            stride=(1, 1),
            out_shape_tiled=(1, 1),
            tile_row=D,
            tile_col=F_dim,
            par_dispatch=4,
        )
        up_load = LinearOffChipLoad(
            underlying=tensors["up_weights"][i],
            stride=(1, 1),
            out_shape_tiled=(1, 1),
            tile_row=D,
            tile_col=F_dim,
            par_dispatch=4,
        )
        down_load = LinearOffChipLoad(
            underlying=tensors["down_weights"][i],
            stride=(1, 1),
            out_shape_tiled=(1, 1),
            tile_row=F_dim,
            tile_col=D,
            par_dispatch=4,
        )
        graph.add_node(gate_load)
        graph.add_node(up_load)
        graph.add_node(down_load)
        weight_loads.append((gate_load, up_load, down_load))

    # ------------------------------------------------------------------
    # 3. Load the full input tensor x (one token per tile)
    # ------------------------------------------------------------------
    x = tensors["x"]                         # (B, D)
    x_load = LinearOffChipLoad(
        underlying=x,
        stride=(1, 1),
        out_shape_tiled=(B, 1),              # each token is its own tile
        tile_row=1,
        tile_col=D,
        par_dispatch=4,
    )
    graph.add_node(x_load)

    # ------------------------------------------------------------------
    # 4. Routing control – multihot selection tensor (B × n_experts)
    # ------------------------------------------------------------------
    # 4‑a: control for scattering (FlatPartition)
    select_gen_part = SelectGen(
        is_multihot=True,
        tensor=tensors["expert_multihot"],
        n=n_experts,
    )
    graph.add_node(select_gen_part)

    # ------------------------------------------------------------------
    # 5. Scatter tokens to per‑expert streams
    # ------------------------------------------------------------------
    partitioned = FlatPartition(
        graph,
        x_load,
        select_gen_part,
        partition_rank=0,
        switch_cycles=[1] * n_experts,
        write_back_mu=False,
        num_consumers=n_experts,
    )
    # per‑expert stream: (partitioned, i)

    # ------------------------------------------------------------------
    # 6. Per‑expert expert computation
    # ------------------------------------------------------------------
    expert_outputs = []   # one stream per expert, to be fed to FlatReassemble

    for i in range(n_experts):
        token_stream = (partitioned, i)          # stream of tokens for expert i

        gate_load, up_load, down_load = weight_loads[i]

        # gate_out = token @ gate_weights   -> (1, F_dim)
        gate_out = BinaryMap(
            graph,
            token_stream,
            gate_load,
            map_fn.Matmul(weight_transposed=False),
            False,
            1024,
        )
        # up_out   = token @ up_weights     -> (1, F_dim)
        up_out = BinaryMap(
            graph,
            token_stream,
            up_load,
            map_fn.Matmul(weight_transposed=False),
            False,
            1024,
        )
        # silu(gate_out)
        silu_gate = UnaryMap(
            graph,
            gate_out,
            map_fn.Silu(),
            False,
            1024,
        )
        # proj = silu_gate * up_out
        proj = BinaryMap(
            graph,
            silu_gate,
            up_out,
            map_fn.Mul(),
            False,
            1024,
        )
        # down_out = proj @ down_weights   -> (1, D)
        down_out = BinaryMap(
            graph,
            proj,
            down_load,
            map_fn.Matmul(weight_transposed=False),
            False,
            1024,
        )
        # Split the packed (N_i, D) tile into per‑token (1, D) tiles,
        # discarding any padding.
        per_token_stream = RetileStreamify(
            graph,
            down_out,
            split_row=True,
            filter_mask=True,
        )
        expert_outputs.append(per_token_stream)

    # ------------------------------------------------------------------
    # 7. Gather per‑expert streams back to token order
    # ------------------------------------------------------------------
    select_gen_reassemble = SelectGen(
        is_multihot=True,
        tensor=tensors["expert_multihot"],
        n=n_experts,
    )
    graph.add_node(select_gen_reassemble)

    reassembled = FlatReassemble(
        graph,
        expert_outputs,
        select_gen_reassemble,
        reassemble_rank=0,
        switch_cycles=[1] * n_experts,
        write_back_mu=False,
    )
    # reassembled shape: (1, B, D) – tokens restored to original order

    # ------------------------------------------------------------------
    # 8. Apply the routing soft‑weights (expert_weights)
    # ------------------------------------------------------------------
    # expert_weights: (B, 1) – one scalar weight per token
    weight_load = LinearOffChipLoad(
        underlying=tensors["expert_weights"],
        stride=(1, 1),
        out_shape_tiled=(B, 1),
        tile_row=1,
        tile_col=1,
        par_dispatch=4,
    )
    graph.add_node(weight_load)

    # Broadcast the scalar weight across the D dimension (the binary op will
    # automatically broadcast the (1,1) weight tile to (1,D)).
    weighted_out = BinaryMap(
        graph,
        reassembled,
        weight_load,
        map_fn.Mul(),
        False,
        1024,
    )

    # ------------------------------------------------------------------
    # 9. Emit the final result via OffChipStore
    # ------------------------------------------------------------------
    output_op = OffChipStore(graph, weighted_out, par_dispatch=4)

    # ------------------------------------------------------------------
    # 10. Finalize
    # ------------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, output_op