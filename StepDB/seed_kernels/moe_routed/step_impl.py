"""STeP implementation: Routed MoE with top-k expert selection.

Full MoE layer with routing via FlatPartition/FlatReassemble:
1. Load input [B, D], generate multihot routing
2. FlatPartition routes tokens to n_experts streams
3. Per-expert: Reshape → Flatten → Accum(RetileRow) to tile tokens
4. Per-expert: gate/up matmul, silu, mul, down matmul (BinaryMapAccum)
5. Per-expert: RetileStreamify to untile, weight scaling
6. FlatReassemble merges expert outputs
7. Accum(Add) sums weighted contributions

Based on step_tl/end_to_end/moe/static_no_timemultiplex.py.
"""

SEED = 42


def build_graph(dims):
    B = dims["B"]
    D = dims["D"]
    F_dim = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]
    tile_n = dims.get("tile_n", B)
    tile_f = dims.get("tile_f", F_dim)

    assert B % tile_n == 0 or tile_n >= B, f"B={B} not compatible with tile_n={tile_n}"
    assert F_dim % tile_f == 0, f"F={F_dim} not divisible by tile_f={tile_f}"

    torch.manual_seed(SEED)

    # Create per-expert weights — must match reference RNG order
    gate_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    up_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    down_weights = [torch.randn(F_dim, D) for _ in range(n_experts)]

    # Input tokens
    x = torch.randn(B, D)

    # Router
    router_w = torch.randn(D, n_experts)
    router_logits = x @ router_w  # [B, n_experts]
    expert_weights_raw, expert_indices = torch.topk(router_logits, n_active, dim=-1)
    expert_weights_tensor = torch.softmax(expert_weights_raw, dim=-1)  # [B, n_active]

    # Build multihot: [B, n_experts] — which experts each token uses
    expert_multihot = torch.zeros(B, n_experts, dtype=torch.int64)
    for b in range(B):
        for k in range(n_active):
            expert_multihot[b, expert_indices[b, k]] = 1

    # Build onehot: [B, n_active, n_experts] — per-weight-slot routing
    expert_onehot = torch.zeros(B, n_active, n_experts, dtype=torch.int64)
    for b in range(B):
        for k in range(n_active):
            expert_onehot[b, k, expert_indices[b, k]] = 1

    step_graph = Graph()

    # Stage 1: Load input [B, D] as stream [B] of tiles [1, D]
    input_load = LinearOffChipLoad(
        underlying=x,
        stride=(1,),
        out_shape_tiled=(B,),
        tile_row=1,
        tile_col=D,
        par_dispatch=4,
    )

    # Stage 2: Generate selection streams
    feature_select_gen = SelectGen(
        is_multihot=True, tensor=expert_multihot, n=n_experts,
    )
    weight_select_gen = SelectGen(
        is_multihot=True, tensor=expert_onehot, n=n_experts,
    )

    # Stage 3: Load expert weights [B, n_active]
    weights_load = LinearOffChipLoad(
        underlying=expert_weights_tensor,
        stride=(n_active, 1),
        out_shape_tiled=(B, n_active),
        tile_row=1,
        tile_col=1,
        par_dispatch=4,
    )

    # Stage 4: Partition input to experts
    # [1, B] -> [Dyn] x n_experts (tile: [1, D])
    partitioned = FlatPartition(
        step_graph,
        input_load,
        feature_select_gen,
        partition_rank=0,
        switch_cycles=[1 for _ in range(n_experts)],
        write_back_mu=False,
        num_consumers=n_experts,
    )

    # Stage 4b: Reshape + Flatten + Accum to tile tokens
    # [Dyn] -> [dyn1, ceil(Dyn/tile_n)*tile_n] -> Accum -> [ceil] of [tile_n, D]
    expert_feature_streams = []
    for i in range(n_experts):
        reshaped = Reshape(
            step_graph,
            (partitioned, i),
            tile_n,
            0,
            write_back_mu=False,
            add_outer_dim=True,
            pad_fn=init_fn.Zero(shape=(1, D), dtype=Float32()),
        )
        flattened = Flatten(step_graph, reshaped, min_rank=1, max_rank=2)
        tiled = Accum(
            step_graph,
            flattened,
            Tile(tile_dtype=Float32(), shape=(tile_n, D)),
            accum_fn.RetileRow(),
            init_fn.Empty(shape=(0, D), dtype=Float32()),
            1,
            False,
            1024,
        )
        expert_feature_streams.append(tiled)

    # Stage 5: Repeat for weight tiling
    repeated_features = [
        RepeatStatic(step_graph, expert_feature_streams[i], repeat_factor=F_dim // tile_f)
        for i in range(n_experts)
    ]

    # Stage 6: Load up weights per expert via LinearOffChipLoadRef
    up_loads = [
        LinearOffChipLoadRef(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=up_weights[i],
            stride=(1, 1),
            out_shape_tiled=(F_dim // tile_f, 1),
            tile_row=D,
            tile_col=tile_f,
            par_dispatch=4,
        )
        for i in range(n_experts)
    ]
    ready_up_loads = [
        Flatten(graph=step_graph, input=up_loads[i], min_rank=0, max_rank=1)
        for i in range(n_experts)
    ]

    # Stage 7: Up matmul
    up_features = [
        BinaryMap(step_graph, repeated_features[i], ready_up_loads[i],
                  map_fn.Matmul(weight_transposed=False), False, 1024)
        for i in range(n_experts)
    ]

    # Stage 8: Load gate weights
    gate_loads = [
        LinearOffChipLoadRef(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=gate_weights[i],
            stride=(1, 1),
            out_shape_tiled=(F_dim // tile_f, 1),
            tile_row=D,
            tile_col=tile_f,
            par_dispatch=4,
        )
        for i in range(n_experts)
    ]
    ready_gate_loads = [
        Flatten(graph=step_graph, input=gate_loads[i], min_rank=0, max_rank=1)
        for i in range(n_experts)
    ]

    # Stage 8b: Gate matmul
    gate_features = [
        BinaryMap(step_graph, repeated_features[i], ready_gate_loads[i],
                  map_fn.Matmul(weight_transposed=False), False, 1024)
        for i in range(n_experts)
    ]

    # Stage 9: SiLU activation
    gate_activated = [
        UnaryMap(graph=step_graph, input=gate_features[i],
                 fn=map_fn.Silu(), write_back_mu=False, compute_bw=1024)
        for i in range(n_experts)
    ]

    # Stage 10: gate * up
    projected = [
        BinaryMap(step_graph, up_features[i], gate_activated[i],
                  map_fn.Mul(), False, 1024)
        for i in range(n_experts)
    ]

    # Stage 11: Load down weights
    down_loads = [
        LinearOffChipLoadRef(
            graph=step_graph,
            ref=expert_feature_streams[i],
            underlying=down_weights[i],
            stride=(D // D, 1),
            out_shape_tiled=(F_dim // tile_f, D // D),
            tile_row=tile_f,
            tile_col=D,
            par_dispatch=4,
        )
        for i in range(n_experts)
    ]
    ready_down_loads = [
        Flatten(graph=step_graph, input=down_loads[i], min_rank=0, max_rank=1)
        for i in range(n_experts)
    ]

    # Stage 12: Down matmul (accumulated)
    down_features = [
        BinaryMapAccum(
            step_graph, projected[i], ready_down_loads[i],
            map_accum_fn.Matmul(weight_transposed=False),
            init_fn.Zero(shape=(tile_n, D), dtype=Float32()),
            1, False, 1024,
        )
        for i in range(n_experts)
    ]

    # Stage 12.5: Retile back to [1, D] per token
    retiled = [
        RetileStreamify(
            graph=step_graph,
            input=down_features[i],
            split_row=True,
            filter_mask=True,
        )
        for i in range(n_experts)
    ]

    # Fix dynamic dim shapes
    for partitioned_stream, retiled_stream in zip(partitioned.stream_list, retiled):
        dyn_i = partitioned_stream.shape[0]
        retiled_stream.stream.shape = (dyn_i,)

    # Stage 13: Partition expert weights
    expert_weight_streams = FlatPartition(
        step_graph, weights_load, weight_select_gen,
        partition_rank=0,
        switch_cycles=[1 for _ in range(n_experts)],
        write_back_mu=False,
        num_consumers=n_experts,
    )

    # Stage 14: Weight scaling
    weighted = [
        BinaryMap(step_graph, (expert_weight_streams, i), retiled[i],
                  map_fn.Mul(), False, 1024)
        for i in range(n_experts)
    ]

    # Stage 15: Reassemble
    feature_select_gen_reassemble = SelectGen(
        is_multihot=True, tensor=expert_multihot, n=n_experts,
    )
    reassembled = FlatReassemble(
        step_graph, weighted, feature_select_gen_reassemble,
        reassemble_rank=0,
        switch_cycles=[1 for _ in range(n_experts)],
        write_back_mu=False,
    )

    # Stage 16: Accumulate weighted expert outputs
    accumed = Accum(
        step_graph, reassembled,
        Tile(tile_dtype=Float32(), shape=(1, D)),
        accum_fn.Add(),
        init_fn.Zero(shape=(1, D), dtype=Float32()),
        1, False, 1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=accumed,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
