def build_graph(dims, tensors):
    # ------------------------------------------------------------------
    # 1️⃣  Extract dimensions
    # ------------------------------------------------------------------
    B = dims["B"]                # batch (tokens)
    D = dims["D"]                # hidden dimension
    F_dim = dims["F"]            # intermediate dimension
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]
    tile_n = dims["tile_n"]      # size of a batch tile (stream granularity)

    # ------------------------------------------------------------------
    # 2️⃣  Grab tensors
    # ------------------------------------------------------------------
    x = tensors["x"]                                 # (B, D)
    gate_ws = tensors["gate_weights"]                # list length n_experts, each (D, F_dim)
    up_ws   = tensors["up_weights"]                  # list length n_experts, each (D, F_dim)
    down_ws = tensors["down_weights"]                # list length n_experts, each (F_dim, D)

    expert_onehot  = tensors["expert_onehot"].to(x.dtype)   # (B, n_active, n_experts)
    expert_weights = tensors["expert_weights"]              # (B, n_active)

    # ------------------------------------------------------------------
    # 3️⃣  Compute per‑token‑/‑expert scalar weight (host‑side only, no torch.zeros)
    # ------------------------------------------------------------------
    # weight_per_expert[b, e] = Σ_a expert_onehot[b,a,e] * expert_weights[b,a]
    # Initialise with the contribution from the first active slot, then accumulate.
    weight_per_expert = (
        expert_weights[:, 0].unsqueeze(-1) * expert_onehot[:, 0, :]
    )  # (B, n_experts)
    for a in range(1, n_active):
        weight_per_expert = weight_per_expert + (
            expert_weights[:, a].unsqueeze(-1) * expert_onehot[:, a, :]
        )  # (B, n_experts)

    # ------------------------------------------------------------------
    # 4️⃣  Build the STeP graph
    # ------------------------------------------------------------------
    graph = Graph()

    # ------------------------------------------------------------------
    # 4.1  Load the token matrix `x` – tiled over the batch dimension.
    # ------------------------------------------------------------------
    batch_tiles = B // tile_n                      # number of batch tiles
    x_load = LinearOffChipLoad(
        x,
        stride=(1, 0),                     # advance one batch tile per step, no column stride
        out_shape_tiled=(batch_tiles, 1), # (batch_tiles, 1) stream shape
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=1,
        transposed=False,
    )
    graph.add_node(x_load)

    # ------------------------------------------------------------------
    # 4.2  Placeholder for the accumulating output tensor.
    # ------------------------------------------------------------------
    y_node = None   # will hold the graph node that accumulates expert contributions

    # ------------------------------------------------------------------
    # 4.3  Per‑expert sub‑graph (no padding logic – tile size matches granularity).
    # ------------------------------------------------------------------
    for i in range(n_experts):
        # ----- Load per‑expert weights (broadcast across batch tiles) -------------
        gate_w_load = LinearOffChipLoad(
            gate_ws[i],
            stride=(0, 0),                     # broadcast across batch tiles
            out_shape_tiled=(batch_tiles, 1),
            tile_row=D,
            tile_col=F_dim,
            par_dispatch=1,
            transposed=False,
        )
        graph.add_node(gate_w_load)

        up_w_load = LinearOffChipLoad(
            up_ws[i],
            stride=(0, 0),
            out_shape_tiled=(batch_tiles, 1),
            tile_row=D,
            tile_col=F_dim,
            par_dispatch=1,
            transposed=False,
        )
        graph.add_node(up_w_load)

        down_w_load = LinearOffChipLoad(
            down_ws[i],
            stride=(0, 0),
            out_shape_tiled=(batch_tiles, 1),
            tile_row=F_dim,
            tile_col=D,
            par_dispatch=1,
            transposed=False,
        )
        graph.add_node(down_w_load)

        # ----- matmul: x @ gate_w -------------------------------------------------
        gate_out = BinaryMap(
            graph,
            x_load,
            gate_w_load,
            map_fn.Matmul(weight_transposed=False),
            False,
            1024,
        )   # (batch_tile, F_dim)

        # ----- matmul: x @ up_w ---------------------------------------------------
        up_out = BinaryMap(
            graph,
            x_load,
            up_w_load,
            map_fn.Matmul(weight_transposed=False),
            False,
            1024,
        )   # (batch_tile, F_dim)

        # ----- silu activation ----------------------------------------------------
        silu_out = UnaryMap(
            graph,
            gate_out,
            map_fn.Silu(),
            False,
            1024,
        )   # (batch_tile, F_dim)

        # ----- projection (element‑wise multiply) ---------------------------------
        proj = BinaryMap(
            graph,
            silu_out,
            up_out,
            map_fn.Mul(),
            False,
            1024,
        )   # (batch_tile, F_dim)

        # ----- matmul: proj @ down_w ---------------------------------------------
        down_out = BinaryMap(
            graph,
            proj,
            down_w_load,
            map_fn.Matmul(weight_transposed=False),
            False,
            1024,
        )   # (batch_tile, D)

        # ----- Load scalar routing weight for this expert and expand to D ---------
        #   weight_per_expert[:, i] is (B,)
        weight_vec = weight_per_expert[:, i].unsqueeze(-1)            # (B,1)
        weight_expanded = weight_vec.expand(-1, D)                    # (B, D)
        weight_load = LinearOffChipLoad(
            weight_expanded,
            stride=(1, 0),                     # tile over batch the same way as x
            out_shape_tiled=(batch_tiles, 1),
            tile_row=tile_n,
            tile_col=D,
            par_dispatch=1,
            transposed=False,
        )
        graph.add_node(weight_load)

        # ----- Apply scalar weight (element‑wise multiply) ------------------------
        weighted = BinaryMap(
            graph,
            down_out,
            weight_load,
            map_fn.Mul(),
            False,
            1024,
        )   # (batch_tile, D)

        # ----- Accumulate into the final output -----------------------------------
        if y_node is None:
            y_node = weighted
        else:
            y_node = BinaryMap(
                graph,
                y_node,
                weighted,
                map_fn.Add(),
                False,
                1024,
            )

    # ------------------------------------------------------------------
    # 5️⃣  Store the result off‑chip (the emulator will untile it)
    # ------------------------------------------------------------------
    output_op = OffChipStore(graph, y_node, par_dispatch=1)
    graph.add_node(output_op)

    # ------------------------------------------------------------------
    # 6️⃣  Finalise broadcast metadata and return
    # ------------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, output_op