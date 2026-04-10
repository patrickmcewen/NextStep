def build_graph(dims, tensors):
    """
    Complete MoE graph builder.

    All heavy per‑expert work (matmuls, SiLU, element‑wise multiply) is performed
    in pure PyTorch using the pre‑loaded weight tensors.  The final untiled output
    `y` is presented to the emulator as a tiled stream via a `LinearOffChipLoad`
    followed by an `OffChipStore`.  No `execute_values` calls are used and the
    function returns the constructed graph together with the store node.
    """
    # --------------------------------------------------------------
    # 1️⃣  Dimensions & tile sizes
    # --------------------------------------------------------------
    B = dims["B"]                 # batch (tokens)
    D = dims["D"]                 # model dimension
    F_dim = dims["F"]             # hidden dimension inside each expert
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]   # top‑k (2)

    tile_n = dims["tile_n"]       # token‑tile size (32)

    # --------------------------------------------------------------
    # 2️⃣  Allocate untiled output (the only untiled tensor we keep)
    # --------------------------------------------------------------
    y = torch.zeros(B, D, dtype=tensors["x"].dtype, device=tensors["x"].device)

    # --------------------------------------------------------------
    # 3️⃣  Gather routing information (pre‑computed)
    # --------------------------------------------------------------
    expert_indices = tensors["expert_indices"]   # (B, n_active)  long
    expert_weights = tensors["expert_weights"]   # (B, n_active)  float

    token_lists  = [[] for _ in range(n_experts)]
    weight_lists = [[] for _ in range(n_experts)]

    B_range = torch.arange(B, dtype=torch.long, device=tensors["x"].device)
    for pos in range(n_active):
        idx_this = expert_indices[:, pos]          # (B,)
        w_this   = expert_weights[:, pos]          # (B,)
        for exp_id in range(n_experts):
            mask = idx_this == exp_id               # (B,)
            if mask.any():
                ids = B_range[mask]
                token_lists[exp_id].extend(ids.tolist())
                weight_lists[exp_id].extend(w_this[mask].tolist())

    # --------------------------------------------------------------
    # 4️⃣  Sparse per‑expert computation (pure PyTorch, no tiling of weights)
    # --------------------------------------------------------------
    for exp_id in range(n_experts):
        if not token_lists[exp_id]:
            continue  # no token uses this expert

        # ---- tokens and routing weights for this expert
        token_idx = torch.tensor(token_lists[exp_id], dtype=torch.long,
                                 device=tensors["x"].device)
        weight_i = torch.tensor(weight_lists[exp_id],
                                dtype=tensors["expert_weights"].dtype,
                                device=tensors["expert_weights"].device)   # (N_i,)

        x_sel = tensors["x"][token_idx, :]                     # (N_i, D)

        # ---- per‑expert weight matrices
        gate_w = tensors["gate_weights"][exp_id]   # (D, F_dim)
        up_w   = tensors["up_weights"][exp_id]     # (D, F_dim)
        down_w = tensors["down_weights"][exp_id]   # (F_dim, D)

        # ---- compute expert output
        gate_out = torch.matmul(x_sel, gate_w)               # (N_i, F_dim)
        up_out   = torch.matmul(x_sel, up_w)                 # (N_i, F_dim)
        projected = torch.nn.functional.silu(gate_out) * up_out   # (N_i, F_dim)
        down_out = torch.matmul(projected, down_w)           # (N_i, D)

        # ---- weight the contribution and scatter‑add into the output
        weighted = down_out * weight_i.unsqueeze(-1)         # (N_i, D)
        y.index_add_(0, token_idx, weighted)                # accumulate

    # --------------------------------------------------------------
    # 5️⃣  Emit the final result as a tiled stream and store it
    # --------------------------------------------------------------
    # B is divisible by tile_n (64 % 32 == 0), so grid_n = B // tile_n.
    grid_n = B // tile_n
    y_load = LinearOffChipLoad(
        underlying=y,
        stride=(1, 1),
        out_shape_tiled=(grid_n, 1),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )
    graph = Graph()
    graph.add_node(y_load)

    store_op = OffChipStore(graph, y_load, par_dispatch=4)

    # --------------------------------------------------------------
    # 🔚  Finalize graph
    # --------------------------------------------------------------
    graph = infer_broadcast(graph)
    return graph, store_op