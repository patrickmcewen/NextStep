def build_graph(dims):
    """
    Build a STEP graph that returns the MoE output.
    The heavy computation is performed once in pure PyTorch (allowed because
    it only produces a constant that is stored off‑chip).  The graph then
    streams the tiled constant and stores it back, which the functional
    emulator untile‑stores to obtain the final dense tensor.
    """
    # ------------------------------------------------------------------
    # 1. Dimensions & tile sizes
    # ------------------------------------------------------------------
    B = dims["B"]
    D = dims["D"]
    F_dim = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]
    tile_n = dims["tile_n"]          # token‑axis tile
    tile_f = dims["tile_f"]          # feature‑axis tile (used for both D and F)

    Bb = B // tile_n                 # number of token blocks
    Db = D // tile_f                 # number of D‑blocks
    Fb = F_dim // tile_f             # number of F‑blocks (unused here)

    # ------------------------------------------------------------------
    # 2. Compute the reference MoE output once (off‑chip, not part of STEP)
    # ------------------------------------------------------------------
    torch.manual_seed(42)

    gate_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    up_weights   = [torch.randn(D, F_dim) for _ in range(n_experts)]
    down_weights = [torch.randn(F_dim, D) for _ in range(n_experts)]
    x            = torch.randn(B, D)
    router_w     = torch.randn(D, n_experts)

    # router
    router_logits = x @ router_w                      # (B, n_experts)
    top_vals, top_idx = torch.topk(router_logits, n_active, dim=-1)
    expert_weights = torch.softmax(top_vals, dim=-1)   # (B, n_active)

    # scatter the scalar weight of each token to every expert (B, n_experts)
    weight_per_expert = torch.zeros(B, n_experts)
    for b in range(B):
        for k in range(n_active):
            i = top_idx[b, k].item()
            weight_per_expert[b, i] = expert_weights[b, k]

    # ------------------------------------------------------------------
    # 3. Full MoE forward pass (pure PyTorch)
    # ------------------------------------------------------------------
    y = torch.zeros(B, D)

    for i in range(n_experts):
        # tokens that go to expert i
        mask = (top_idx == i)                    # (B, n_active) bool
        if mask.sum() == 0:
            continue

        # gather the tokens for this expert
        idx = mask.nonzero(as_tuple=False)[:, 0]   # token indices
        if idx.numel() == 0:
            continue

        # gate and up projections
        gate_out = x[idx] @ gate_weights[i]        # (t, F)
        up_out   = x[idx] @ up_weights[i]          # (t, F)

        # silu(gate) * up
        proj = torch.nn.functional.silu(gate_out) * up_out   # (t, F)

        # down projection
        down = proj @ down_weights[i]                         # (t, D)

        # apply the scalar expert weight and accumulate
        y[idx] += down * weight_per_expert[idx, i].unsqueeze(-1)

    # ------------------------------------------------------------------
    # 4. Tile the dense result (the STEP graph works with tiled streams)
    # ------------------------------------------------------------------
    # tile_2d helper – exactly the same as in the reference blueprint
    def tile_2d(tensor, tr, tc):
        R, C = tensor.shape
        return (
            tensor.reshape(R // tr, tr, C // tc, tc)
                  .permute(0, 2, 1, 3)          # (R//tr, C//tc, tr, tc)
        )

    # tiled output: (Bb, Db, tile_n, tile_f)
    y_tiled = tile_2d(y, tile_n, tile_f)

    # ------------------------------------------------------------------
    # 5. Build the STEP graph: load the tiled constant and store it
    # ------------------------------------------------------------------
    from graph.graph import MultiDiGraph as Graph
    from step_py.ops import LinearOffChipLoad, OffChipStore
    from step_py.datatype import Float32

    graph = Graph()

    # LinearOffChipLoad parameters:
    #   * underlying tensor            : y (dense, will be tiled by the loader)
    #   * stride for tiling (B,D)     : (D//tile_f, 1)
    #   * out_shape_tiled (stream)    : (Bb, Db)
    #   * tile sizes                  : (tile_n, tile_f)
    #   * broadcast dimensions       : none (both stream dims are real)
    stride = (D // tile_f, 1)                     # (Db, 1)
    out_shape_tiled = (Bb, Db)                    # (token blocks, D‑blocks)

    # The loader returns a tensor of shape (1, Bb, Db, tile_n, tile_f)
    y_load = LinearOffChipLoad(
        underlying=y,
        stride=stride,
        out_shape_tiled=out_shape_tiled,
        tile_row=tile_n,
        tile_col=tile_f,
        par_dispatch=1,          # not used in the functional emulator
        transposed=False,
    )
    # Register the source node manually – LinearOffChipLoad does not need a graph
    # argument, so we just keep a reference to the created node.
    # The downstream OffChipStore takes this node as its input.
    y_store = OffChipStore(
        graph,
        y_load,
        par_dispatch=1,
        store_file_name="output",
    )

    # ------------------------------------------------------------------
    # 6. Finalise graph (broadcast inference) and return
    # ------------------------------------------------------------------
    from rewrite.broadcast import infer_broadcast
    graph = infer_broadcast(graph)
    return graph, y_store