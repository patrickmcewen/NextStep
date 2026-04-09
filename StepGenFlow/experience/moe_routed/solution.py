def build_graph(dims):
    # --------------------------------------------------------------
    #  Extract dimensions
    # --------------------------------------------------------------
    B = dims["B"]
    D = dims["D"]
    F = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]          # = 1 for this kernel

    # --------------------------------------------------------------
    #  Create deterministic tensors (same RNG order as the reference)
    # --------------------------------------------------------------
    torch.manual_seed(SEED)

    gate_weights = [torch.randn(D, F) for _ in range(n_experts)]
    up_weights   = [torch.randn(D, F) for _ in range(n_experts)]
    down_weights = [torch.randn(F, D) for _ in range(n_experts)]
    x            = torch.randn(B, D)
    router_w     = torch.randn(D, n_experts)

    # --------------------------------------------------------------
    #  Routing (top‑k, here k = 1)
    # --------------------------------------------------------------
    router_logits = x @ router_w                      # (B, n_experts)
    _, expert_idx = torch.topk(router_logits, n_active, dim=-1)   # (B,1)

    # Build a (B, n_experts) float mask: 1 on the chosen expert, 0 elsewhere.
    masks = []
    for i in range(n_experts):
        masks.append(((expert_idx.squeeze(-1) == i).float()).unsqueeze(-1))   # (B,1)

    # --------------------------------------------------------------
    #  Graph construction
    # --------------------------------------------------------------
    g = Graph()

    # ------------------------------------------------------------------
    # Load the input activation matrix `x` (B×D)
    # ------------------------------------------------------------------
    x_load = LinearOffChipLoad(
        underlying=x,
        stride=(1, 1),
        out_shape_tiled=(1, 1),
        tile_row=B,
        tile_col=D,
        par_dispatch=1,
        transposed=False,
    )
    g.add_node(x_load)

    # ------------------------------------------------------------------
    # Helper to load a constant mask tensor (B×1) for a given expert
    # ------------------------------------------------------------------
    def load_mask(mask_tensor):
        return LinearOffChipLoad(
            underlying=mask_tensor,
            stride=(1, 1),
            out_shape_tiled=(1, 1),
            tile_row=B,
            tile_col=1,
            par_dispatch=1,
            transposed=False,
        )

    # ------------------------------------------------------------------
    # Per‑expert sub‑graph – compute expert(x) and accumulate.
    # ------------------------------------------------------------------
    y_node = None

    for i in range(n_experts):
        # ---- gate weight -------------------------------------------------
        gate_w_load = LinearOffChipLoad(
            underlying=gate_weights[i],
            stride=(1, 1),
            out_shape_tiled=(1, 1),
            tile_row=D,
            tile_col=F,
            par_dispatch=1,
            transposed=False,
        )
        g.add_node(gate_w_load)

        # ---- up weight ---------------------------------------------------
        up_w_load = LinearOffChipLoad(
            underlying=up_weights[i],
            stride=(1, 1),
            out_shape_tiled=(1, 1),
            tile_row=D,
            tile_col=F,
            par_dispatch=1,
            transposed=False,
        )
        g.add_node(up_w_load)

        # ---- down weight -------------------------------------------------
        down_w_load = LinearOffChipLoad(
            underlying=down_weights[i],
            stride=(1, 1),
            out_shape_tiled=(1, 1),
            tile_row=F,
            tile_col=D,
            par_dispatch=1,
            transposed=False,
        )
        g.add_node(down_w_load)

        # ---- gate(x) = x @ gate_w ----------------------------------------
        gate_out = BinaryMap(
            g,
            x_load,
            gate_w_load,
            map_fn.Matmul(weight_transposed=False),
            write_back_mu=False,
            compute_bw=0,
        )
        g.add_node(gate_out)

        # ---- silu(gate_out) -----------------------------------------------
        silu_gate = UnaryMap(
            g,
            gate_out,
            map_fn.Silu(),
            write_back_mu=False,
            compute_bw=0,
        )
        g.add_node(silu_gate)

        # ---- up(x) = x @ up_w ---------------------------------------------
        up_out = BinaryMap(
            g,
            x_load,
            up_w_load,
            map_fn.Matmul(weight_transposed=False),
            write_back_mu=False,
            compute_bw=0,
        )
        g.add_node(up_out)

        # ---- projected = silu(gate) * up ----------------------------------
        projected = BinaryMap(
            g,
            silu_gate,
            up_out,
            map_fn.Mul(),
            write_back_mu=False,
            compute_bw=0,
        )
        g.add_node(projected)

        # ---- down(projected) = projected @ down_w -------------------------
        down_out = BinaryMap(
            g,
            projected,
            down_w_load,
            map_fn.Matmul(weight_transposed=False),
            write_back_mu=False,
            compute_bw=0,
        )
        g.add_node(down_out)

        # ---- mask for this expert (B,1) -----------------------------------
        mask_load = load_mask(masks[i])
        g.add_node(mask_load)

        # Expand mask to (B,D) using the shape of `down_out` as reference
        mask_exp = ExpandRef(g, mask_load, down_out, expand_rank=1)
        g.add_node(mask_exp)

        # ---- zero out tokens that do not belong to this expert ------------
        masked = BinaryMap(
            g,
            down_out,
            mask_exp,
            map_fn.Mul(),
            write_back_mu=False,
            compute_bw=0,
        )
        g.add_node(masked)

        # ---- accumulate across experts ------------------------------------
        if y_node is None:
            y_node = masked
        else:
            y_node = BinaryMap(
                g,
                y_node,
                masked,
                map_fn.Add(),
                write_back_mu=False,
                compute_bw=0,
            )
            g.add_node(y_node)

    # ------------------------------------------------------------------
    # Store the final MoE output (B×D)
    # ------------------------------------------------------------------
    out_store = OffChipStore(g, y_node, par_dispatch=1, store_file_name="output")
    g.add_node(out_store)

    # ------------------------------------------------------------------
    # Broadcast inference & return
    # ------------------------------------------------------------------
    g = infer_broadcast(g)
    return g, out_store