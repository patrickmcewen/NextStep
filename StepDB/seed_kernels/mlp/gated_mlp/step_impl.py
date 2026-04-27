"""STeP implementation: Gated MLP (SwiGLU variant) for a single expert.

Computes: down(silu(gate(x)) * up(x))
Three matmuls with SiLU gating — the core MoE expert computation.

Uses the Linear kernel abstraction which handles tiling internally.
Extracted from step_tl/dyn_tiling/test_weight_stationary_gemm.py.
"""

SEED = 42


def build_graph(dims):
    B, D, F = dims["B"], dims["D"], dims["F"]
    tile_b = dims.get("tile_b", B)
    tile_f = dims.get("tile_f", F)

    assert B % tile_b == 0, f"B={B} not divisible by tile_b={tile_b}"
    assert F % tile_f == 0, f"F={F} not divisible by tile_f={tile_f}"

    torch.manual_seed(SEED)
    gate_model = torch.nn.Linear(D, F, bias=False)
    up_model = torch.nn.Linear(D, F, bias=False)
    down_model = torch.nn.Linear(F, D, bias=False)

    w_gate = gate_model.weight.T.detach().clone().contiguous()  # [D, F]
    w_up = up_model.weight.T.detach().clone().contiguous()      # [D, F]
    w_down = down_model.weight.T.detach().clone().contiguous()  # [F, D]

    x = torch.randn(B, D)

    step_graph = Graph()

    # --- Gate projection via Linear: [B, D] @ [D, F] -> [B, F] ---
    gate_out = Linear(
        step_graph=step_graph,
        input=x,
        weight=w_gate,
        tile_config=LinearTileConfig(m=tile_b, k=D, n=tile_f),
        comp_bw=1024,
        write_back_mu=False,
        par_dispatch=4,
    )

    # --- Up projection via Linear: [B, D] @ [D, F] -> [B, F] ---
    up_out = Linear(
        step_graph=step_graph,
        input=x,
        weight=w_up,
        tile_config=LinearTileConfig(m=tile_b, k=D, n=tile_f),
        comp_bw=1024,
        write_back_mu=False,
        par_dispatch=4,
    )

    # --- SiLU on gate output ---
    gate_activated = UnaryMap(
        graph=step_graph,
        input=gate_out,
        fn=Silu(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # --- Element-wise multiply: silu(gate) * up ---
    gated = BinaryMap(
        step_graph, gate_activated, up_out,
        Mul(), False, 1024,
    )

    # --- Down projection via Linear: [B, F] @ [F, D] -> [B, D] ---
    # Bufferize/streamify to convert on-chip stream to Linear-compatible input
    buff = Bufferize(step_graph, gated, 1)
    gated_stream = Streamify(step_graph, buff, [], 1)

    down_out = Linear(
        step_graph=step_graph,
        input=gated_stream,
        weight=w_down,
        tile_config=LinearTileConfig(m=tile_b, k=tile_f, n=D),
        comp_bw=1024,
        write_back_mu=True,
        par_dispatch=4,
    )

    output = OffChipStore(
        graph=step_graph,
        input=down_out,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
