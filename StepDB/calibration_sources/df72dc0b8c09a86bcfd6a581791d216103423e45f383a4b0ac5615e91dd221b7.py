"""STeP implementation: SiLU (Swish) activation.

Applies SiLU element-wise via UnaryMap(Silu).
Used in the gated MLP of MoE expert layers.
Extracted from step_tl/end_to_end/moe/ and dyn_tiling/ patterns.
"""

SEED = 42


def build_graph(dims):
    M, K = dims["M"], dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    torch.manual_seed(SEED)
    A = torch.randn(M, K)

    step_graph = Graph()

    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    activated = UnaryMap(
        graph=step_graph,
        input=load,
        fn=Silu(),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=activated,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
