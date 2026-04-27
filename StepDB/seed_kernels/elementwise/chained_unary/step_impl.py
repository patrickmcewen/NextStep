"""STeP implementation: Chained unary operations.

Deep pipeline of 4 consecutive UnaryMap ops:
  Load -> Square -> Exp -> Silu -> Rsqrt -> Store

Tests pipeline depth timing — whether OTI propagates correctly through
long chains of dependent unary operations.
"""

SEED = 42


def build_graph(dims):
    M, K = dims["M"], dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    torch.manual_seed(SEED)
    # Use small positive values to avoid NaN in rsqrt(silu(exp(x^2)))
    A = torch.rand(M, K) * 0.5 + 0.1

    step_graph = Graph()

    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # Stage 1: x^2
    squared = UnaryMap(
        graph=step_graph,
        input=load,
        fn=Square(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Stage 2: exp(x^2)
    exped = UnaryMap(
        graph=step_graph,
        input=squared,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Stage 3: silu(exp(x^2))
    silued = UnaryMap(
        graph=step_graph,
        input=exped,
        fn=Silu(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Stage 4: rsqrt(silu(exp(x^2)))
    result = UnaryMap(
        graph=step_graph,
        input=silued,
        fn=Rsqrt(),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=result,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
