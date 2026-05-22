"""STeP implementation: Layer Normalization.

Computes: (x - mean(x)) * rsqrt(var(x) + eps)
  where mean and var are per-row (dim=-1).

Pipeline:
  Load x -> [fan-out via infer_broadcast]
    Path 1 (mean): MulImmediate(1/K) -> RowWiseSum -> [Accum] -> MulImmediate(-1) -> [Repeat]
    x + (-mean) -> [fan-out]
      Path A (variance): Square -> MulImmediate(1/K) -> RowWiseSum -> [Accum] -> AddImmediate(eps) -> Rsqrt -> [Repeat]
      (x - mean) * rsqrt(var + eps) -> Store

Tests: two sequential reductions, deep dependency chain, multiple fan-outs.
"""

SEED = 42


def build_graph(dims):
    M, K = dims["M"], dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", K)
    eps = dims.get("eps", 1e-6)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    k_tiles = K // tile_k

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

    # === Mean computation ===
    scaled_for_mean = UnaryMap(
        graph=step_graph,
        input=load,
        fn=MulImmediate(constant=1.0 / K),
        write_back_mu=False,
        compute_bw=1024,
    )

    mean_rowsum = UnaryMap(
        graph=step_graph,
        input=scaled_for_mean,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    if k_tiles > 1:
        mean_rowsum = Accum(
            step_graph,
            mean_rowsum,
            Tile(tile_dtype=Float32(), shape=(tile_m, 1)),
            AccumAdd(),
            Zero(shape=(tile_m, 1), dtype=Float32()),
            1,
            False,
            1024,
        )

    # Negate mean: -mean
    neg_mean = UnaryMap(
        graph=step_graph,
        input=mean_rowsum,
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False,
        compute_bw=1024,
    )

    if k_tiles > 1:
        neg_mean = RepeatStatic(
            graph=step_graph,
            input=neg_mean,
            repeat_factor=k_tiles,
        )

    # === x - mean (via x + (-mean)) ===
    # load is consumed by both mean path and this Add; infer_broadcast handles it
    x_centered = BinaryMap(
        graph=step_graph,
        in1=load,
        in2=neg_mean,
        fn=Add(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # === Variance computation on (x - mean) ===
    x_sq = UnaryMap(
        graph=step_graph,
        input=x_centered,
        fn=Square(),
        write_back_mu=False,
        compute_bw=1024,
    )

    scaled_sq = UnaryMap(
        graph=step_graph,
        input=x_sq,
        fn=MulImmediate(constant=1.0 / K),
        write_back_mu=False,
        compute_bw=1024,
    )

    var_rowsum = UnaryMap(
        graph=step_graph,
        input=scaled_sq,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    if k_tiles > 1:
        var_rowsum = Accum(
            step_graph,
            var_rowsum,
            Tile(tile_dtype=Float32(), shape=(tile_m, 1)),
            AccumAdd(),
            Zero(shape=(tile_m, 1), dtype=Float32()),
            1,
            False,
            1024,
        )

    # var + eps
    var_eps = UnaryMap(
        graph=step_graph,
        input=var_rowsum,
        fn=AddImmediate(constant=eps),
        write_back_mu=False,
        compute_bw=1024,
    )

    # rsqrt(var + eps)
    inv_std = UnaryMap(
        graph=step_graph,
        input=var_eps,
        fn=Rsqrt(),
        write_back_mu=False,
        compute_bw=1024,
    )

    if k_tiles > 1:
        inv_std = RepeatStatic(
            graph=step_graph,
            input=inv_std,
            repeat_factor=k_tiles,
        )

    # === (x - mean) * rsqrt(var + eps) ===
    # x_centered is consumed by both variance path and this Mul; infer_broadcast handles it
    normed = BinaryMap(
        graph=step_graph,
        in1=x_centered,
        in2=inv_std,
        fn=Mul(),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=normed,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
