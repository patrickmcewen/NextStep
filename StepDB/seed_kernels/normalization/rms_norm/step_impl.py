"""STeP implementation: RMS Normalization.

Computes x * rsqrt(mean(x^2, dim=-1) + eps) using the STeP element-wise
operator chain: Square(x) -> MulImmediate(1/K) -> RowWiseSum
-> AddImmediate(eps) -> Rsqrt -> Mul with original input.

Based on step_tl/end_to_end/attention/qkv_gen.py::rms_norm, but using
Square instead of Pow2 (which is 2^x, not x^2).
"""

SEED = 42


def build_graph(dims):
    M, K = dims["M"], dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", K)  # default: full K in one tile
    eps = dims.get("eps", 1e-6)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    torch.manual_seed(SEED)
    A = torch.randn(M, K)

    step_graph = Graph()

    # Load input
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # Load input
    load2 = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    k_tiles = K // tile_k

    # x^2 via Square unary op
    x_squared = UnaryMap(
        graph=step_graph,
        input=load,
        fn=Square(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # x^2 * (1/K) — scaling for mean (use full K, not tile_k)
    scaled = UnaryMap(
        graph=step_graph,
        input=x_squared,
        fn=MulImmediate(constant=1.0 / K),
        write_back_mu=False,
        compute_bw=1024,
    )

    # row-wise sum -> partial mean(x^2) per tile -> tile: [tile_m, 1]
    row_sum = UnaryMap(
        graph=step_graph,
        input=scaled,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Accumulate partial sums across K tiles to get full mean(x^2)
    if k_tiles > 1:
        row_sum = Accum(
            step_graph,
            row_sum,
            Tile(tile_dtype=Float32(), shape=(tile_m, 1)),
            AccumAdd(),
            Zero(shape=(tile_m, 1), dtype=Float32()),
            1,
            False,
            1024,
        )

    # + eps
    add_eps = UnaryMap(
        graph=step_graph,
        input=row_sum,
        fn=AddImmediate(constant=eps),
        write_back_mu=False,
        compute_bw=1024,
    )

    # rsqrt
    rsqrt_node = UnaryMap(
        graph=step_graph,
        input=add_eps,
        fn=Rsqrt(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Repeat rsqrt for each K tile so it matches the load stream shape
    if k_tiles > 1:
        rsqrt_node = RepeatStatic(
            graph=step_graph,
            input=rsqrt_node,
            repeat_factor=k_tiles,
        )

    # x * rsqrt(mean(x^2) + eps)
    # load feeds both the Square path above and this BinaryMap;
    # infer_broadcast will insert the necessary Broadcast node.
    normed = BinaryMap(
        graph=step_graph,
        in1=load2,
        in2=rsqrt_node,
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
