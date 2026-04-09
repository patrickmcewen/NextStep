"""STeP implementation: Residual add + RMS norm (fused).

Computes rms_norm(x + residual):
  y = x + residual
  y * rsqrt(mean(y^2, dim=-1) + eps)

Two loads feed a BinaryMap(Add), whose output feeds the RMS norm pipeline.
Tests element-wise -> accumulation transition and fused kernel timing.
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
    X = torch.randn(M, K)
    R = torch.randn(M, K)

    step_graph = Graph()

    load_x = LinearOffChipLoad(
        underlying=X,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    load_r = LinearOffChipLoad(
        underlying=R,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # y = x + residual
    y = BinaryMap(
        graph=step_graph,
        in1=load_x,
        in2=load_r,
        fn=Add(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # === RMS Norm on y ===
    # y^2
    y_sq = UnaryMap(
        graph=step_graph,
        input=y,
        fn=Square(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # y^2 * (1/K)
    scaled = UnaryMap(
        graph=step_graph,
        input=y_sq,
        fn=MulImmediate(constant=1.0 / K),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Row-wise sum -> partial mean(y^2)
    row_sum = UnaryMap(
        graph=step_graph,
        input=scaled,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

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

    if k_tiles > 1:
        rsqrt_node = RepeatStatic(
            graph=step_graph,
            input=rsqrt_node,
            repeat_factor=k_tiles,
        )

    # y * rsqrt(mean(y^2) + eps)
    # y feeds both the Square path and this Mul; infer_broadcast handles it
    normed = BinaryMap(
        graph=step_graph,
        in1=y,
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
