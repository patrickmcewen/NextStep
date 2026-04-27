"""STeP implementation: Row-wise softmax (exp-normalize).

Computes softmax(x) = exp(x) / sum(exp(x), dim=-1) per row.
Uses Broadcast to fan exp(x) to both the sum path and the divide path.
Tests accumulation + broadcast + multi-pass pipeline timing.

Tiling: (M//tile_m, K//tile_k) tiles of [tile_m, tile_k].
When K > tile_k, partial sums are accumulated across K tiles, then
RowWiseSum reduces within each tile to get per-row denominators.
"""

SEED = 42


def build_graph(dims):
    M, K = dims["M"], dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", K)

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

    # exp(x)
    exp_x = UnaryMap(
        graph=step_graph,
        input=load,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Broadcast exp(x) to two consumers: numerator path and sum path
    exp_broadcast = Broadcast(step_graph, exp_x, 2)

    # Sum path: accumulate exp tiles across K, then intra-tile row sum
    if k_tiles > 1:
        tile_accum = Accum(
            step_graph,
            (exp_broadcast, 1),
            Tile(tile_dtype=Float32(), shape=(tile_m, tile_k)),
            AccumAdd(),
            Zero(shape=(tile_m, tile_k), dtype=Float32()),
            1,
            False,
            1024,
        )
    else:
        tile_accum = (exp_broadcast, 1)

    row_sum = UnaryMap(
        graph=step_graph,
        input=tile_accum,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Repeat row_sum for each K tile so shapes match the numerator
    if k_tiles > 1:
        row_sum = RepeatStatic(
            graph=step_graph,
            input=row_sum,
            repeat_factor=k_tiles,
        )

    # Divide: exp(x) / sum(exp(x))
    softmax_out = BinaryMap(
        graph=step_graph,
        in1=(exp_broadcast, 0),
        in2=row_sum,
        fn=Div(),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=softmax_out,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
