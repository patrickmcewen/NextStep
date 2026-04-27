"""STeP implementation: Vector reduction (sum).

Pure accumulation kernel: loads a 2D tensor and reduces along the K dimension
via Accum(Add). Tests accumulation timing in isolation with no element-wise
compute — just Load -> Accum -> Store.
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

    # Accumulate across K tiles: (M//tile_m, K//tile_k) -> (M//tile_m,)
    reduced = Accum(
        step_graph,
        load,
        Tile(tile_dtype=Float32(), shape=(tile_m, tile_k)),
        AccumAdd(),
        Zero(shape=(tile_m, tile_k), dtype=Float32()),
        1,
        True,
        1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=reduced,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
