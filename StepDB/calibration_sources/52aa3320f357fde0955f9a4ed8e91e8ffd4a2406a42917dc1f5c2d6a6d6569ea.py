"""STeP implementation: MatMul using raw BinaryMap + RepeatStatic (no Linear kernel).

This is a lower-level pattern than the Linear kernel — it manually tiles and
dispatches the matmul via OffChipLoad -> RepeatStatic -> BinaryMap -> OffChipStore.
Extracted from step_tl/tests/test_step_py.py.
"""

SEED = 42


def build_graph(dims):
    B, H = dims["B"], dims["H"]
    tile_m = dims.get("tile_m", 16)
    tile_k = H  # full K dimension in one tile (TileMN strategy)
    tile_n = dims.get("tile_n", 32)

    assert B % tile_m == 0, f"B={B} not divisible by tile_m={tile_m}"
    assert H % tile_n == 0, f"H={H} not divisible by tile_n={tile_n}"

    torch.manual_seed(SEED)
    A = torch.randn(B, H)
    model = torch.nn.Linear(H, H, bias=False)
    W = model.weight.T.detach().clone().contiguous()  # [H, H]

    step_graph = Graph()

    # Load input: tile along M dimension, full K in each tile
    input_stream = LinearOffChipLoad(
        underlying=A,
        stride=(H // tile_k, 1),
        out_shape_tiled=(B // tile_m, H // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # Repeat input for each N-tile of the weight
    repeat_input = RepeatStatic(
        graph=step_graph,
        input=input_stream,
        repeat_factor=H // tile_n,
    )

    # Load weight with appropriate stride for N-tiling
    weight_stride = (0, H // tile_n, 1) if H // tile_k == 1 or H // tile_n == 1 else (0, 1, H // tile_n)
    weight_stream = LinearOffChipLoad(
        underlying=W,
        stride=weight_stride,
        out_shape_tiled=(B // tile_m, H // tile_k, H // tile_n),
        tile_row=tile_k,
        tile_col=tile_n,
        par_dispatch=4,
    )

    # Compute matmul
    matmul = BinaryMap(
        step_graph,
        repeat_input,
        weight_stream,
        Matmul(),
        True,
        1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=matmul,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
