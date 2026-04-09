"""STeP implementation: Identity via Bufferize -> Streamify roundtrip.

Loads a 2D tensor, bufferizes it (stream -> buffer), streamifies it back
(buffer -> stream), then stores. Tests the on-chip buffering data path.
Extracted from step_tl/tests/test_buff_streamify.py.
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

    buff = Bufferize(step_graph, load, 1)
    stream = Streamify(step_graph, buff, [], 1)

    output = OffChipStore(
        graph=step_graph,
        input=stream,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
