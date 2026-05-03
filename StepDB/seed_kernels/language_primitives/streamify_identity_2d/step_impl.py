"""STeP implementation: rank-2 Bufferize + new-API Streamify identity.

Path: Load -> Bufferize(rank=2) -> Streamify(stride=row-major, out=full grid) -> Store.
Tests the new Streamify interface (stride, out_shape_tiled) with a 2D
out_shape_tiled — exercises lex-order walk and multi-dim non-zero stride
in both the Python functional model and the Rust simulator.
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

    # Rank-2 bufferize: the entire tile-grid becomes a single Buffer.
    buff = Bufferize(step_graph, load, 2)

    # New-API Streamify: row-major linear indexing over the buffer's tile-grid.
    # linear_idx(i, j) = i*(K//tile_k) + j = the (i, j)-th tile in row-major order.
    stream = Streamify(
        step_graph,
        buff,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
    )

    output = OffChipStore(
        graph=step_graph,
        input=stream,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
