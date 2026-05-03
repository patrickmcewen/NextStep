"""STeP implementation: rank-2 Bufferize + new-API Streamify outer-repeat.

Path: Load -> Bufferize(rank=2) -> Streamify(stride leading=0, out has leading R) -> Store.

The leading-dim stride=0 broadcast is the canonical Streamify use case and
what the cost model's "repeat" intent compiles to under the new interface.
"""

SEED = 42


def build_graph(dims):
    M, K = dims["M"], dims["K"]
    R = dims["R"]
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

    buff = Bufferize(step_graph, load, 2)

    # Outer dim has stride=0 — emits R copies of the buffer in sequence.
    # linear_idx(r, i, j) = 0*r + (K//tile_k)*i + 1*j = i*(K//tile_k) + j.
    stream = Streamify(
        step_graph,
        buff,
        stride=(0, K // tile_k, 1),
        out_shape_tiled=(R, M // tile_m, K // tile_k),
    )

    output = OffChipStore(
        graph=step_graph,
        input=stream,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
