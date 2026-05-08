"""STeP implementation: 2D matrix transpose via Bufferize + Streamify.

Path: Load(transposed=True) -> Bufferize(rank=2) -> Streamify(swap strides) -> Store.

The full tensor transpose decomposes into:
  - per-tile transpose (transposed=True at load), and
  - tile-grid axis swap (streamify with stride=(1, K_g) over out=(K_g, M_g))
which together turn (M, K) row-major -> (K, M) row-major.
"""

SEED = 42


def build_graph(dims):
    M, K = dims["M"], dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert K % tile_k == 0, f"K={K} not divisible by tile_k={tile_k}"

    M_g = M // tile_m
    K_g = K // tile_k

    torch.manual_seed(SEED)
    A = torch.randn(M, K)

    step_graph = Graph()

    # Load with per-tile transpose: each loaded tile is shape (tile_k, tile_m).
    load = LinearOffChipLoad(
        underlying=A,
        stride=(K_g, 1),
        out_shape_tiled=(M_g, K_g),
        tile_row=tile_m,
        tile_col=tile_k,
        transposed=True,
        par_dispatch=4,
    )

    # Capture the (M_g, K_g) tile grid as an on-chip buffer.
    buff = Bufferize(step_graph, load, 2)

    # Streamify with stride=(1, K_g) over out=(K_g, M_g) reads the row-major
    # buffer in column-major order over the tile grid -- i.e. swaps the two
    # tile-grid axes. linear_idx(k, m) = k*1 + m*K_g picks buffer[m, k].
    stream = Streamify(
        step_graph,
        buff,
        stride=(1, K_g),
        out_shape_tiled=(K_g, M_g),
    )

    output = OffChipStore(
        graph=step_graph,
        input=stream,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
