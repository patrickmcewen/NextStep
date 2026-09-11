"""STeP implementation: head split + permute via Bufferize + Streamify.

Path: Load(tile (1, D)) -> Bufferize(rank=2) -> Streamify(swap strides) -> Store.

Tile shape (1, D) keeps each head's vector as a single tile, so the
per-tile transpose is unnecessary. The (S, H) tile grid is swapped to
(H, S) by streamify with stride=(1, H) over out=(H, S).
"""

SEED = 42


def build_graph(dims):
    S = dims["S"]
    H = dims["H"]
    D = dims["D"]

    torch.manual_seed(SEED)
    A = torch.randn(S, H * D)

    step_graph = Graph()

    # Each tile is one (head, position) D-vector.
    load = LinearOffChipLoad(
        underlying=A,
        stride=(H, 1),
        out_shape_tiled=(S, H),
        tile_row=1,
        tile_col=D,
        par_dispatch=4,
    )

    # Capture (S, H) tile grid as a buffer.
    buff = Bufferize(step_graph, load, 2)

    # linear_idx(h, s) = h*1 + s*H -> picks buffer flat at s*H + h,
    # which is buffer position (s=s, h=h). That swaps the tile-grid axes.
    stream = Streamify(
        step_graph,
        buff,
        stride=(1, H),
        out_shape_tiled=(H, S),
    )

    output = OffChipStore(
        graph=step_graph,
        input=stream,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
