"""STeP implementation: 4D axis permute via Bufferize(rank=3) + Streamify.

Path: Load(tile (1, D), batch (B, H)) -> Bufferize(rank=3) -> Streamify -> Store.

Demonstrates the n-D buffer-grid permutation recipe: pick the row-major
flat strides for the buffer dims, then index them in the desired
output-axis order.
"""

SEED = 42


def build_graph(dims):
    B = dims["B"]
    H = dims["H"]
    S = dims["S"]
    D = dims["D"]

    torch.manual_seed(SEED)
    A = torch.randn(B, H, S, D)

    step_graph = Graph()

    # Tile (1, D) keeps each (b, h, s) D-vector as a single tile.
    # Identity load: tile-grid (B, H, S) in row-major order.
    load = LinearOffChipLoad(
        underlying=A,
        stride=(H * S, S, 1),
        out_shape_tiled=(B, H, S),
        tile_row=1,
        tile_col=D,
        par_dispatch=4,
    )

    # Capture the (B, H, S) tile grid.
    buff = Bufferize(step_graph, load, 3)

    # Buffer flat stride for (B, H, S) is (H*S, S, 1). Output axes in
    # order (B, S, H) want strides (H*S, 1, S):
    #   linear_idx(b, s, h) = b*H*S + s*1 + h*S = b*H*S + h*S + s,
    # which is the row-major flat index of buffer position (b, h, s).
    stream = Streamify(
        step_graph,
        buff,
        stride=(H * S, 1, S),
        out_shape_tiled=(B, S, H),
    )

    output = OffChipStore(
        graph=step_graph,
        input=stream,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
