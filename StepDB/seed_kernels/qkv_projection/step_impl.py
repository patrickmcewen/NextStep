"""STeP implementation: QKV linear projection.

Implements the projection function from qkv_gen.py: loads input, repeats it
for each output tile, loads weight tiles, and computes via BinaryMap(Matmul).

This is the pattern used to generate Q, K, V in the attention layer:
  input [B, D] tiled as (1,) [tile_b, D]
  weight [D, proj_dim] tiled as (proj_dim // tile_proj,) [D, tile_proj]
  output [B, proj_dim] tiled as (proj_dim // tile_proj,) [tile_b, tile_proj]

Extracted from step_tl/end_to_end/attention/qkv_gen.py::projection.
"""

SEED = 42


def build_graph(dims):
    B, D, proj_dim = dims["B"], dims["D"], dims["proj_dim"]
    tile_b = dims.get("tile_b", B)
    tile_proj = dims.get("tile_proj", 16)

    assert B % tile_b == 0, f"B={B} not divisible by tile_b={tile_b}"
    assert proj_dim % tile_proj == 0, f"proj_dim={proj_dim} not divisible by tile_proj={tile_proj}"

    torch.manual_seed(SEED)
    x = torch.randn(B, D)
    W = torch.randn(D, proj_dim)

    step_graph = Graph()

    # Load input: (B // tile_b,) tiles of [tile_b, D]
    load_x = LinearOffChipLoad(
        underlying=x,
        stride=(1,),
        out_shape_tiled=(B // tile_b,),
        tile_row=tile_b,
        tile_col=D,
        par_dispatch=4,
    )

    # Repeat input for each output N-tile
    # (B // tile_b,) -> (B // tile_b, proj_dim // tile_proj) tiles of [tile_b, D]
    repeat_x = RepeatStatic(
        graph=step_graph,
        input=load_x,
        repeat_factor=proj_dim // tile_proj,
    )

    # Load weight: (B // tile_b, proj_dim // tile_proj) tiles of [D, tile_proj]
    # stride=(0, 1): outer dim doesn't advance (weight is shared across batch)
    #                inner dim advances through proj_dim tiles
    load_w = LinearOffChipLoad(
        underlying=W,
        stride=(0, 1),
        out_shape_tiled=(B // tile_b, proj_dim // tile_proj),
        tile_row=D,
        tile_col=tile_proj,
        par_dispatch=4,
    )

    # Matmul: [tile_b, D] @ [D, tile_proj] -> [tile_b, tile_proj]
    proj = BinaryMap(
        graph=step_graph,
        in1=repeat_x,
        in2=load_w,
        fn=Matmul(),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=proj,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
