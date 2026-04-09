"""STeP implementation: Broadcast diamond pattern.

Load -> Broadcast(3) -> [Square, Silu, Exp] -> Add pairs -> Store

Tests fan-out to 3 consumers with diamond reconvergence:
  branch0 = Square(x)
  branch1 = Silu(x)
  branch2 = Exp(x)
  result = (branch0 + branch1) + branch2

Stresses broadcast timing and reconvergence scheduling.
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

    # Fan-out to 3 consumers
    bc = Broadcast(step_graph, load, 3)

    # Branch 0: Square
    branch0 = UnaryMap(
        graph=step_graph,
        input=(bc, 0),
        fn=Square(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Branch 1: Silu
    branch1 = UnaryMap(
        graph=step_graph,
        input=(bc, 1),
        fn=Silu(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Branch 2: Exp
    branch2 = UnaryMap(
        graph=step_graph,
        input=(bc, 2),
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Merge: (branch0 + branch1)
    sum_01 = BinaryMap(
        graph=step_graph,
        in1=branch0,
        in2=branch1,
        fn=Add(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Merge: (branch0 + branch1) + branch2
    result = BinaryMap(
        graph=step_graph,
        in1=sum_01,
        in2=branch2,
        fn=Add(),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=result,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
