"""STeP implementation: Multiple concurrent HBM loads feeding compute.

4 independent loads (A, B, C, D) -> (A*B) + (C*D) -> Store

Tests HBM contention from 4 simultaneous off-chip loads feeding
into a chain of binary operations. Stresses the timing model's
HBM bandwidth contention modeling (pass 2).
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
    B = torch.randn(M, K)
    C = torch.randn(M, K)
    D = torch.randn(M, K)

    step_graph = Graph()

    load_a = LinearOffChipLoad(
        underlying=A,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    load_b = LinearOffChipLoad(
        underlying=B,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    load_c = LinearOffChipLoad(
        underlying=C,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    load_d = LinearOffChipLoad(
        underlying=D,
        stride=(K // tile_k, 1),
        out_shape_tiled=(M // tile_m, K // tile_k),
        tile_row=tile_m,
        tile_col=tile_k,
        par_dispatch=4,
    )

    # A * B
    mul_ab = BinaryMap(
        graph=step_graph,
        in1=load_a,
        in2=load_b,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # C * D
    mul_cd = BinaryMap(
        graph=step_graph,
        in1=load_c,
        in2=load_d,
        fn=Mul(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # (A*B) + (C*D)
    result = BinaryMap(
        graph=step_graph,
        in1=mul_ab,
        in2=mul_cd,
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
