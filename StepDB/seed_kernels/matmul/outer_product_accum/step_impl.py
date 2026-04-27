"""STeP implementation: Batched outer product via A^T @ B.

Computes C[M,N] = A^T[M,B] @ B[B,N], which is equivalent to
sum_b( A[b,:].unsqueeze(1) @ B[b,:].unsqueeze(0) ) — a sum of
rank-1 outer products accumulated across the batch dimension.

Uses standard GEMM tiling with the batch dimension as the reduction
(K) axis. Tests non-standard matmul shapes where the "K" dimension
represents batch accumulation rather than a typical hidden dimension.
"""

SEED = 42


def build_graph(dims):
    B = dims["B"]
    M, N = dims["M"], dims["N"]
    tile_m = dims.get("tile_m", 16)
    tile_n = dims.get("tile_n", 16)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    torch.manual_seed(SEED)
    A_data = torch.randn(B, M)
    B_data = torch.randn(B, N)

    # Transpose A so it's [M, B] for standard GEMM tiling
    A_T = A_data.T.contiguous()  # [M, B]

    step_graph = Graph()

    # Standard GEMM: C = A_T @ B_data, shapes [M,B] @ [B,N] -> [M,N]
    # Tiling: accumulate over B (the "K" dimension)
    # A_T tiles: (M//tile_m, N//tile_n, B) of [tile_m, 1]
    #   - but tile_col=1 means B tiles per row, so tile the B dim as individual elements
    # Actually, let's just treat B as K with tile_k = B (full reduction in one pass)
    # This gives a simpler tiling.

    # A_T: [M, B], tile as (M//tile_m,) with tiles [tile_m, B]
    # B_data: [B, N], tile as (N//tile_n,) with tiles [B, tile_n]
    # Matmul: [tile_m, B] @ [B, tile_n] -> [tile_m, tile_n]
    # Stream shape: (M//tile_m, N//tile_n) — no accumulation needed since full B in one tile

    load_a = LinearOffChipLoad(
        underlying=A_T,
        stride=(1, 0),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=tile_m,
        tile_col=B,
        par_dispatch=4,
    )

    load_b = LinearOffChipLoad(
        underlying=B_data,
        stride=(0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n),
        tile_row=B,
        tile_col=tile_n,
        par_dispatch=4,
    )

    # Non-accumulated matmul: [tile_m, B] @ [B, tile_n] -> [tile_m, tile_n]
    matmul = BinaryMap(
        graph=step_graph,
        in1=load_a,
        in2=load_b,
        fn=Matmul(weight_transposed=False),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=matmul,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
