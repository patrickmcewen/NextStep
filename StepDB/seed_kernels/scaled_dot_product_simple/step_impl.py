"""STeP implementation: Simplified scaled dot-product (Q @ K^T * scale).

Computes: scale * (Q @ K^T) where scale = 1/sqrt(D).
No softmax, no V multiply — isolates matmul + scalar multiply timing.

Tiling: Q is [M, D], K is [N, D].
  Q tiles: (M//tile_m,) repeated over N -> (M//tile_m, N//tile_n, D//tile_d)
  K tiles: (N//tile_n,) repeated over M -> (M//tile_m, N//tile_n, D//tile_d)
  Accumulated matmul over D dimension -> (M//tile_m, N//tile_n) tiles of [tile_m, tile_n]
  Scaled by MulImmediate(1/sqrt(D)).

Tests matmul with accumulation + immediate scalar multiply.
"""

SEED = 42


def build_graph(dims):
    M, N, D = dims["M"], dims["N"], dims["D"]
    tile_m = dims.get("tile_m", M)
    tile_n = dims.get("tile_n", N)
    tile_d = dims.get("tile_d", D)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"
    assert D % tile_d == 0, f"D={D} not divisible by tile_d={tile_d}"

    scale = 1.0 / (D ** 0.5)

    torch.manual_seed(SEED)
    Q_data = torch.randn(M, D)
    K_data = torch.randn(N, D)

    step_graph = Graph()

    # Load Q: (M//tile_m, N//tile_n, D//tile_d) tiles of [tile_m, tile_d]
    load_q = LinearOffChipLoad(
        underlying=Q_data,
        stride=(D // tile_d, 0, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, D // tile_d),
        tile_row=tile_m,
        tile_col=tile_d,
        par_dispatch=4,
    )

    # Load K: (M//tile_m, N//tile_n, D//tile_d) tiles of [tile_n, tile_d]
    load_k = LinearOffChipLoad(
        underlying=K_data,
        stride=(0, D // tile_d, 1),
        out_shape_tiled=(M // tile_m, N // tile_n, D // tile_d),
        tile_row=tile_n,
        tile_col=tile_d,
        par_dispatch=4,
    )

    # Q @ K^T accumulated over D: [tile_m, tile_d] @ [tile_n, tile_d]^T -> [tile_m, tile_n]
    qkt = BinaryMapAccum(
        graph=step_graph,
        in1=load_q,
        in2=load_k,
        fn=MapAccumMatmul(weight_transposed=True),
        init_fn=Zero(shape=(tile_m, tile_n), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    # Scale by 1/sqrt(D)
    scaled = UnaryMap(
        graph=step_graph,
        input=qkt,
        fn=MulImmediate(constant=scale),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=scaled,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
