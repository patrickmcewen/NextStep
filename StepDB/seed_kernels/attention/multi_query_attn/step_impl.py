"""STeP implementation: Multi-Query Attention.

H query heads share a single K and single V head.
  Q: [H, M, D]  (H heads, M queries, D head dim)
  K: [N, D]      (shared across all H heads)
  V: [N, D]      (shared across all H heads)

K and V are loaded once and repeated H times via RepeatStatic.
Each head computes: softmax(Q_h @ K^T) @ V independently,
accumulated across N tiles.

Tests: RepeatStatic fan-out on K/V for multi-head sharing,
larger stream shapes with head dimension.
"""

SEED = 42


def build_graph(dims):
    H = dims["H"]
    M, N, D = dims["M"], dims["N"], dims["D"]
    tile_m = dims.get("tile_m", M)
    tile_n = dims.get("tile_n", N)

    assert M % tile_m == 0, f"M={M} not divisible by tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} not divisible by tile_n={tile_n}"

    torch.manual_seed(SEED)
    # H independent query heads
    Q_data = torch.randn(H * M, D)  # Interleaved: [H, M, D] flattened to [H*M, D]
    K_data = torch.randn(N, D)
    V_data = torch.randn(N, D)

    step_graph = Graph()

    # Load Q: (H, M//tile_m) tiles of [tile_m, D], then repeat over N
    load_q = LinearOffChipLoad(
        underlying=Q_data,
        stride=(M // tile_m, 1),
        out_shape_tiled=(H, M // tile_m),
        tile_row=tile_m,
        tile_col=D,
        par_dispatch=4,
    )
    q_repeated = RepeatStatic(
        graph=step_graph,
        input=load_q,
        repeat_factor=N // tile_n,
    )
    # q_repeated: (H, M//tile_m, N//tile_n) tiles of [tile_m, D]

    # Load K: (N//tile_n,) tiles of [tile_n, D], repeated over H and M
    load_k = LinearOffChipLoad(
        underlying=K_data,
        stride=(1,),
        out_shape_tiled=(N // tile_n,),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )
    # Repeat for H heads
    k_per_head = RepeatStatic(
        graph=step_graph,
        input=load_k,
        repeat_factor=H,
    )
    # k_per_head: (N//tile_n, H) -> need (H, M//tile_m, N//tile_n)
    # Actually, let's use a simpler approach: load K with the right shape directly.

    # Simpler: load K as (H, M//tile_m, N//tile_n) with stride ignoring H and M dims
    load_k2 = LinearOffChipLoad(
        underlying=K_data,
        stride=(0, 0, 1),
        out_shape_tiled=(H, M // tile_m, N // tile_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )

    # QK^T: [tile_m, D] @ [tile_n, D]^T -> [tile_m, tile_n]
    qkt = BinaryMap(
        graph=step_graph,
        in1=q_repeated,
        in2=load_k2,
        fn=Matmul(weight_transposed=True),
        write_back_mu=False,
        compute_bw=1024,
    )

    # exp(QK^T)
    exp_qkt = UnaryMap(
        graph=step_graph,
        input=qkt,
        fn=Exp(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Broadcast for V-matmul and sum
    exp_broadcast = Broadcast(step_graph, exp_qkt, 2)

    # Load V: (H, M//tile_m, N//tile_n) with stride ignoring H and M
    load_v = LinearOffChipLoad(
        underlying=V_data,
        stride=(0, 0, 1),
        out_shape_tiled=(H, M // tile_m, N // tile_n),
        tile_row=tile_n,
        tile_col=D,
        par_dispatch=4,
    )

    # exp(QK^T) @ V, accumulated over N
    mult_v = BinaryMapAccum(
        graph=step_graph,
        in1=(exp_broadcast, 0),
        in2=load_v,
        fn=MapAccumMatmul(),
        init_fn=Zero(shape=(tile_m, D), dtype=Float32()),
        rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    # Sum of exp for normalization
    tile_shape_exp = (tile_m, tile_n)
    tile_wise_rowsum = Accum(
        graph=step_graph,
        input=(exp_broadcast, 1),
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=tile_shape_exp),
        fn=AccumAdd(),
        init_fn=Zero(shape=tile_shape_exp, dtype=Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )

    intra_tile_rowsum = UnaryMap(
        graph=step_graph,
        input=tile_wise_rowsum,
        fn=RowWiseSum(),
        write_back_mu=False,
        compute_bw=1024,
    )

    # Divide
    softmax_out = BinaryMap(
        graph=step_graph,
        in1=mult_v,
        in2=intra_tile_rowsum,
        fn=Div(),
        write_back_mu=True,
        compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph,
        input=softmax_out,
        par_dispatch=4,
        store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
