"""STeP implementation: QKV generation (projection + per-head RMSNorm + RoPE).

Mirrors step_tl/end_to_end/attention/qkv_gen.py but written in the StepDB
idiom (single LinearOffChipLoad per tensor with internal par_dispatch,
following seed_kernels/rope/step_impl.py) so there is exactly one output
OffChipStore per graph, in a deterministic row-major layout.

Pipeline (stream shape / tile shape):
  Load x         -> (1,)       / (B, D)              -- par_dispatch splits B
  Repeat N_HEAD  -> (1, N_HEAD)/ (B, D)
  Load W_q/k/v   -> (1, N_HEAD)/ (D, HEAD_DIM)
  Matmul         -> (1, N_HEAD)/ (B, HEAD_DIM)
  (Q, K): Pow2 * (1/HEAD_DIM) -> RowWiseSum -> + eps -> Rsqrt -> Mul -> rms-normed
  Load cos, sin  -> (1,)       / (B, HEAD_DIM)
  Repeat N_HEAD  -> (1, N_HEAD)/ (B, HEAD_DIM)
  (Q, K): rotate_half -> *sin, *cos then Add -> RoPE applied
  Flatten each to 1D stream of N_HEAD tiles -> concat [Q, K, V] -> store

Output layout: [3*B*N_HEAD, HEAD_DIM], grouped (Q, K, V) with batch-outer,
head-inner within each section. Each store-tile is [N_HEAD, HEAD_DIM] and
the stream walks B tiles per Q/K/V. The reference's compute_gold reshapes
to match.
"""

SEED = 42


def _rotate_half(step_graph, input_stream, batch, num_heads, head_dim):
    """qkv_gen.py-style half-swap rotate_half on a (1, batch, 1) stream with
    tile (num_heads, head_dim). Returns a node with the same stream / tile.
    """
    split_half = RetileStreamify(
        graph=step_graph, input=input_stream,
        split_row=False, chunk=head_dim // 2,
    )

    flat = Flatten(
        graph=step_graph, input=split_half, min_rank=0, max_rank=2,
    )

    splitted = Parallelize(
        graph=step_graph, input=flat,
        parallelize_rank=0, num_consumers=2, switch_cycles=[1, 1],
    )

    neg_second = UnaryMap(
        graph=step_graph, input=(splitted, 1),
        fn=MulImmediate(constant=-1.0),
        write_back_mu=False, compute_bw=1024,
    )

    concat = StaticReassemble(
        graph=step_graph, inputs=[neg_second, (splitted, 0)],
        merge_rank=0, switch_cycles=[1, 1],
    )

    reshaped = ReshapePadStream(
        graph=step_graph, input=concat,
        chunk_size=2, reshape_rank=0,
        write_back_mu=False, pad_fn=None, have_pad_stream=False,
    )

    full_tile = Accum(
        graph=step_graph, input=reshaped,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(num_heads, head_dim)),
        fn=RetileCol(),
        init_fn=Empty(shape=(num_heads, 0), dtype=Float32()),
        accum_rank=1, write_back_mu=True, compute_bw=1024,
    )

    restored_inner = ReshapePadStream(
        graph=step_graph, input=full_tile,
        chunk_size=1, reshape_rank=0,
        write_back_mu=False, pad_fn=None, have_pad_stream=False,
    )
    restored_outer = ReshapePadStream(
        graph=step_graph, input=restored_inner,
        chunk_size=batch, reshape_rank=1,
        write_back_mu=False, pad_fn=None, have_pad_stream=False,
    )
    return restored_outer


def build_graph(dims):
    B, D, N_HEAD, HEAD_DIM = dims["B"], dims["D"], dims["N_HEAD"], dims["HEAD_DIM"]
    PAR_DISPATCH = dims.get("par_dispatch", 4)

    assert HEAD_DIM % 2 == 0, f"HEAD_DIM={HEAD_DIM} must be even for rotate_half"

    torch.manual_seed(SEED)
    q_proj = torch.randn(D, N_HEAD* HEAD_DIM)
    k_proj = torch.randn(D, N_HEAD* HEAD_DIM)
    v_proj = torch.randn(D, N_HEAD* HEAD_DIM)
    x = torch.randn(B, D)
    cos = torch.randn(B, 1, HEAD_DIM)
    sin = torch.randn(B, 1, HEAD_DIM)

    step_graph = Graph()

    # ---------- Input + projection ----------
    # x: one tile of [B, D], repeated N_HEAD times
    x_load = LinearOffChipLoad(
        underlying=x, stride=(1,),
        out_shape_tiled=(B, 1),
        tile_row=1, tile_col=D, par_dispatch=PAR_DISPATCH,
    )

    def projection(W):
        w_load = LinearOffChipLoad(
            underlying=W, stride=(0,1),
            out_shape_tiled=(B, 1),
            tile_row=D, tile_col=HEAD_DIM*N_HEAD, par_dispatch=PAR_DISPATCH,
        )
        proj = BinaryMap(
            graph=step_graph, in1=x_load, in2=w_load,
            fn=Matmul(), write_back_mu=True, compute_bw=1024,
        )
        proj_reshaped = RetileStreamify(
            graph=step_graph, input=proj,
            split_row=False, chunk=HEAD_DIM,
        )
        proj_reshaped_2 = Accum(
            graph=step_graph, input=proj_reshaped,
            output_stream_dtype=Tile(tile_dtype=Float32(), shape=(N_HEAD, HEAD_DIM)),
            fn=RetileRow(),
            init_fn=Empty(shape=(N_HEAD, 0), dtype=Float32()),
            accum_rank=1, compute_bw=1024, write_back_mu=False,
        )
        proj_reshaped_3 = Promote(
            graph=step_graph, input=proj_reshaped_2,
            promote_rank=0,
        )
        return proj_reshaped_3

    print(q_proj.shape, k_proj.shape, v_proj.shape)
    Q = projection(q_proj)
    K = projection(k_proj)
    V = projection(v_proj)
    print(Q._stream.shape, K._stream.shape, V._stream.shape)

    # ---------- Per-head RMSNorm on Q, K (tile [B, HEAD_DIM]) ----------
    def rms_norm(stream):
        pow2 = UnaryMap(
            graph=step_graph, input=stream, fn=Square(),
            write_back_mu=False, compute_bw=1024,
        )
        scaled = UnaryMap(
            graph=step_graph, input=pow2,
            fn=MulImmediate(constant=1.0 / HEAD_DIM),
            write_back_mu=False, compute_bw=1024,
        )
        rowsum = UnaryMap(
            graph=step_graph, input=scaled, fn=RowWiseSum(),
            write_back_mu=False, compute_bw=1024,
        )
        add_eps = UnaryMap(
            graph=step_graph, input=rowsum,
            fn=AddImmediate(constant=1e-6),
            write_back_mu=False, compute_bw=1024,
        )
        rsqrt = UnaryMap(
            graph=step_graph, input=add_eps, fn=Rsqrt(),
            write_back_mu=False, compute_bw=1024,
        )
        return BinaryMap(
            graph=step_graph, in1=stream, in2=rsqrt,
            fn=Mul(), write_back_mu=False, compute_bw=1024,
        )

    Q_norm = rms_norm(Q)
    K_norm = rms_norm(K)

    print(Q_norm._stream.shape, Q_norm._stream.stream_dtype, K_norm._stream.shape, K_norm._stream.stream_dtype)

    # ---------- RoPE: cos/sin broadcast across heads ----------
    cos_load = LinearOffChipLoad(
        underlying=cos, stride=(1, 1), out_shape_tiled=(B, 1),
        tile_row=1, tile_col=HEAD_DIM, par_dispatch=4,
    )
    sin_load = LinearOffChipLoad(
        underlying=sin, stride=(1, 1), out_shape_tiled=(B, 1),
        tile_row=1, tile_col=HEAD_DIM, par_dispatch=4,
    )

    def rope(weight, cos, sin, num_heads, head_dim):
        weight_rot = _rotate_half(step_graph, weight, B, num_heads, head_dim)
        print("finished rotate_half")

        weight_cos = BinaryMap(
            graph=step_graph, in1=weight, in2=cos,
            fn=Mul(), write_back_mu=False, compute_bw=1024,
        )
        weight_sin = BinaryMap(
            graph=step_graph, in1=weight_rot, in2=sin,
            fn=Mul(), write_back_mu=False, compute_bw=1024,
        )
        weight_out = BinaryMap(
            graph=step_graph, in1=weight_cos, in2=weight_sin,
            fn=Add(), write_back_mu=False, compute_bw=1024,
        )
        return weight_out

    Q_out = rope(Q_norm, cos_load, sin_load, N_HEAD, HEAD_DIM)
    K_out = rope(K_norm, cos_load, sin_load, N_HEAD, HEAD_DIM)

    print(Q_out._stream.shape, Q_out._stream.stream_dtype, K_out._stream.shape, K_out._stream.stream_dtype, V._stream.shape, V._stream.stream_dtype)

    combined = StaticReassemble(
        graph=step_graph, inputs=[Q_out, K_out, V],merge_rank=2
    )

    print(combined._stream.shape, combined._stream.stream_dtype)

    output = OffChipStore(
        graph=step_graph, input=combined, 
        par_dispatch=PAR_DISPATCH, store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
