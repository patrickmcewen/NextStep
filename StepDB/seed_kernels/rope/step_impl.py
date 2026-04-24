"""STeP implementation: Rotary Position Embedding (RoPE) on Q and K.

Mirrors the rotate_half pipeline from
step_tl/end_to_end/attention/qkv_gen.py:rotate_half / qkv_gen.

Key insight: RoPE is per-head (per-row). Q and K only differ in number of
heads; cos/sin broadcast over the head axis. So we concatenate Q and K along
the heads dim into a single (batch, Nq+Nk, head_dim) tensor and run ONE
rotate_half + cos/sin multiply-add pipeline. Output rows are stacked
per-batch as [B0_Q_h0..B0_Q_h(Nq-1), B0_K_h0..B0_K_h(Nk-1), B1_Q...].
compute_gold matches this via torch.cat([Q_out, K_out], dim=1).

Pipeline (stream shape / tile shape):
  Load QK   -> (1, batch, 1) / (Nq+Nk, head_dim)
  Load cos  -> (1, batch, 1) / (1, head_dim)
  Load sin  -> (1, batch, 1) / (1, head_dim)
  rotate_half(QK)        (qkv_gen.py half-swap pipeline)
  qk_cos = QK * cos      (cos/sin tile (1, head_dim) broadcasts over heads)
  qk_sin = rot * sin
  out    = qk_cos + qk_sin
  OffChipStore -> (batch*(Nq+Nk), head_dim)
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
    batch = dims["batch"]
    num_q_heads = dims["num_q_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    assert head_dim % 2 == 0, f"head_dim={head_dim} must be even"

    total_heads = num_q_heads + num_kv_heads

    torch.manual_seed(SEED)
    Q = torch.randn(batch, num_q_heads, head_dim)
    K = torch.randn(batch, num_kv_heads, head_dim)
    cos = torch.randn(batch, 1, head_dim)
    sin = torch.randn(batch, 1, head_dim)

    # Stack Q and K along the heads dim, then flatten to (batch*total_heads, head_dim).
    QK = torch.cat([Q, K], dim=1).contiguous()
    QK_underlying = QK.reshape(batch * total_heads, head_dim).contiguous()
    cos_underlying = cos.reshape(batch, head_dim).contiguous()
    sin_underlying = sin.reshape(batch, head_dim).contiguous()

    step_graph = Graph()

    qk_load = LinearOffChipLoad(
        underlying=QK_underlying, stride=(1, 1), out_shape_tiled=(batch, 1),
        tile_row=total_heads, tile_col=head_dim, par_dispatch=4,
    )
    cos_load = LinearOffChipLoad(
        underlying=cos_underlying, stride=(1, 1), out_shape_tiled=(batch, 1),
        tile_row=1, tile_col=head_dim, par_dispatch=4,
    )
    sin_load = LinearOffChipLoad(
        underlying=sin_underlying, stride=(1, 1), out_shape_tiled=(batch, 1),
        tile_row=1, tile_col=head_dim, par_dispatch=4,
    )

    qk_rot = _rotate_half(step_graph, qk_load, batch, total_heads, head_dim)

    qk_cos = BinaryMap(
        graph=step_graph, in1=qk_load, in2=cos_load,
        fn=Mul(), write_back_mu=False, compute_bw=1024,
    )
    qk_sin = BinaryMap(
        graph=step_graph, in1=qk_rot, in2=sin_load,
        fn=Mul(), write_back_mu=False, compute_bw=1024,
    )
    qk_out = BinaryMap(
        graph=step_graph, in1=qk_cos, in2=qk_sin,
        fn=Add(), write_back_mu=False, compute_bw=1024,
    )

    output = OffChipStore(
        graph=step_graph, input=qk_out,
        par_dispatch=4, store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
