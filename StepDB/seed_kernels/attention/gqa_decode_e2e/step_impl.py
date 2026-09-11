"""STeP implementation: ragged-batched GQA decode + O-projection.

Implements ``seed_kernels/attention/gqa_decode_e2e/reference.py`` with no
host-side data movement on the cache: the underlying KV tensors keep their
natural ``[B, S, Hkv, D]`` precompute layout. Per-(b, h_kv) tile reads are
expressed entirely via ``LinearOffChipLoad`` strides — a single-element
stride pattern walks the cache in ``(b, h_kv, n)`` order without an
explicit permute.

Ragged per-batch ``seq_lens`` are emulated by uniform-S compute plus a
per-tile mask multiplication after ``exp``: ``exp_masked = exp * mask``
zeros out invalid-tile contributions so the per-batch ``num`` and ``sum``
match the ragged reference.

Skips max-subtraction softmax (mathematically equivalent to the reference's
max-sub form for ``head_dim<=128`` and the test seq_len ranges; matches
``flashattn.py``'s no-max-sub choice).

Layout flow (leading singleton-1 from LinearOffChipLoad's prepended dim):

  Stage 1 — loads
    K_cache : tile_row=1, tile_col=D, stride=(S*Hkv, 1, Hkv),
              out_shape_tiled=(B, Hkv, S)
              -> stream (1, B, Hkv, S) / (1, D)
              [stride walks underlying [B, S, Hkv, D] in (b, h, n) order]
    V_cache : same shape and stride as K_cache
    Q       : tile_row=qpkv, tile_col=D, stride=(Hkv, 1),
              out_shape_tiled=(B, Hkv)
              -> stream (1, B, Hkv) / (qpkv, D)
    mask    : tile_row=1, tile_col=1, stride=(S/tile_seq, 0, 1),
              out_shape_tiled=(B, Hkv, S/tile_seq)
              -> stream (1, B, Hkv, S/tile_seq) / (1, 1)
    O_w     : tile_row=H*D, tile_col=tile_hidden, stride=(0, 1),
              out_shape_tiled=(B, n_hid_tiles)
              -> stream (1, B, n_hid_tiles) / (H*D, tile_hidden)

  Stage 2 — group K/V row-tiles into seq tiles
    ReshapePadStream(chunk=tile_seq, reshape_rank=0)  on (1, B, Hkv, S)
        -> (1, B, Hkv, S/tile_seq, tile_seq) / (1, D)
    Accum(RetileRow, accum_rank=1, init=Empty(0, D))
        -> (1, B, Hkv, S/tile_seq) / (tile_seq, D)

  Stage 3 — Q expand to seq dim
    RepeatStatic(Q, S/tile_seq)
        -> (1, B, Hkv, S/tile_seq) / (qpkv, D)

  Stage 4 — attention compute (no max-sub)
    qkt = Matmul(weight_transposed=True)        / (qpkv, tile_seq)
    exp = Exp(qkt)                              / (qpkv, tile_seq)
    exp_masked = Mul(exp, mask)   # broadcast [1,1] -> [qpkv, tile_seq]
    Broadcast(exp_masked, 2)
    num = exp_masked @ V (BinaryMapAccum, rank=1)
        -> (1, B, Hkv) / (qpkv, D)
    sum_tiles = Accum(Add, accum_rank=1)
        -> (1, B, Hkv) / (qpkv, tile_seq)
    sum_row = RowWiseSum(sum_tiles)             / (qpkv, 1)
    softmax = Div(num, sum_row)                 / (qpkv, D)

  Stage 5 — O-projection
    RetileStreamify(split_row=True)             -> (1, B, Hkv*qpkv) / (1, D)
    Accum(RetileCol, accum_rank=1, out=(1, H*D))-> (1, B) / (1, H*D)
    RepeatStatic(n_hid_tiles)                   -> (1, B, n_hid_tiles) / (1, H*D)
    BinaryMap(Matmul) with O_w                  -> (1, B, n_hid_tiles) / (1, tile_hidden)
    Accum(RetileCol, accum_rank=1, out=(1, hidden_dim))
                                                 -> (1, B) / (1, hidden_dim)
    OffChipStore.
"""

SEED = 42


def build_graph(dims, tensors):
    batch = dims["batch"]
    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    hidden_dim = dims["hidden_dim"]
    seq_len_max = dims["seq_len_max"]
    tile_seq = dims["tile_seq"]
    tile_hidden = dims["tile_hidden"]
    PAR_DISPATCH = dims.get("par_dispatch", 4)

    qpkv = num_heads // num_kv_heads
    n_seq_tiles = seq_len_max // tile_seq
    n_hid_tiles = hidden_dim // tile_hidden
    H = num_heads
    Hkv = num_kv_heads
    D = head_dim
    B = batch
    S = seq_len_max

    Q = tensors["Q"]                           # [B, H, D]
    k_cache = tensors["k_cache"]               # [B, S, Hkv, D]   (natural)
    v_cache = tensors["v_cache"]               # [B, S, Hkv, D]   (natural)
    tile_mask = tensors["tile_mask"]           # [B, n_seq_tiles] float32 0/1
    o_proj_weight = tensors["o_proj_weight"]   # [H*D, hidden_dim]

    step_graph = Graph()

    # ============================================================
    # Stage 1 — loads. Cache underlying stays [B, S, Hkv, D] (natural).
    #          tensor_shape_tiled = (B, S, Hkv, 1) with tile_row=1, tile_col=D.
    #          Flat row-major position (b, n, h, 0) is at index
    #              b*S*Hkv + n*Hkv + h.
    #          For out_shape_tiled=(B, Hkv, S) we want tile (i=b, j=h, k=n)
    #          to address that same position, i.e. flat = b*S*Hkv + h + n*Hkv,
    #          so stride=(S*Hkv, 1, Hkv) walks the underlying in (b, h, n).
    # ============================================================
    load_k = LinearOffChipLoad(
        underlying=k_cache,
        stride=(S * Hkv, 1, Hkv),
        out_shape_tiled=(B, Hkv, S),
        tile_row=1, tile_col=D,
        par_dispatch=PAR_DISPATCH,
    )
    load_v = LinearOffChipLoad(
        underlying=v_cache,
        stride=(S * Hkv, 1, Hkv),
        out_shape_tiled=(B, Hkv, S),
        tile_row=1, tile_col=D,
        par_dispatch=PAR_DISPATCH,
    )
    # load_k, load_v: (1, B, Hkv, S) / (1, D)

    # Q underlying [B, H, D] = [B, Hkv*qpkv, D]. tile_row=qpkv yields
    # tensor_shape_tiled (B, Hkv, 1); stride (Hkv, 1) walks tile (b, h_kv).
    load_q = LinearOffChipLoad(
        underlying=Q,
        stride=(Hkv, 1),
        out_shape_tiled=(B, Hkv),
        tile_row=qpkv, tile_col=D,
        par_dispatch=PAR_DISPATCH,
    )
    # load_q: (1, B, Hkv) / (qpkv, D)

    # tile_mask underlying [B, n_seq_tiles]. tile_row=1, tile_col=1 yields
    # tensor_shape_tiled (B, n_seq_tiles). Broadcast over Hkv via stride 0.
    load_mask = LinearOffChipLoad(
        underlying=tile_mask,
        stride=(n_seq_tiles, 0, 1),
        out_shape_tiled=(B, Hkv, n_seq_tiles),
        tile_row=1, tile_col=1,
        par_dispatch=PAR_DISPATCH,
    )
    # load_mask: (1, B, Hkv, n_seq_tiles) / (1, 1)

    # ============================================================
    # Stage 2 — group K/V row-tiles into seq tiles
    # ============================================================
    k_grouped = ReshapePadStream(
        graph=step_graph, input=load_k,
        chunk_size=tile_seq, reshape_rank=0,
        write_back_mu=False, pad_fn=None, have_pad_stream=False,
    )
    # k_grouped: (1, B, Hkv, n_seq_tiles, tile_seq) / (1, D)
    k_seq = Accum(
        graph=step_graph, input=k_grouped,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_seq, D)),
        fn=accum_fn.RetileRow(),
        init_fn=Empty(shape=(0, D), dtype=Float32()),
        accum_rank=1, write_back_mu=False, compute_bw=1024,
    )
    # k_seq: (1, B, Hkv, n_seq_tiles) / (tile_seq, D)

    v_grouped = ReshapePadStream(
        graph=step_graph, input=load_v,
        chunk_size=tile_seq, reshape_rank=0,
        write_back_mu=False, pad_fn=None, have_pad_stream=False,
    )
    v_seq = Accum(
        graph=step_graph, input=v_grouped,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(tile_seq, D)),
        fn=accum_fn.RetileRow(),
        init_fn=Empty(shape=(0, D), dtype=Float32()),
        accum_rank=1, write_back_mu=False, compute_bw=1024,
    )

    # ============================================================
    # Stage 3 — Q expand to (1, B, Hkv, n_seq_tiles)
    # ============================================================
    q_repeated = RepeatStatic(
        graph=step_graph, input=load_q, repeat_factor=n_seq_tiles,
    )
    # q_repeated: (1, B, Hkv, n_seq_tiles) / (qpkv, D)

    # ============================================================
    # Stage 4 — attention compute (no max-sub) with per-tile masking
    # ============================================================
    qkt = BinaryMap(
        graph=step_graph, in1=q_repeated, in2=k_seq,
        fn=Matmul(weight_transposed=True),
        write_back_mu=False, compute_bw=1024,
    )
    # qkt: (1, B, Hkv, n_seq_tiles) / (qpkv, tile_seq)

    exp_qkt = UnaryMap(
        graph=step_graph, input=qkt,
        fn=Exp(), write_back_mu=False, compute_bw=1024,
    )

    # Mul broadcasts (1, 1) tile against (qpkv, tile_seq) tile element-wise
    # (see step_py.functions.map_fn.Mul.apply). Invalid-tile mask=0 zeroes
    # out the exp tile entirely, contributing nothing to num/sum below.
    exp_masked = BinaryMap(
        graph=step_graph, in1=exp_qkt, in2=load_mask,
        fn=Mul(), write_back_mu=False, compute_bw=1024,
    )

    exp_branches = Broadcast(step_graph, exp_masked, 2)

    num = BinaryMapAccum(
        graph=step_graph, in1=(exp_branches, 0), in2=v_seq,
        fn=MapAccumMatmul(),
        init_fn=Zero(shape=(qpkv, D), dtype=Float32()),
        rank=1, write_back_mu=False, compute_bw=1024,
    )
    # num: (1, B, Hkv) / (qpkv, D)

    tile_shape_exp = (qpkv, tile_seq)
    sum_tiles = Accum(
        graph=step_graph, input=(exp_branches, 1),
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=tile_shape_exp),
        fn=AccumAdd(),
        init_fn=Zero(shape=tile_shape_exp, dtype=Float32()),
        accum_rank=1, write_back_mu=False, compute_bw=1024,
    )
    sum_row = UnaryMap(
        graph=step_graph, input=sum_tiles,
        fn=RowWiseSum(), write_back_mu=False, compute_bw=1024,
    )
    # sum_row: (1, B, Hkv) / (qpkv, 1)

    softmax = BinaryMap(
        graph=step_graph, in1=num, in2=sum_row,
        fn=Div(), write_back_mu=False, compute_bw=1024,
    )
    # softmax: (1, B, Hkv) / (qpkv, D)

    # ============================================================
    # Stage 5 — reassemble for O-proj and matmul
    # ============================================================
    split_rows = RetileStreamify(
        graph=step_graph, input=softmax, split_row=True,
    )
    # split_rows: (1, B, Hkv*qpkv) / (1, D)
    attn_flat = Accum(
        graph=step_graph, input=split_rows,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(1, H * D)),
        fn=accum_fn.RetileCol(),
        init_fn=Zero(shape=(1, 0), dtype=Float32()),
        accum_rank=1, write_back_mu=False, compute_bw=1024,
    )
    # attn_flat: (1, B) / (1, H*D)

    attn_rep = RepeatStatic(
        graph=step_graph, input=attn_flat, repeat_factor=n_hid_tiles,
    )
    # attn_rep: (1, B, n_hid_tiles) / (1, H*D)

    load_o_w = LinearOffChipLoad(
        underlying=o_proj_weight,
        stride=(0, 1),
        out_shape_tiled=(B, n_hid_tiles),
        tile_row=H * D, tile_col=tile_hidden,
        par_dispatch=PAR_DISPATCH,
    )
    # load_o_w: (1, B, n_hid_tiles) / (H*D, tile_hidden)

    o_tiled = BinaryMap(
        graph=step_graph, in1=attn_rep, in2=load_o_w,
        fn=Matmul(), write_back_mu=False, compute_bw=1024,
    )
    # o_tiled: (1, B, n_hid_tiles) / (1, tile_hidden)

    o_out = Accum(
        graph=step_graph, input=o_tiled,
        output_stream_dtype=Tile(tile_dtype=Float32(), shape=(1, hidden_dim)),
        fn=accum_fn.RetileCol(),
        init_fn=Zero(shape=(1, 0), dtype=Float32()),
        accum_rank=1, write_back_mu=True, compute_bw=1024,
    )
    # o_out: (1, B) / (1, hidden_dim)

    output = OffChipStore(
        graph=step_graph, input=o_out,
        par_dispatch=PAR_DISPATCH, store_file_name="output",
    )

    step_graph = infer_broadcast(step_graph)
    return step_graph, output
