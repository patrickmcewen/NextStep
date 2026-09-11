_StepOps = BinaryMap.__mro__[1]
if not hasattr(_StepOps, 'shape'):
    _StepOps.shape = property(lambda self: tuple(self.stream.shape) + tuple(self.stream.stream_dtype.shape))
if not hasattr(_StepOps, 'stream_dtype'):
    _StepOps.stream_dtype = property(lambda self: self.stream.stream_dtype)

class _BranchRef(tuple):

    def __new__(cls, node, idx):
        return tuple.__new__(cls, (node, idx))

    @property
    def shape(self):
        node, idx = self
        s = node.stream_idx(idx)
        return tuple(s.shape) + tuple(s.stream_dtype.shape)

    @property
    def stream_dtype(self):
        node, idx = self
        return node.stream_idx(idx).stream_dtype

def _dsl2step_stream(x):
    if isinstance(x, tuple) and len(x) == 2:
        node, idx = x
        return node.stream_idx(idx)
    return x.stream

def _dsl2step_out_tile(x, mode, accum_rank):
    s = _dsl2step_stream(x)
    sd = s.stream_dtype
    if mode == 'elem':
        return sd
    dims = s.shape[-accum_rank:]
    if any((not isinstance(d, int) for d in dims)):
        return sd
    tr, tc = sd.shape
    mul = 1
    for d in dims:
        mul *= d
    if mode == 'row':
        return Tile(tile_dtype=sd.tile_dtype, shape=(tr * mul, tc))
    return Tile(tile_dtype=sd.tile_dtype, shape=(tr, tc * mul))

def _dsl2step_init(x, mode):
    sd = _dsl2step_stream(x).stream_dtype
    tr, tc = sd.shape
    dt = sd.tile_dtype
    if mode == 'row':
        return Empty(shape=(0, tc), dtype=dt)
    if mode == 'col':
        return Empty(shape=(tr, 0), dtype=dt)
    return Zero(shape=(tr, tc), dtype=dt)

def _dsl2step_in_tile(x):
    return _dsl2step_stream(x).stream_dtype

def _seal_unused_branches(graph):
    for node in list(graph.nodes):
        n_branches = getattr(node, 'num_consumers', None)
        if n_branches is None:
            continue
        used = set()
        for consumer in graph.successors(node):
            for inp in consumer.input_list:
                if isinstance(inp, tuple) and len(inp) == 2 and (inp[0] is node):
                    used.add(inp[1])
        for idx in range(n_branches):
            if idx not in used:
                ConsumerContext(graph, (node, idx))

def _offchip_load_or_restream(graph, underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False, par_dispatch=8):
    if not isinstance(underlying, (_StepOps, _BranchRef)):
        node = LinearOffChipLoad(underlying, stride=tuple(stride), out_shape_tiled=tuple(out_shape_tiled), tile_row=tile_row, tile_col=tile_col, transposed=transposed, par_dispatch=par_dispatch)
        graph.add_node(node)
        return node
    import warnings
    warnings.warn(f'offchip_load: underlying is an on-chip stream ({type(underlying).__name__}); substituting a restream lowering. The refactor should use streamify / retile_streamify / restream — this backstop keeps the graph buildable but downstream correctness is not guaranteed.', stacklevel=2)
    rs1 = RetileStreamify(graph, underlying, split_row=True, chunk=1)
    rs2 = RetileStreamify(graph, rs1, split_row=False, chunk=1)
    buf = Bufferize(graph, rs2, rank=len(rs2.stream.shape))
    tile_area = tile_row * tile_col
    scaled_stride = tuple((s * tile_area for s in stride)) + (tile_col, 1)
    extended_shape = tuple(out_shape_tiled) + (tile_row, tile_col)
    sm = Streamify(graph, buf, stride=scaled_stride, out_shape_tiled=extended_shape)
    rcol = Accum(graph, sm, output_stream_dtype=_dsl2step_out_tile(sm, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(sm, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    rrow = Accum(graph, rcol, output_stream_dtype=_dsl2step_out_tile(rcol, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(rcol, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    return PromoteOuter(graph, rrow)

def build_graph(dims, tensors):
    graph = Graph()

    def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
        seq_len = input_tensor.shape[0]
        hidden_dim = input_tensor.shape[1]
        head_dim = cos.shape[-1]
        eps = 1e-06
        inp = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=hidden_dim, par_dispatch=8)
        sq = UnaryMap(graph, inp, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / hidden_dim), write_back_mu=False, compute_bw=4096)
        mean_eps = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        rsqrt = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        normed = BinaryMap(graph, inp, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        out_dim_q = q_proj.shape[1]
        num_heads = out_dim_q // head_dim
        q_weight = _offchip_load_or_restream(graph, q_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=out_dim_q, par_dispatch=8)
        q_mat = BinaryMap(graph, normed, q_weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        q_split = RetileStreamify(graph, q_mat, split_row=False, chunk=head_dim)
        Q = Reshape(graph, q_split, chunk_size=num_heads, reshape_rank=0, write_back_mu=False)
        Q = Accum(graph, Q, output_stream_dtype=_dsl2step_out_tile(Q, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(Q, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Q = Flatten(graph, Q, min_rank=0, max_rank=1)
        out_dim_k = k_proj.shape[1]
        num_kv_heads = out_dim_k // head_dim
        k_weight = _offchip_load_or_restream(graph, k_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=out_dim_k, par_dispatch=8)
        k_mat = BinaryMap(graph, normed, k_weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        k_split = RetileStreamify(graph, k_mat, split_row=False, chunk=head_dim)
        K = Reshape(graph, k_split, chunk_size=num_kv_heads, reshape_rank=0, write_back_mu=False)
        K = Accum(graph, K, output_stream_dtype=_dsl2step_out_tile(K, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(K, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        K = Flatten(graph, K, min_rank=0, max_rank=1)
        out_dim_v = v_proj.shape[1]
        v_weight = _offchip_load_or_restream(graph, v_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=out_dim_v, par_dispatch=8)
        v_mat = BinaryMap(graph, normed, v_weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        v_split = RetileStreamify(graph, v_mat, split_row=False, chunk=head_dim)
        V = Reshape(graph, v_split, chunk_size=num_kv_heads, reshape_rank=0, write_back_mu=False)
        V = Accum(graph, V, output_stream_dtype=_dsl2step_out_tile(V, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(V, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        V = Flatten(graph, V, min_rank=0, max_rank=1)
        return (Q, K, V)

    def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
        seq_len = Q.shape[0]
        head_dim = Q.stream_dtype.shape[1]
        cos_s = _offchip_load_or_restream(graph, cos, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=head_dim, par_dispatch=8)
        sin_s = _offchip_load_or_restream(graph, sin, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=head_dim, par_dispatch=8)
        cos_s = Flatten(graph, cos_s, min_rank=0, max_rank=1)
        sin_s = Flatten(graph, sin_s, min_rank=0, max_rank=1)

        def rms_norm(x):
            eps = 1e-06
            x_sq = UnaryMap(graph, x, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
            sum_sq = UnaryMap(graph, x_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
            mean = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / x.stream_dtype.shape[1]), write_back_mu=False, compute_bw=4096)
            mean_eps = UnaryMap(graph, mean, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
            rsqrt = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
            _tmp1 = BinaryMap(graph, x, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            return _tmp1

        def rotate_half(x):
            half = x.stream_dtype.shape[1] // 2
            split = RetileStreamify(graph, x, split_row=False, chunk=half)
            _parallelize12 = Parallelize(graph, split, parallelize_rank=split.stream.rank, num_consumers=2)
            halves = [_BranchRef(_parallelize12, _i) for _i in range(2)]
            first_half = halves[0]
            second_half = halves[1]
            second_half = UnaryMap(graph, second_half, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
            merged = StaticReassemble(graph, inputs=[second_half, first_half], merge_rank=_dsl2step_stream([second_half, first_half][0]).rank)
            merged = Reshape(graph, merged, chunk_size=2, reshape_rank=0, write_back_mu=False)
            _tmp2 = Accum(graph, merged, output_stream_dtype=_dsl2step_out_tile(merged, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(merged, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            return _tmp2
        Q_norm = rms_norm(Q)
        K_norm = rms_norm(K)
        Q_rot = rotate_half(Q_norm)
        K_rot = rotate_half(K_norm)
        _tmp3 = BinaryMap(graph, Q_norm, cos_s, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp4 = BinaryMap(graph, Q_rot, sin_s, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q_out = BinaryMap(graph, _tmp3, _tmp4, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        _tmp5 = BinaryMap(graph, K_norm, cos_s, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp6 = BinaryMap(graph, K_rot, sin_s, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K_out = BinaryMap(graph, _tmp5, _tmp6, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return (Q_out, K_out, V)

    def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
        Q, K, V = pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, out_shapes=out_shapes, out_perms=out_perms)
        Q, K, V = per_head_norm_and_rope(Q, K, V, cos, sin, out_shapes=out_shapes, out_perms=out_perms)
        return (Q, K, V)

    def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
        """
    Convert the on‑chip tensors Q, K, V to the layout required by the reference
    model:

        Qh : (kv_heads, q_per_kv, seq_len, dim)
        Kh : (kv_heads, 1,       seq_len, dim)
        Vh : (kv_heads, 1,       seq_len, dim)

    All tensors are StepTensors (stream shape + tile shape).  The DSL operators
    used only manipulate stream dimensions; they never touch the underlying
    torch.Tensor directly.

    Steps (the same for Q, K and V, with a slight difference for Q):
    1. `retile_streamify` moves the tile‑row (heads / kv_heads) into the stream,
       leaving a unit tile‑row.
    2. `parallelize` splits the stream cyclically by the number of heads
       (or kv‑heads) and returns a list of sub‑streams, each containing the
       values for a single head in token order.
    3. `eager_merge` concatenates the sub‑streams back, yielding a stream whose
       order is head‑major (head0‑seq0…seq63, head1‑seq0…seq63, …).
    4. `reshape_stream` first splits the stream into (heads, seq_len) and then
       splits the head dimension into (kv_heads, q_per_kv) for Q, or inserts a
       singleton dimension for K/V.
    5. `accum_retile_row` absorbs the innermost stream dimension (seq_len) into
       the tile‑row, producing the final tile shape (seq_len, dim).

    The required numeric parameters (seq_len, heads, kv_heads, q_per_kv) are
    obtained from the StepTensor shapes via the public `.shape` accessor.
    """
        seq_len = Q.shape[0]
        heads = Q.shape[1]
        kv_heads = K.shape[1]
        q_per_kv = heads // kv_heads
        q = RetileStreamify(graph, Q, split_row=True, chunk=1)
        _parallelize13 = Parallelize(graph, q, parallelize_rank=q.stream.rank, num_consumers=heads)
        q_sub = [_BranchRef(_parallelize13, _i) for _i in range(heads)]
        _eager_merge14 = EagerMerge(graph, q_sub, input_rank=1)
        _tmp7 = [_BranchRef(_eager_merge14, _i) for _i in range(2)]
        q = _tmp7[0]
        q = Reshape(graph, q, chunk_size=seq_len, reshape_rank=0, write_back_mu=False)
        q = Reshape(graph, q, chunk_size=q_per_kv, reshape_rank=1, write_back_mu=False)
        q = Accum(graph, q, output_stream_dtype=_dsl2step_out_tile(q, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(q, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        k = RetileStreamify(graph, K, split_row=True, chunk=1)
        _parallelize15 = Parallelize(graph, k, parallelize_rank=k.stream.rank, num_consumers=kv_heads)
        k_sub = [_BranchRef(_parallelize15, _i) for _i in range(kv_heads)]
        _eager_merge16 = EagerMerge(graph, k_sub, input_rank=1)
        _tmp8 = [_BranchRef(_eager_merge16, _i) for _i in range(2)]
        k = _tmp8[0]
        k = Reshape(graph, k, chunk_size=seq_len, reshape_rank=0, write_back_mu=False)
        k = Reshape(graph, k, chunk_size=1, reshape_rank=1, write_back_mu=False)
        k = Accum(graph, k, output_stream_dtype=_dsl2step_out_tile(k, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(k, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        v = RetileStreamify(graph, V, split_row=True, chunk=1)
        _parallelize17 = Parallelize(graph, v, parallelize_rank=v.stream.rank, num_consumers=kv_heads)
        v_sub = [_BranchRef(_parallelize17, _i) for _i in range(kv_heads)]
        _eager_merge18 = EagerMerge(graph, v_sub, input_rank=1)
        _tmp9 = [_BranchRef(_eager_merge18, _i) for _i in range(2)]
        v = _tmp9[0]
        v = Reshape(graph, v, chunk_size=seq_len, reshape_rank=0, write_back_mu=False)
        v = Reshape(graph, v, chunk_size=1, reshape_rank=1, write_back_mu=False)
        v = Accum(graph, v, output_stream_dtype=_dsl2step_out_tile(v, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(v, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        return (q, k, v)

    def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
        Kh_exp = ExpandRef(graph, Kh, ref=Qh, expand_rank=1)
        Vh_exp = ExpandRef(graph, Vh, ref=Qh, expand_rank=1)
        scores = BinaryMap(graph, Qh, Kh_exp, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=4096)
        scores_p = Promote(graph, scores, promote_rank=0)
        scores_spl = RetileStreamify(graph, scores_p, split_row=False, chunk=1)
        row_max = Accum(graph, scores_spl, output_stream_dtype=_dsl2step_out_tile(scores_spl, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(scores_spl, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        _tmp10 = UnaryMap(graph, row_max, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        centered = BinaryMap(graph, scores, _tmp10, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        e = UnaryMap(graph, centered, fn=map_fn.Exp(), write_back_mu=False, compute_bw=4096)
        num = BinaryMap(graph, e, Vh_exp, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        denom = UnaryMap(graph, e, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        attn = BinaryMap(graph, num, denom, fn=map_fn.Div(), write_back_mu=False, compute_bw=4096)
        attn_seq = RetileStreamify(graph, attn, split_row=True, chunk=1)
        attn_flat = Flatten(graph, attn_seq, min_rank=0, max_rank=1)
        seq_len = out_shapes[0][0]
        _parallelize19 = Parallelize(graph, attn_flat, parallelize_rank=attn_flat.stream.rank, num_consumers=seq_len)
        streams = [_BranchRef(_parallelize19, _i) for _i in range(seq_len)]
        processed = []
        for s in streams:
            s_outer = PromoteOuter(graph, s)
            s_tile = Accum(graph, s_outer, output_stream_dtype=_dsl2step_out_tile(s_outer, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(s_outer, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            processed.append(s_tile)
        final = StaticReassemble(graph, inputs=processed, merge_rank=_dsl2step_stream(processed[0]).rank)
        return final

    def attention(Q, K, V, *, out_shapes, out_perms=None):
        Qh, Kh, Vh = compute_qkv(Q, K, V, out_shapes=((4, 4, 64, 32), (4, 1, 64, 32), (4, 1, 64, 32)), out_perms=(None, None, None))
        attn = attention_compute(Qh, Kh, Vh, out_shapes=out_shapes, out_perms=out_perms)
        return attn

    def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
        attn = attention(Q, K, V, out_shapes=((64, 16, 32),), out_perms=(None,))
        attn_retiled = RetileStreamify(graph, attn, split_row=True, chunk=1)
        attn_reshaped = Reshape(graph, attn_retiled, chunk_size=16, reshape_rank=0, write_back_mu=False)
        attn_flat = Accum(graph, attn_reshaped, output_stream_dtype=_dsl2step_out_tile(attn_reshaped, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn_reshaped, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        w = _offchip_load_or_restream(graph, o_proj_weight, stride=tuple((0,)), out_shape_tiled=tuple((64,)), tile_row=512, tile_col=512, par_dispatch=8)
        w = Flatten(graph, w, min_rank=0, max_rank=1)
        x = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((64,)), tile_row=1, tile_col=512, par_dispatch=8)
        x = Flatten(graph, x, min_rank=0, max_rank=1)
        proj = BinaryMap(graph, attn_flat, w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        out = BinaryMap(graph, proj, x, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return out

    def rms_norm(res_add_0, *, out_shapes, out_perms=None):
        eps = 1e-06
        sq = UnaryMap(graph, res_add_0, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / 512.0), write_back_mu=False, compute_bw=4096)
        mean_sq_eps = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        factor = UnaryMap(graph, mean_sq_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        out = BinaryMap(graph, res_add_0, factor, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        return out

    def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        """
    MoE dispatch & aggregation expressed entirely with the Step DSL.

    The implementation follows the reference algorithm:
    • Replicate the token stream over the top‑k (2) dimension.
    • Build a MultiHot control from ``expert_onehot``.
    • Partition the token stream and the per‑token scalar expert‑weight stream
      into per‑expert sub‑streams.
    • For each expert, create a constant address stream (all entries equal to the
      expert index) using ``unary_to_const_int`` and load the three weight
      matrices with ``random_offchip_load``.
    • Perform the two matmuls, SiLU activation, hidden‑multiply, down‑projection,
      and finally scale by the per‑token expert weight.
    • Re‑assemble the per‑expert contributions, sum over the top‑k and the
      intermediate “active‑expert” dimension, and flatten the leading singleton
      with the token dimension to obtain the required output shape
      (seq_len\u202f×\u202f1\u202f×\u202fhidden) – i.e. ``(64, 1, 512)`` for the reference config.
    """
        normed_rep = RepeatStatic(graph, normed_2, repeat_factor=2)
        normed_rep = PromoteOuter(graph, normed_rep)
        control = SelectGen(is_multihot=True, tensor=expert_onehot, n=8)
        graph.add_node(control)
        _flat_partition20 = FlatPartition(graph, normed_rep, control=control, partition_rank=0, switch_cycles=[1] * 8, write_back_mu=False, num_consumers=8)
        per_expert_normed = [_BranchRef(_flat_partition20, _i) for _i in range(8)]
        seq_len = normed_2.shape[0]
        weight_stream = _offchip_load_or_restream(graph, expert_weights, stride=tuple((2, 1)), out_shape_tiled=tuple((seq_len, 2)), tile_row=1, tile_col=1, par_dispatch=8)
        _flat_partition21 = FlatPartition(graph, weight_stream, control=control, partition_rank=0, switch_cycles=[1] * 8, write_back_mu=False, num_consumers=8)
        per_expert_weights = [_BranchRef(_flat_partition21, _i) for _i in range(8)]
        per_expert_outputs = []
        for e_idx in range(8):
            tok_stream = per_expert_normed[e_idx]
            weight_tok = per_expert_weights[e_idx]
            addr_int = UnaryMap(graph, weight_tok, fn=map_fn.ToConstInt(e_idx), write_back_mu=False, compute_bw=4096)
            w_gate_e = RandomOffChipLoad(graph, underlying=w_gate, raddr=addr_int, tile_row=512, tile_col=1792, base_addr_byte=0, par_dispatch=8)
            w_up_e = RandomOffChipLoad(graph, underlying=w_up, raddr=addr_int, tile_row=512, tile_col=1792, base_addr_byte=0, par_dispatch=8)
            w_down_e = RandomOffChipLoad(graph, underlying=w_down, raddr=addr_int, tile_row=1792, tile_col=512, base_addr_byte=0, par_dispatch=8)
            gate_out = BinaryMap(graph, tok_stream, w_gate_e, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            up_out = BinaryMap(graph, tok_stream, w_up_e, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            gate_act = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
            hidden = BinaryMap(graph, gate_act, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            down_out = BinaryMap(graph, hidden, w_down_e, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            contrib = BinaryMap(graph, down_out, weight_tok, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            per_expert_outputs.append(contrib)
        reassembled = FlatReassemble(graph, inputs=per_expert_outputs, control=control, reassemble_rank=0, switch_cycles=[1] * len(per_expert_outputs), write_back_mu=False)
        summed = Accum(graph, reassembled, output_stream_dtype=_dsl2step_out_tile(reassembled, 'elem', 2), fn=accum_fn.Add(), init_fn=_dsl2step_init(reassembled, 'elem'), accum_rank=2, write_back_mu=False, compute_bw=4096)
        final = Flatten(graph, summed, min_rank=0, max_rank=1)
        return final

    def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        normed_2 = rms_norm(res_add_0, out_shapes=((64, 1, 512),), out_perms=(None,))
        return moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes, out_perms=out_perms)

    def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        dispatch = moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes, out_perms=out_perms)
        _tmp11 = BinaryMap(graph, dispatch, res_add_0, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return _tmp11
    input_tensor = tensors['input_tensor']
    q_proj = tensors['q_proj']
    k_proj = tensors['k_proj']
    v_proj = tensors['v_proj']
    cos = tensors['cos']
    sin = tensors['sin']
    o_proj_weight = tensors['o_proj_weight']
    w_gate = tensors['w_gate']
    w_up = tensors['w_up']
    w_down = tensors['w_down']
    expert_weights = tensors['expert_weights']
    expert_onehot = tensors['expert_onehot']
    seq_len = dims['seq_len']
    Q_shape = (seq_len, 16, 32)
    K_shape = (seq_len, 4, 32)
    V_shape = (seq_len, 4, 32)
    Q, K, V = pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, out_shapes=(Q_shape, K_shape, V_shape), out_perms=(None, None, None))
    o_shape = (seq_len, 1, 512)
    res_add_0 = attention_o_proj(Q, K, V, o_proj_weight, input_tensor, out_shapes=(o_shape,), out_perms=(None,))
    out_stream = moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=(o_shape,), out_perms=(None,))
    out_with_outer = PromoteOuter(graph, out_stream)
    _store22 = OffChipStore(graph, out_with_outer, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store22)