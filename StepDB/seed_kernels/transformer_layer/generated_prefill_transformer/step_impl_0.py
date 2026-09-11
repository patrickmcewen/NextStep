_StepOps = BinaryMap.__mro__[1]
if not hasattr(_StepOps, 'shape'):
    _StepOps.shape = property(lambda self: tuple(self.stream.shape) + tuple(self.stream.stream_dtype.shape))

class _BranchRef(tuple):

    def __new__(cls, node, idx):
        return tuple.__new__(cls, (node, idx))

    @property
    def shape(self):
        node, idx = self
        s = node.stream_idx(idx)
        return tuple(s.shape) + tuple(s.stream_dtype.shape)

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
        num_heads_q = q_proj.shape[1] // head_dim
        num_heads_kv = k_proj.shape[1] // head_dim
        inp = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=hidden_dim, par_dispatch=8)
        sq = UnaryMap(graph, inp, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / hidden_dim), write_back_mu=False, compute_bw=4096)
        eps = 1e-06
        mean_e = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        rsqrt = UnaryMap(graph, mean_e, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        normed = BinaryMap(graph, inp, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        q_weight = _offchip_load_or_restream(graph, q_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=num_heads_q * head_dim, par_dispatch=8)
        k_weight = _offchip_load_or_restream(graph, k_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=num_heads_kv * head_dim, par_dispatch=8)
        v_weight = _offchip_load_or_restream(graph, v_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=num_heads_kv * head_dim, par_dispatch=8)
        Q_raw = BinaryMap(graph, normed, q_weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        K_raw = BinaryMap(graph, normed, k_weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        V_raw = BinaryMap(graph, normed, v_weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)

        def reshape_proj(tensor, num_heads):
            tensor = RetileStreamify(graph, tensor, split_row=False, chunk=head_dim)
            tensor = Reshape(graph, tensor, chunk_size=num_heads, reshape_rank=0, write_back_mu=False)
            tensor = Accum(graph, tensor, output_stream_dtype=_dsl2step_out_tile(tensor, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(tensor, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            tensor = Flatten(graph, tensor, min_rank=0, max_rank=1)
            return tensor
        Q = reshape_proj(Q_raw, num_heads_q)
        K = reshape_proj(K_raw, num_heads_kv)
        V = reshape_proj(V_raw, num_heads_kv)
        return (Q, K, V)

    def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
        seq_len = Q.shape[0]
        head_dim = Q.shape[-1]
        half_dim = head_dim // 2
        cos = _offchip_load_or_restream(graph, cos, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=head_dim, par_dispatch=8)
        sin = _offchip_load_or_restream(graph, sin, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=head_dim, par_dispatch=8)
        cos = Flatten(graph, cos, min_rank=0, max_rank=1)
        sin = Flatten(graph, sin, min_rank=0, max_rank=1)

        def rms_norm(x):
            x2 = UnaryMap(graph, x, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
            sum_sq = UnaryMap(graph, x2, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
            mean = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / head_dim), write_back_mu=False, compute_bw=4096)
            mean_eps = UnaryMap(graph, mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
            rs = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
            _tmp1 = BinaryMap(graph, x, rs, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            return _tmp1
        Qn = rms_norm(Q)
        Kn = rms_norm(K)
        Qs = RetileStreamify(graph, Qn, split_row=False, chunk=half_dim)
        Ks = RetileStreamify(graph, Kn, split_row=False, chunk=half_dim)
        cos_s = RetileStreamify(graph, cos, split_row=False, chunk=half_dim)
        sin_s = RetileStreamify(graph, sin, split_row=False, chunk=half_dim)
        _parallelize15 = Parallelize(graph, Qs, parallelize_rank=Qs.stream.rank, num_consumers=2)
        Q_parts = [_BranchRef(_parallelize15, _i) for _i in range(2)]
        _parallelize16 = Parallelize(graph, Ks, parallelize_rank=Ks.stream.rank, num_consumers=2)
        K_parts = [_BranchRef(_parallelize16, _i) for _i in range(2)]
        _parallelize17 = Parallelize(graph, cos_s, parallelize_rank=cos_s.stream.rank, num_consumers=2)
        cos_parts = [_BranchRef(_parallelize17, _i) for _i in range(2)]
        _parallelize18 = Parallelize(graph, sin_s, parallelize_rank=sin_s.stream.rank, num_consumers=2)
        sin_parts = [_BranchRef(_parallelize18, _i) for _i in range(2)]
        Q0, Q1 = (Q_parts[0], Q_parts[1])
        K0, K1 = (K_parts[0], K_parts[1])
        cos0, cos1 = (cos_parts[0], cos_parts[1])
        sin0, sin1 = (sin_parts[0], sin_parts[1])
        _tmp2 = BinaryMap(graph, Q0, cos0, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp3 = BinaryMap(graph, Q1, sin0, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp4 = UnaryMap(graph, _tmp3, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        q_left = BinaryMap(graph, _tmp2, _tmp4, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        _tmp5 = BinaryMap(graph, Q1, cos1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp6 = BinaryMap(graph, Q0, sin1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        q_right = BinaryMap(graph, _tmp5, _tmp6, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        _tmp7 = BinaryMap(graph, K0, cos0, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp8 = BinaryMap(graph, K1, sin0, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp9 = UnaryMap(graph, _tmp8, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        k_left = BinaryMap(graph, _tmp7, _tmp9, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        _tmp10 = BinaryMap(graph, K1, cos1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp11 = BinaryMap(graph, K0, sin1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        k_right = BinaryMap(graph, _tmp10, _tmp11, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        Q_comb = StaticReassemble(graph, inputs=[q_left, q_right], merge_rank=_dsl2step_stream([q_left, q_right][0]).rank)
        Q_resh = Reshape(graph, Q_comb, chunk_size=2, reshape_rank=0, write_back_mu=False)
        Q_out = Accum(graph, Q_resh, output_stream_dtype=_dsl2step_out_tile(Q_resh, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(Q_resh, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        K_comb = StaticReassemble(graph, inputs=[k_left, k_right], merge_rank=_dsl2step_stream([k_left, k_right][0]).rank)
        K_resh = Reshape(graph, K_comb, chunk_size=2, reshape_rank=0, write_back_mu=False)
        K_out = Accum(graph, K_resh, output_stream_dtype=_dsl2step_out_tile(K_resh, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(K_resh, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        V_out = V
        return (Q_out, K_out, V_out)

    def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
        Q, K, V = pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, out_shapes=out_shapes, out_perms=(None, None, None))
        Q, K, V = per_head_norm_and_rope(Q, K, V, cos, sin, out_shapes=out_shapes, out_perms=out_perms)
        return (Q, K, V)

    def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
        seq_len = Q.shape[0]
        num_heads = Q.shape[-2]
        kv_heads = K.shape[-2]
        q_per_kv = num_heads // kv_heads
        q_retiled = RetileStreamify(graph, Q, split_row=True, chunk=1)
        q_buf = Bufferize(graph, q_retiled, rank=1)
        q_stride = [q_per_kv, 1, kv_heads * q_per_kv]
        q_stream = Streamify(graph, q_buf, stride=tuple(q_stride), out_shape_tiled=tuple((kv_heads, q_per_kv, seq_len)))
        Qh = Accum(graph, q_stream, output_stream_dtype=_dsl2step_out_tile(q_stream, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(q_stream, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        k_retiled = RetileStreamify(graph, K, split_row=True, chunk=1)
        k_buf = Bufferize(graph, k_retiled, rank=1)
        k_stride = [1, kv_heads]
        k_stream = Streamify(graph, k_buf, stride=tuple(k_stride), out_shape_tiled=tuple((kv_heads, seq_len)))
        k_tile = Accum(graph, k_stream, output_stream_dtype=_dsl2step_out_tile(k_stream, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(k_stream, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Kh = Promote(graph, k_tile, promote_rank=0)
        v_retiled = RetileStreamify(graph, V, split_row=True, chunk=1)
        v_buf = Bufferize(graph, v_retiled, rank=1)
        v_stride = [1, kv_heads]
        v_stream = Streamify(graph, v_buf, stride=tuple(v_stride), out_shape_tiled=tuple((kv_heads, seq_len)))
        v_tile = Accum(graph, v_stream, output_stream_dtype=_dsl2step_out_tile(v_stream, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(v_stream, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Vh = Promote(graph, v_tile, promote_rank=0)
        return (Qh, Kh, Vh)

    def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
        K_exp = ExpandRef(graph, Kh, ref=Qh, expand_rank=1)
        V_exp = ExpandRef(graph, Vh, ref=Qh, expand_rank=1)
        scores = BinaryMap(graph, Qh, K_exp, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=4096)
        scores_prom = Promote(graph, scores, promote_rank=0)
        scores_split = RetileStreamify(graph, scores_prom, split_row=False, chunk=1)
        row_max = Accum(graph, scores_split, output_stream_dtype=_dsl2step_out_tile(scores_split, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(scores_split, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        _tmp12 = UnaryMap(graph, row_max, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        scores_centered = BinaryMap(graph, scores, _tmp12, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        e = UnaryMap(graph, scores_centered, fn=map_fn.Exp(), write_back_mu=False, compute_bw=4096)
        num = BinaryMap(graph, e, V_exp, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        denom = UnaryMap(graph, e, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        denom = UnaryMap(graph, denom, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        attn = BinaryMap(graph, num, denom, fn=map_fn.Div(), write_back_mu=False, compute_bw=4096)
        attn_flat = Flatten(graph, attn, min_rank=0, max_rank=1)
        attn_stream = RetileStreamify(graph, attn_flat, split_row=True, chunk=1)
        seq_len = out_shapes[0][0]
        _parallelize19 = Parallelize(graph, attn_stream, parallelize_rank=attn_stream.stream.rank, num_consumers=seq_len)
        substreams = [_BranchRef(_parallelize19, _i) for _i in range(seq_len)]
        _eager_merge20 = EagerMerge(graph, substreams, input_rank=1)
        merged = _BranchRef(_eager_merge20, 0)
        _ = _BranchRef(_eager_merge20, 1)
        heads = out_shapes[0][1]
        reshaped = Reshape(graph, merged, chunk_size=heads, reshape_rank=0, write_back_mu=False)
        out = Accum(graph, reshaped, output_stream_dtype=_dsl2step_out_tile(reshaped, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(reshaped, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        return out

    def attention(Q, K, V, *, out_shapes, out_perms=None):
        Qh, Kh, Vh = compute_qkv(Q, K, V, out_shapes=((4, 4, 64, 32), (4, 1, 64, 32), (4, 1, 64, 32)), out_perms=(None, None, None))
        attn = attention_compute(Qh, Kh, Vh, out_shapes=out_shapes, out_perms=out_perms)
        return attn

    def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
        seq_len = Q.shape[0]
        num_heads = Q.shape[1]
        head_dim = Q.shape[2]
        proj_dim = num_heads * head_dim
        attn = attention(Q, K, V, out_shapes=((seq_len, num_heads, head_dim),), out_perms=(None,))
        attn_retiled = RetileStreamify(graph, attn, split_row=True, chunk=1)
        attn_reshaped = Reshape(graph, attn_retiled, chunk_size=num_heads, reshape_rank=0, write_back_mu=False)
        attn_flat = Accum(graph, attn_reshaped, output_stream_dtype=_dsl2step_out_tile(attn_reshaped, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn_reshaped, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        weight_raw = _offchip_load_or_restream(graph, o_proj_weight, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=proj_dim, tile_col=proj_dim, par_dispatch=8, transposed=False)
        weight = Flatten(graph, weight_raw, min_rank=0, max_rank=1)
        input_raw = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=proj_dim, par_dispatch=8, transposed=False)
        input_stream = Flatten(graph, input_raw, min_rank=0, max_rank=1)
        proj = BinaryMap(graph, attn_flat, weight, fn=map_fn.Matmul(weight_transposed=False), write_back_mu=False, compute_bw=4096)
        result = BinaryMap(graph, proj, input_stream, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return result

    def rms_norm(res_add_0, *, out_shapes, out_perms=None):
        x_sq = UnaryMap(graph, res_add_0, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sum_sq = UnaryMap(graph, x_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        hidden_dim = out_shapes[0][2]
        inv_hidden = 1.0 / hidden_dim
        mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(inv_hidden), write_back_mu=False, compute_bw=4096)
        eps = 1e-06
        var = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        inv_rms = UnaryMap(graph, var, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        out = BinaryMap(graph, res_add_0, inv_rms, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        return out

    def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        n_experts = w_gate.shape[0]
        expert_onehot_sel = SelectGen(is_multihot=True, tensor=expert_onehot, n=n_experts)
        graph.add_node(expert_onehot_sel)
        normed_outer = PromoteOuter(graph, normed_2)
        normed_rep = RepeatStatic(graph, normed_outer, repeat_factor=2)
        expert_weights_stream = _offchip_load_or_restream(graph, expert_weights, stride=tuple((2, 1)), out_shape_tiled=tuple((64, 2)), tile_row=1, tile_col=1, par_dispatch=8)
        _flat_partition21 = FlatPartition(graph, normed_rep, control=expert_onehot_sel, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
        tokens_per_expert = [_BranchRef(_flat_partition21, _i) for _i in range(n_experts)]
        _flat_partition22 = FlatPartition(graph, expert_weights_stream, control=expert_onehot_sel, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
        weights_per_expert = [_BranchRef(_flat_partition22, _i) for _i in range(n_experts)]
        per_expert_contrib = []
        for e_idx in range(n_experts):
            gate_w = _offchip_load_or_restream(graph, w_gate[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=512, tile_col=1792, par_dispatch=8)
            up_w = _offchip_load_or_restream(graph, w_up[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=512, tile_col=1792, par_dispatch=8)
            down_w = _offchip_load_or_restream(graph, w_down[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=1792, tile_col=512, par_dispatch=8)
            tok = Promote(graph, tokens_per_expert[e_idx], promote_rank=0)
            w_scalar = Promote(graph, weights_per_expert[e_idx], promote_rank=0)
            gate_w = ExpandRef(graph, gate_w, ref=tok, expand_rank=2)
            up_w = ExpandRef(graph, up_w, ref=tok, expand_rank=2)
            down_w = ExpandRef(graph, down_w, ref=tok, expand_rank=2)
            gate_out = BinaryMap(graph, tok, gate_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            up_out = BinaryMap(graph, tok, up_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            _tmp13 = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
            hidden = BinaryMap(graph, _tmp13, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            down_out = BinaryMap(graph, hidden, down_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            weighted = BinaryMap(graph, down_out, w_scalar, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            per_expert_contrib.append(weighted)
        reassembled = FlatReassemble(graph, inputs=per_expert_contrib, control=expert_onehot_sel, reassemble_rank=0, switch_cycles=[1] * len(per_expert_contrib), write_back_mu=False)
        summed_nactive = Accum(graph, reassembled, output_stream_dtype=_dsl2step_out_tile(reassembled, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(reassembled, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        summed_pos = Accum(graph, summed_nactive, output_stream_dtype=_dsl2step_out_tile(summed_nactive, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(summed_nactive, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        final_stream = Flatten(graph, summed_pos, min_rank=0, max_rank=1)
        return final_stream

    def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        normed_2 = rms_norm(res_add_0, out_shapes=out_shapes, out_perms=out_perms)
        return moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes, out_perms=out_perms)

    def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        moe_out = moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes, out_perms=out_perms)
        _tmp14 = BinaryMap(graph, moe_out, res_add_0, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return _tmp14
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
    seq_len = input_tensor.shape[0]
    hidden = input_tensor.shape[1]
    head_dim = cos.shape[-1]
    q_heads = hidden // head_dim
    kv_heads = k_proj.shape[1] // head_dim
    q_shape = (seq_len, q_heads, head_dim)
    k_shape = (seq_len, kv_heads, head_dim)
    v_shape = (seq_len, kv_heads, head_dim)
    out_shape = (seq_len, 1, hidden)
    Q, K, V = pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, out_shapes=(q_shape, k_shape, v_shape))
    res_add_0 = attention_o_proj(Q, K, V, o_proj_weight, input_tensor, out_shapes=(out_shape,))
    out = moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=(out_shape,))
    _store23 = OffChipStore(graph, out, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store23)