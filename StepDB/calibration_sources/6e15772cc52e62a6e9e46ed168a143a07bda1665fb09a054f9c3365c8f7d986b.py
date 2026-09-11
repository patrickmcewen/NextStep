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
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim
        eps = 1e-06
        inv_hidden = 1.0 / hidden_dim
        inp = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=seq_len, tile_col=hidden_dim, par_dispatch=8)
        w_q = _offchip_load_or_restream(graph, q_proj, stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=hidden_dim, tile_col=q_proj.shape[1], par_dispatch=8)
        w_k = _offchip_load_or_restream(graph, k_proj, stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=hidden_dim, tile_col=k_proj.shape[1], par_dispatch=8)
        w_v = _offchip_load_or_restream(graph, v_proj, stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=hidden_dim, tile_col=v_proj.shape[1], par_dispatch=8)
        sq = UnaryMap(graph, inp, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sum_rows = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        mean_rows = UnaryMap(graph, sum_rows, fn=map_fn.MulImmediate(inv_hidden), write_back_mu=False, compute_bw=4096)
        mean_eps = UnaryMap(graph, mean_rows, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        rsqrt = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        normed = BinaryMap(graph, inp, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)

        def _project_and_reshape(weight, heads):
            proj = BinaryMap(graph, normed, weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            proj = RetileStreamify(graph, proj, split_row=True, chunk=1)
            proj = RetileStreamify(graph, proj, split_row=False, chunk=head_dim)
            proj = Reshape(graph, proj, chunk_size=heads, reshape_rank=0, write_back_mu=False)
            proj = Accum(graph, proj, output_stream_dtype=_dsl2step_out_tile(proj, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(proj, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            proj = Flatten(graph, proj, min_rank=0, max_rank=1)
            return proj
        Q = _project_and_reshape(w_q, num_heads)
        K = _project_and_reshape(w_k, num_kv_heads)
        V = _project_and_reshape(w_v, num_kv_heads)
        return (Q, K, V)

    def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
        seq_len = Q.shape[0]
        head_dim = Q.shape[-1]
        stride = (1,)
        out_shape = (seq_len,)
        cos_stream = _offchip_load_or_restream(graph, cos, stride=tuple(stride), out_shape_tiled=tuple(out_shape), tile_row=1, tile_col=head_dim, par_dispatch=8)
        sin_stream = _offchip_load_or_restream(graph, sin, stride=tuple(stride), out_shape_tiled=tuple(out_shape), tile_row=1, tile_col=head_dim, par_dispatch=8)
        cos_stream = Flatten(graph, cos_stream, min_rank=0, max_rank=1)
        sin_stream = Flatten(graph, sin_stream, min_rank=0, max_rank=1)
        eps = 1e-06
        inv_head = 1.0 / head_dim
        q_sq = UnaryMap(graph, Q, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        q_sum = UnaryMap(graph, q_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        q_mean = UnaryMap(graph, q_sum, fn=map_fn.MulImmediate(inv_head), write_back_mu=False, compute_bw=4096)
        q_meps = UnaryMap(graph, q_mean, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        q_rsqrt = UnaryMap(graph, q_meps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        Q_norm = BinaryMap(graph, Q, q_rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        k_sq = UnaryMap(graph, K, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        k_sum = UnaryMap(graph, k_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        k_mean = UnaryMap(graph, k_sum, fn=map_fn.MulImmediate(inv_head), write_back_mu=False, compute_bw=4096)
        k_meps = UnaryMap(graph, k_mean, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        k_rsqrt = UnaryMap(graph, k_meps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        K_norm = BinaryMap(graph, K, k_rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        half = head_dim // 2

        def rotate_half(x):
            x_split = RetileStreamify(graph, x, split_row=False, chunk=half)
            _parallelize5 = Parallelize(graph, x_split, parallelize_rank=x_split.stream.rank, num_consumers=2)
            halves = [_BranchRef(_parallelize5, _i) for _i in range(2)]
            neg_second = UnaryMap(graph, halves[1], fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
            interleaved = StaticReassemble(graph, inputs=[neg_second, halves[0]], merge_rank=_dsl2step_stream([neg_second, halves[0]][0]).rank)
            reshaped = Reshape(graph, interleaved, chunk_size=2, reshape_rank=0, write_back_mu=False)
            _tmp1 = Accum(graph, reshaped, output_stream_dtype=_dsl2step_out_tile(reshaped, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(reshaped, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            return _tmp1
        Q_rot = rotate_half(Q_norm)
        K_rot = rotate_half(K_norm)
        Q_cos = BinaryMap(graph, Q_norm, cos_stream, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q_sin = BinaryMap(graph, Q_rot, sin_stream, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q_out = BinaryMap(graph, Q_cos, Q_sin, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        K_cos = BinaryMap(graph, K_norm, cos_stream, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K_sin = BinaryMap(graph, K_rot, sin_stream, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K_out = BinaryMap(graph, K_cos, K_sin, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        V_out = V
        return (Q_out, K_out, V_out)

    def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
        Q, K, V = pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, out_shapes=out_shapes, out_perms=out_perms)
        Q, K, V = per_head_norm_and_rope(Q, K, V, cos, sin, out_shapes=out_shapes, out_perms=out_perms)
        return (Q, K, V)

    def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
        Q_ret = RetileStreamify(graph, Q, split_row=True, chunk=1)
        Q_outer = PromoteOuter(graph, Q_ret)
        Q_buf = Bufferize(graph, Q_outer, rank=1)
        Q_str = Streamify(graph, Q_buf, stride=tuple((4, 1, 16)), out_shape_tiled=tuple((4, 4, 64)))
        Q_abs = Accum(graph, Q_str, output_stream_dtype=_dsl2step_out_tile(Q_str, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(Q_str, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Qh = Flatten(graph, Q_abs, min_rank=1, max_rank=2)
        K_ret = RetileStreamify(graph, K, split_row=True, chunk=1)
        K_outer = PromoteOuter(graph, K_ret)
        K_buf = Bufferize(graph, K_outer, rank=1)
        K_str = Streamify(graph, K_buf, stride=tuple((1, 4)), out_shape_tiled=tuple((4, 64)))
        K_abs = Accum(graph, K_str, output_stream_dtype=_dsl2step_out_tile(K_str, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(K_str, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        K_flat = Flatten(graph, K_abs, min_rank=0, max_rank=1)
        Kh = Promote(graph, K_flat, promote_rank=0)
        V_ret = RetileStreamify(graph, V, split_row=True, chunk=1)
        V_outer = PromoteOuter(graph, V_ret)
        V_buf = Bufferize(graph, V_outer, rank=1)
        V_str = Streamify(graph, V_buf, stride=tuple((1, 4)), out_shape_tiled=tuple((4, 64)))
        V_abs = Accum(graph, V_str, output_stream_dtype=_dsl2step_out_tile(V_str, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(V_str, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        V_flat = Flatten(graph, V_abs, min_rank=0, max_rank=1)
        Vh = Promote(graph, V_flat, promote_rank=0)
        return (Qh, Kh, Vh)

    def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
        Kh_exp = ExpandRef(graph, Kh, ref=Qh, expand_rank=1)
        Vh_exp = ExpandRef(graph, Vh, ref=Qh, expand_rank=1)
        scores = BinaryMap(graph, Qh, Kh_exp, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=4096)
        col_stream = RetileStreamify(graph, scores, split_row=False, chunk=1)
        col_split = Reshape(graph, col_stream, chunk_size=64, reshape_rank=0, write_back_mu=False)
        row_max = Accum(graph, col_split, output_stream_dtype=_dsl2step_out_tile(col_split, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(col_split, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        _tmp2 = UnaryMap(graph, row_max, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        centered = BinaryMap(graph, scores, _tmp2, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        e = UnaryMap(graph, centered, fn=map_fn.Exp(), write_back_mu=False, compute_bw=4096)
        num = BinaryMap(graph, e, Vh_exp, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        denom = UnaryMap(graph, e, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        attn = BinaryMap(graph, num, denom, fn=map_fn.Div(), write_back_mu=False, compute_bw=4096)
        attn_flat = Flatten(graph, attn, min_rank=0, max_rank=1)
        seq_stream = RetileStreamify(graph, attn_flat, split_row=True, chunk=1)
        _parallelize6 = Parallelize(graph, seq_stream, parallelize_rank=seq_stream.stream.rank, num_consumers=64)
        parts = [_BranchRef(_parallelize6, _i) for _i in range(64)]
        processed = []
        for p in parts:
            p = PromoteOuter(graph, p)
            p = Accum(graph, p, output_stream_dtype=_dsl2step_out_tile(p, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(p, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            processed.append(p)
        out = StaticReassemble(graph, inputs=processed, merge_rank=_dsl2step_stream(processed[0]).rank)
        return out

    def attention(Q, K, V, *, out_shapes, out_perms=None):
        Qh, Kh, Vh = compute_qkv(Q, K, V, out_shapes=((4, 4, 64, 32), (4, 1, 64, 32), (4, 1, 64, 32)))
        attn = attention_compute(Qh, Kh, Vh, out_shapes=out_shapes, out_perms=out_perms)
        return attn

    def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
        attn = attention(Q, K, V, out_shapes=((64, 16, 32),), out_perms=(None,))
        a = RetileStreamify(graph, attn, split_row=True, chunk=1)
        a = RetileStreamify(graph, a, split_row=False, chunk=1)
        a = Reshape(graph, a, chunk_size=512, reshape_rank=0, write_back_mu=False)
        attn_flat = Accum(graph, a, output_stream_dtype=_dsl2step_out_tile(a, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(a, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        w = _offchip_load_or_restream(graph, o_proj_weight, stride=tuple((0,)), out_shape_tiled=tuple((64,)), tile_row=512, tile_col=512, par_dispatch=8, transposed=True)
        w = Flatten(graph, w, min_rank=0, max_rank=1)
        inp = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((64,)), tile_row=1, tile_col=512, par_dispatch=8, transposed=True)
        inp = Flatten(graph, inp, min_rank=0, max_rank=1)
        proj = BinaryMap(graph, w, attn_flat, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        out = BinaryMap(graph, proj, inp, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return out

    def rms_norm(res_add_0, *, out_shapes, out_perms=None):
        sq = UnaryMap(graph, res_add_0, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sq_stream = RetileStreamify(graph, sq, split_row=True, chunk=1)
        hidden = res_add_0.stream_dtype.shape[0]
        sq_grouped = Reshape(graph, sq_stream, chunk_size=hidden, reshape_rank=0, write_back_mu=False)
        sum_sq = Accum(graph, sq_grouped, output_stream_dtype=_dsl2step_out_tile(sq_grouped, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(sq_grouped, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / hidden), write_back_mu=False, compute_bw=4096)
        eps_added = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        rsqrt = UnaryMap(graph, eps_added, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        out = BinaryMap(graph, res_add_0, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        return out

    def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        tokens = RepeatStatic(graph, normed_2, repeat_factor=2)
        tokens = PromoteOuter(graph, tokens)
        selector = SelectGen(is_multihot=True, tensor=expert_onehot, n=8)
        graph.add_node(selector)
        _flat_partition7 = FlatPartition(graph, tokens, control=selector, partition_rank=0, switch_cycles=[1] * 8, write_back_mu=False, num_consumers=8)
        token_parts = [_BranchRef(_flat_partition7, _i) for _i in range(8)]
        expw = _offchip_load_or_restream(graph, expert_weights, stride=tuple((2, 1)), out_shape_tiled=tuple((64, 2)), tile_row=1, tile_col=1, par_dispatch=8)
        _flat_partition8 = FlatPartition(graph, expw, control=selector, partition_rank=0, switch_cycles=[1] * 8, write_back_mu=False, num_consumers=8)
        weight_parts = [_BranchRef(_flat_partition8, _i) for _i in range(8)]
        per_expert_outputs = []
        num_experts = w_gate.shape[0]
        for e in range(num_experts):
            wg_ref = LinearOffChipLoadRef(graph, ref=token_parts[e], underlying=w_gate[e], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=512, tile_col=1792, par_dispatch=8, transposed=True)
            wg = Flatten(graph, wg_ref, min_rank=0, max_rank=1)
            wu_ref = LinearOffChipLoadRef(graph, ref=token_parts[e], underlying=w_up[e], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=512, tile_col=1792, par_dispatch=8, transposed=True)
            wu = Flatten(graph, wu_ref, min_rank=0, max_rank=1)
            wd_ref = LinearOffChipLoadRef(graph, ref=token_parts[e], underlying=w_down[e], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=1792, tile_col=512, par_dispatch=8, transposed=True)
            wd = Flatten(graph, wd_ref, min_rank=0, max_rank=1)
            gate = BinaryMap(graph, wg, token_parts[e], fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            up = BinaryMap(graph, wu, token_parts[e], fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            _tmp3 = UnaryMap(graph, gate, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
            hidden = BinaryMap(graph, _tmp3, up, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            down = BinaryMap(graph, wd, hidden, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            scaled = BinaryMap(graph, down, weight_parts[e], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            per_expert_outputs.append(scaled)
        merged = FlatReassemble(graph, inputs=per_expert_outputs, control=selector, reassemble_rank=0, switch_cycles=[1] * len(per_expert_outputs), write_back_mu=False)
        summed = Accum(graph, merged, output_stream_dtype=_dsl2step_out_tile(merged, 'elem', 2), fn=accum_fn.Add(), init_fn=_dsl2step_init(merged, 'elem'), accum_rank=2, write_back_mu=False, compute_bw=4096)
        final = Flatten(graph, summed, min_rank=0, max_rank=1)
        return final

    def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        normed_2 = rms_norm(res_add_0, out_shapes=(out_shapes[0],), out_perms=(None,))
        return moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes, out_perms=out_perms)

    def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        child_out_shapes = out_shapes
        child_out_perms = out_perms if out_perms is not None else (None,)
        moe_out = moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=child_out_shapes, out_perms=child_out_perms)
        _tmp4 = BinaryMap(graph, moe_out, res_add_0, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return _tmp4
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
    Q, K, V = pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, out_shapes=((64, 16, 32), (64, 4, 32), (64, 4, 32)))
    res_add_0 = attention_o_proj(Q, K, V, o_proj_weight, input_tensor, out_shapes=((64, 512, 1),))
    out = moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=((64, 512, 1),))
    # Reshape stream/tile so the Rust accum's row-major reading matches gold's
    # (token, embed) layout. The moe output is stream (64,) tile (512, 1) — a
    # column-tile per token. Rust OffChipStore stores accum=(tile_row, last*tile_col)
    # = (512, 64) = (embed, token), but gold is (token, embed). Inserting a
    # Promote(rank=0) turns stream (64,) → (64, 1); then OffChipStore emits one
    # ValStop per token (tile_row=512 rows each), accumulating vertically →
    # accum (32768, 1) flat in (token, embed) row-major order.
    out = Promote(graph, out, promote_rank=0)
    out = PromoteOuter(graph, out)
    _store9 = OffChipStore(graph, out, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store9)