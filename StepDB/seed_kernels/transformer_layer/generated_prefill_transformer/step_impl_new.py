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

def build_graph(dims, tensors):
    graph = Graph()

    def per_head_norm_and_rope_0(Q, K, V, cos, sin, *, out_shapes):

        def _rms_norm(x, eps):
            sq = UnaryMap(graph, x, fn=map_fn.Square(), write_back_mu=False, compute_bw=1)
            sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
            head_dim = int(x.stream_dtype.shape[1])
            mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / head_dim), write_back_mu=False, compute_bw=1)
            mean_eps = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=1)
            rsqrt = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=1)
            _tmp1 = BinaryMap(graph, x, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
            return _tmp1
        seq_len = int(Q.shape[0])
        head_dim = int(Q.shape[-1])
        half_dim = head_dim // 2
        cos_stream = LinearOffChipLoad(cos, stride=tuple((1, 0)), out_shape_tiled=tuple((seq_len, 1)), tile_row=1, tile_col=head_dim, par_dispatch=1, start_tile_idx=0)
        sin_stream = LinearOffChipLoad(sin, stride=tuple((1, 0)), out_shape_tiled=tuple((seq_len, 1)), tile_row=1, tile_col=head_dim, par_dispatch=1, start_tile_idx=0)
        Q_outer = PromoteOuter(graph, Q)
        K_outer = PromoteOuter(graph, K)
        V_outer = PromoteOuter(graph, V)
        cos_exp_q = ExpandRef(graph, cos_stream, ref=Q_outer, expand_rank=1)
        sin_exp_q = ExpandRef(graph, sin_stream, ref=Q_outer, expand_rank=1)
        cos_exp_k = ExpandRef(graph, cos_stream, ref=K_outer, expand_rank=1)
        sin_exp_k = ExpandRef(graph, sin_stream, ref=K_outer, expand_rank=1)
        eps = 1e-06
        Q_norm = _rms_norm(Q_outer, eps)
        K_norm = _rms_norm(K_outer, eps)

        def _apply_rope(x_norm, cos_exp, sin_exp):
            x_split = RetileStreamify(graph, x_norm, split_row=False, chunk=half_dim)
            cos_split = RetileStreamify(graph, cos_exp, split_row=False, chunk=half_dim)
            sin_split = RetileStreamify(graph, sin_exp, split_row=False, chunk=half_dim)
            x_flat = Flatten(graph, x_split, min_rank=0, max_rank=2)
            cos_flat = Flatten(graph, cos_split, min_rank=0, max_rank=2)
            sin_flat = Flatten(graph, sin_split, min_rank=0, max_rank=2)
            _parallelize9 = Parallelize(graph, x_flat, parallelize_rank=x_flat.stream.rank, num_consumers=2)
            x_even = _BranchRef(_parallelize9, 0)
            x_odd = _BranchRef(_parallelize9, 1)
            _parallelize10 = Parallelize(graph, cos_flat, parallelize_rank=cos_flat.stream.rank, num_consumers=2)
            cos_even = _BranchRef(_parallelize10, 0)
            cos_odd = _BranchRef(_parallelize10, 1)
            _parallelize11 = Parallelize(graph, sin_flat, parallelize_rank=sin_flat.stream.rank, num_consumers=2)
            sin_even = _BranchRef(_parallelize11, 0)
            sin_odd = _BranchRef(_parallelize11, 1)
            term_a = BinaryMap(graph, x_even, cos_even, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
            term_b = BinaryMap(graph, x_odd, sin_even, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
            term_b_neg = UnaryMap(graph, term_b, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=1)
            first_half = BinaryMap(graph, term_a, term_b_neg, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
            term_c = BinaryMap(graph, x_odd, cos_odd, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
            term_d = BinaryMap(graph, x_even, sin_odd, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
            second_half = BinaryMap(graph, term_c, term_d, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
            merged = StaticReassemble(graph, inputs=[first_half, second_half], merge_rank=_dsl2step_stream([first_half, second_half][0]).rank)
            merged2 = Reshape(graph, merged, chunk_size=2, reshape_rank=0, write_back_mu=False)
            heads = int(x_norm.shape[2])
            merged3 = Reshape(graph, merged2, chunk_size=heads, reshape_rank=1, write_back_mu=False)
            result = Accum(graph, merged3, output_stream_dtype=_dsl2step_out_tile(merged3, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(merged3, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
            return result
        Q_rot = _apply_rope(Q_norm, cos_exp_q, sin_exp_q)
        K_rot = _apply_rope(K_norm, cos_exp_k, sin_exp_k)
        V_out = Flatten(graph, V_outer, min_rank=1, max_rank=2)
        return (Q_rot, K_rot, V_out)

    def pre_attention_0(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes):
        seq_len = input_tensor.shape[0]
        hidden_dim = input_tensor.shape[1]
        head_dim = cos.shape[-1]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim
        par_factor = 4
        assert seq_len % par_factor == 0, f'seq_len={seq_len} not divisible by par_factor={par_factor}'
        tokens_per_branch = seq_len // par_factor

        def load_and_flatten_proj(branch_ref, proj, out_dim):
            proj_loaded = LinearOffChipLoadRef(graph, ref=branch_ref, underlying=proj, stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=hidden_dim, tile_col=out_dim, par_dispatch=1, start_tile_idx=0)
            _tmp2 = Flatten(graph, proj_loaded, min_rank=0, max_rank=1)
            return _tmp2
        eps = 1e-06
        Q_parts = []
        K_parts = []
        V_parts = []
        for i in range(par_factor):
            inp_loaded = LinearOffChipLoad(input_tensor, stride=tuple((par_factor,)), out_shape_tiled=tuple((tokens_per_branch,)), tile_row=1, tile_col=hidden_dim, par_dispatch=1, start_tile_idx=i)
            inp = Flatten(graph, inp_loaded, min_rank=0, max_rank=1)
            sq = UnaryMap(graph, inp, fn=map_fn.Square(), write_back_mu=False, compute_bw=1)
            sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
            mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / hidden_dim), write_back_mu=False, compute_bw=1)
            mean_e = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=1)
            inv_sqrt = UnaryMap(graph, mean_e, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=1)
            normed = BinaryMap(graph, inp, inv_sqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=32)
            q_proj_flat = load_and_flatten_proj(normed, q_proj, q_proj.shape[1])
            q_mat = BinaryMap(graph, normed, q_proj_flat, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=32)
            q_retiled = RetileStreamify(graph, q_mat, split_row=False, chunk=head_dim)
            _tmp3 = Reshape(graph, q_retiled, chunk_size=num_heads, reshape_rank=0, write_back_mu=False)
            Q_parts.append(_tmp3)
            k_proj_flat = load_and_flatten_proj(normed, k_proj, k_proj.shape[1])
            k_mat = BinaryMap(graph, normed, k_proj_flat, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=32)
            k_retiled = RetileStreamify(graph, k_mat, split_row=False, chunk=head_dim)
            _tmp4 = Reshape(graph, k_retiled, chunk_size=num_kv_heads, reshape_rank=0, write_back_mu=False)
            K_parts.append(_tmp4)
            v_proj_flat = load_and_flatten_proj(normed, v_proj, v_proj.shape[1])
            v_mat = BinaryMap(graph, normed, v_proj_flat, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=32)
            v_retiled = RetileStreamify(graph, v_mat, split_row=False, chunk=head_dim)
            _tmp5 = Reshape(graph, v_retiled, chunk_size=num_kv_heads, reshape_rank=0, write_back_mu=False)
            V_parts.append(_tmp5)
        Q = StaticReassemble(graph, inputs=Q_parts, merge_rank=_dsl2step_stream(Q_parts[0]).rank)
        K = StaticReassemble(graph, inputs=K_parts, merge_rank=_dsl2step_stream(K_parts[0]).rank)
        V = StaticReassemble(graph, inputs=V_parts, merge_rank=_dsl2step_stream(V_parts[0]).rank)
        Q, K, V = per_head_norm_and_rope_0(Q, K, V, cos, sin, out_shapes=out_shapes)
        return (Q, K, V)

    def compute_qkv(Q, K, V, *, out_shapes):
        q = PromoteOuter(graph, Q)
        q = Bufferize(graph, q, rank=2)
        q = Streamify(graph, q, stride=tuple([4, 1, 16]), out_shape_tiled=tuple((4, 4, 64)))
        q = Flatten(graph, q, min_rank=2, max_rank=3)
        k = PromoteOuter(graph, K)
        k = Bufferize(graph, k, rank=2)
        k = Streamify(graph, k, stride=tuple([1, 0, 4]), out_shape_tiled=tuple((4, 1, 64)))
        k = Flatten(graph, k, min_rank=2, max_rank=3)
        v = PromoteOuter(graph, V)
        v = Bufferize(graph, v, rank=2)
        v = Streamify(graph, v, stride=tuple([1, 0, 4]), out_shape_tiled=tuple((4, 1, 64)))
        v = Flatten(graph, v, min_rank=2, max_rank=3)
        return (q, k, v)

    def attention_1(Q, K, V, *, out_shapes):
        """
    Variant with:
      • Parallelism factor 4 (each consumer handles 4 heads).
      • Aggressive compute bandwidth (compute_bw=8) on all heavy ops.
    The design differs from all previously accepted variants (different
    parallel factor and higher compute bandwidth) while staying well under
    the 10\u202fMiB on‑chip budget.
    """
        Qh, Kh, Vh = compute_qkv(Q, K, V, out_shapes=((4, 4, 64, 1, 32), (4, 1, 64, 1, 32), (4, 1, 64, 1, 32)))
        kv = Qh.shape[0]
        heads_per_kv = Qh.shape[1]
        seq_len = Qh.shape[2]
        total_heads = kv * heads_per_kv
        Q_rep = RepeatStatic(graph, Qh, repeat_factor=seq_len)
        Q_flat = Flatten(graph, Q_rep, min_rank=2, max_rank=3)

        def _broadcast_and_merge(tensor):
            outer = PromoteOuter(graph, tensor)
            buf = Bufferize(graph, outer, rank=3)
            streamed = Streamify(graph, buf, stride=tuple((seq_len, 0, 0, 1)), out_shape_tiled=tuple((kv, heads_per_kv, seq_len, seq_len)))
            merged = Flatten(graph, streamed, min_rank=2, max_rank=4)
            return merged
        K_flat = _broadcast_and_merge(Kh)
        V_flat = _broadcast_and_merge(Vh)
        par_factor = 4
        assert total_heads % par_factor == 0, f'total_heads={total_heads} must be divisible by par_factor={par_factor}'
        _parallelize12 = Parallelize(graph, Q_flat, parallelize_rank=Q_flat.stream.rank, num_consumers=par_factor)
        Q_par = [_BranchRef(_parallelize12, _i) for _i in range(par_factor)]
        _parallelize13 = Parallelize(graph, K_flat, parallelize_rank=K_flat.stream.rank, num_consumers=par_factor)
        K_par = [_BranchRef(_parallelize13, _i) for _i in range(par_factor)]
        _parallelize14 = Parallelize(graph, V_flat, parallelize_rank=V_flat.stream.rank, num_consumers=par_factor)
        V_par = [_BranchRef(_parallelize14, _i) for _i in range(par_factor)]
        partials = []
        for i in range(par_factor):
            scores = BinaryMap(graph, Q_par[i], K_par[i], fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=8)
            row_max = Accum(graph, scores, output_stream_dtype=_dsl2step_out_tile(scores, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(scores, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=8)
            row_max = Promote(graph, row_max, promote_rank=0)
            row_max = ExpandRef(graph, row_max, ref=scores, expand_rank=1)
            _tmp6 = UnaryMap(graph, row_max, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=8)
            diff = BinaryMap(graph, scores, _tmp6, fn=map_fn.Add(), write_back_mu=False, compute_bw=8)
            e = UnaryMap(graph, diff, fn=map_fn.Exp(), write_back_mu=False, compute_bw=8)
            num = BinaryMapAccum(graph, e, V_par[i], fn=map_accum_fn.Matmul(weight_transposed=False), init_fn=Zero(shape=(_dsl2step_in_tile(e).shape[0], _dsl2step_in_tile(V_par[i]).shape[0 if False else 1]), dtype=_dsl2step_in_tile(e).tile_dtype), rank=1, write_back_mu=False, compute_bw=8)
            denom = Accum(graph, e, output_stream_dtype=_dsl2step_out_tile(e, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(e, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=8)
            attn = BinaryMap(graph, num, denom, fn=map_fn.Div(), write_back_mu=False, compute_bw=8)
            partials.append(attn)
        attn_merged = StaticReassemble(graph, inputs=partials, merge_rank=_dsl2step_stream(partials[0]).rank)
        attn_with_outer = PromoteOuter(graph, attn_merged)
        attn_buf = Bufferize(graph, attn_with_outer, rank=2)
        stride_out = (1, seq_len)
        out_shape = (seq_len, total_heads)
        attn_stream = Streamify(graph, attn_buf, stride=tuple(stride_out), out_shape_tiled=tuple(out_shape))
        attn_final = Flatten(graph, attn_stream, min_rank=1, max_rank=2)
        return attn_final

    def attention_o_proj_1(Q, K, V, o_proj_weight, input_tensor, *, out_shapes):
        attn = attention_1(Q, K, V, out_shapes=((64, 16, 1, 32),))
        attn_flat = Accum(graph, attn, output_stream_dtype=_dsl2step_out_tile(attn, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
        seq_par = 4
        assert 64 % seq_par == 0, 'seq_len must be divisible by seq_par'
        _parallelize15 = Parallelize(graph, attn_flat, parallelize_rank=attn_flat.stream.rank, num_consumers=seq_par)
        attn_parts = [_BranchRef(_parallelize15, _i) for _i in range(seq_par)]
        w_ref = LinearOffChipLoadRef(graph, ref=attn_flat, underlying=o_proj_weight, stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=512, tile_col=512, par_dispatch=1, start_tile_idx=0)
        _parallelize16 = Parallelize(graph, w_ref, parallelize_rank=w_ref.stream.rank, num_consumers=seq_par)
        w_parts = [_BranchRef(_parallelize16, _i) for _i in range(seq_par)]
        inp_raw = LinearOffChipLoad(input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((64,)), tile_row=1, tile_col=512, par_dispatch=1, start_tile_idx=0)
        inp_flat = Flatten(graph, inp_raw, min_rank=0, max_rank=1)
        _parallelize17 = Parallelize(graph, inp_flat, parallelize_rank=inp_flat.stream.rank, num_consumers=seq_par)
        inp_parts = [_BranchRef(_parallelize17, _i) for _i in range(seq_par)]
        outputs = []
        for i in range(seq_par):
            a = RepeatStatic(graph, attn_parts[i], repeat_factor=1)
            p = BinaryMap(graph, a, w_parts[i], fn=map_fn.Matmul(weight_transposed=False), write_back_mu=False, compute_bw=1)
            b = RepeatStatic(graph, inp_parts[i], repeat_factor=1)
            out_i = BinaryMap(graph, p, b, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
            outputs.append(out_i)
        out = StaticReassemble(graph, inputs=outputs, merge_rank=_dsl2step_stream(outputs[0]).rank)
        return out

    def moe_1(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes):
        hidden_dim = res_add_0.shape[-1]
        eps = 1e-06
        par_factor = 16
        assert res_add_0.shape[0] % par_factor == 0, f'seq_len {res_add_0.shape[0]} not divisible by par_factor {par_factor}'
        _parallelize18 = Parallelize(graph, res_add_0, parallelize_rank=res_add_0.stream.rank, num_consumers=par_factor)
        streams = [_BranchRef(_parallelize18, _i) for _i in range(par_factor)]
        rms_parts = []
        for i in range(par_factor):
            x = streams[i]
            sq = UnaryMap(graph, x, fn=map_fn.Square(), write_back_mu=False, compute_bw=64)
            sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=64)
            mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / hidden_dim), write_back_mu=False, compute_bw=64)
            mean_eps = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=64)
            rsqrt = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=64)
            out_i = BinaryMap(graph, x, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=64)
            rms_parts.append(out_i)
        normed_2 = StaticReassemble(graph, inputs=rms_parts, merge_rank=_dsl2step_stream(rms_parts[0]).rank)
        token_dup = RepeatStatic(graph, normed_2, repeat_factor=2)
        token_flat = Flatten(graph, token_dup, min_rank=0, max_rank=1)
        token_ref = PromoteOuter(graph, token_flat)
        seq_len = normed_2.shape[0]
        weight_raw = LinearOffChipLoad(expert_weights, stride=tuple((2, 1)), out_shape_tiled=tuple((seq_len, 2)), tile_row=1, tile_col=1, par_dispatch=1, start_tile_idx=0)
        selector = SelectGen(is_multihot=True, tensor=expert_onehot, n=8)
        graph.add_node(selector)
        _flat_partition19 = FlatPartition(graph, token_ref, control=selector, partition_rank=0, switch_cycles=[1] * 8, write_back_mu=False, num_consumers=8)
        token_parts = [_BranchRef(_flat_partition19, _i) for _i in range(8)]
        _flat_partition20 = FlatPartition(graph, weight_raw, control=selector, partition_rank=0, switch_cycles=[1] * 8, write_back_mu=False, num_consumers=8)
        weight_parts = [_BranchRef(_flat_partition20, _i) for _i in range(8)]
        inter_dim = w_gate.shape[2]
        chunk_size = 16
        assert inter_dim % chunk_size == 0, 'moe_inter_dim must be divisible by chunk_size'
        num_chunks = inter_dim // chunk_size
        contribs = []
        for e_idx in range(8):
            base_tok = token_parts[e_idx]
            base_wgt = weight_parts[e_idx]
            tok_chunks = RepeatStatic(graph, base_tok, repeat_factor=num_chunks)
            w_gate_chunks = LinearOffChipLoadRef(graph, ref=base_tok, underlying=w_gate[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((num_chunks,)), tile_row=hidden_dim, tile_col=chunk_size, par_dispatch=1, start_tile_idx=0)
            w_up_chunks = LinearOffChipLoadRef(graph, ref=base_tok, underlying=w_up[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((num_chunks,)), tile_row=hidden_dim, tile_col=chunk_size, par_dispatch=1, start_tile_idx=0)
            w_down_chunks = LinearOffChipLoadRef(graph, ref=base_tok, underlying=w_down[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((num_chunks,)), tile_row=chunk_size, tile_col=hidden_dim, par_dispatch=1, start_tile_idx=0)
            gate_chunk = BinaryMap(graph, tok_chunks, w_gate_chunks, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=64)
            up_chunk = BinaryMap(graph, tok_chunks, w_up_chunks, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=64)
            _tmp7 = UnaryMap(graph, gate_chunk, fn=map_fn.Silu(), write_back_mu=False, compute_bw=64)
            hidden_chunk = BinaryMap(graph, _tmp7, up_chunk, fn=map_fn.Mul(), write_back_mu=False, compute_bw=64)
            down_chunk = BinaryMap(graph, hidden_chunk, w_down_chunks, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=64)
            down_out = Accum(graph, down_chunk, output_stream_dtype=_dsl2step_out_tile(down_chunk, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(down_chunk, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
            contrib = BinaryMap(graph, down_out, base_wgt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=64)
            contribs.append(contrib)
        agg = FlatReassemble(graph, inputs=contribs, control=selector, reassemble_rank=0, switch_cycles=[1] * len(contribs), write_back_mu=False)
        merged = Flatten(graph, agg, min_rank=0, max_rank=1)
        summed = Accum(graph, merged, output_stream_dtype=_dsl2step_out_tile(merged, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(merged, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
        collapsed = Flatten(graph, summed, min_rank=0, max_rank=1)
        final = RepeatStatic(graph, collapsed, repeat_factor=1)
        _tmp8 = BinaryMap(graph, final, res_add_0, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        return _tmp8
    Q, K, V = pre_attention_0(tensors['input_tensor'], tensors['q_proj'], tensors['k_proj'], tensors['v_proj'], tensors['cos'], tensors['sin'], out_shapes=((64, 16, 1, 32), (64, 4, 1, 32), (64, 4, 1, 32)))
    attn_res = attention_o_proj_1(Q, K, V, tensors['o_proj_weight'], tensors['input_tensor'], out_shapes=((64, 16, 1, 32),))
    moe_out = moe_1(attn_res, tensors['w_gate'], tensors['w_up'], tensors['w_down'], tensors['expert_weights'], tensors['expert_onehot'], out_shapes=((64, 1, 1, 512),))
    _store21 = OffChipStore(graph, moe_out, par_dispatch=1)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store21)