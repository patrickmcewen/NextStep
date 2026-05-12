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
        head_dim = cos.shape[2]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim
        x = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=hidden_dim, par_dispatch=8)
        sq = UnaryMap(graph, x, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / hidden_dim), write_back_mu=False, compute_bw=4096)
        eps_added = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        rsqrt = UnaryMap(graph, eps_added, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        normed = BinaryMap(graph, x, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        w_q = _offchip_load_or_restream(graph, q_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=num_heads * head_dim, par_dispatch=8)
        w_k = _offchip_load_or_restream(graph, k_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=num_kv_heads * head_dim, par_dispatch=8)
        w_v = _offchip_load_or_restream(graph, v_proj, stride=tuple((0,)), out_shape_tiled=tuple((seq_len,)), tile_row=hidden_dim, tile_col=num_kv_heads * head_dim, par_dispatch=8)

        def _project_from_loaded(weight, n_heads):
            y = BinaryMap(graph, normed, weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            y = RetileStreamify(graph, y, split_row=False, chunk=head_dim)
            y = Reshape(graph, y, chunk_size=n_heads, reshape_rank=0, write_back_mu=False)
            y = Flatten(graph, y, min_rank=1, max_rank=2)
            y = Accum(graph, y, output_stream_dtype=_dsl2step_out_tile(y, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(y, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            return y
        Q = _project_from_loaded(w_q, num_heads)
        K = _project_from_loaded(w_k, num_kv_heads)
        V = _project_from_loaded(w_v, num_kv_heads)
        return (Q, K, V)

    def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
        seq_len = Q.shape[0]
        head_dim = Q.shape[-1]
        half = head_dim // 2
        eps = 1e-06
        cos_loaded = _offchip_load_or_restream(graph, cos, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=head_dim, par_dispatch=8)
        sin_loaded = _offchip_load_or_restream(graph, sin, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=head_dim, par_dispatch=8)
        cos_loaded = Flatten(graph, cos_loaded, min_rank=0, max_rank=1)
        sin_loaded = Flatten(graph, sin_loaded, min_rank=0, max_rank=1)
        Q_sq = UnaryMap(graph, Q, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        Q_sq_sum = UnaryMap(graph, Q_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        Q_mean = UnaryMap(graph, Q_sq_sum, fn=map_fn.MulImmediate(1.0 / head_dim), write_back_mu=False, compute_bw=4096)
        Q_mean_eps = UnaryMap(graph, Q_mean, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        Q_scale = UnaryMap(graph, Q_mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        Q_norm = BinaryMap(graph, Q, Q_scale, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K_sq = UnaryMap(graph, K, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        K_sq_sum = UnaryMap(graph, K_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        K_mean = UnaryMap(graph, K_sq_sum, fn=map_fn.MulImmediate(1.0 / head_dim), write_back_mu=False, compute_bw=4096)
        K_mean_eps = UnaryMap(graph, K_mean, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        K_scale = UnaryMap(graph, K_mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        K_norm = BinaryMap(graph, K, K_scale, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q_split = RetileStreamify(graph, Q_norm, split_row=False, chunk=half)
        K_split = RetileStreamify(graph, K_norm, split_row=False, chunk=half)
        cos_split = RetileStreamify(graph, cos_loaded, split_row=False, chunk=half)
        sin_split = RetileStreamify(graph, sin_loaded, split_row=False, chunk=half)
        _parallelize3 = Parallelize(graph, Q_split, parallelize_rank=Q_split.stream.rank, num_consumers=2)
        Q0 = _BranchRef(_parallelize3, 0)
        Q1 = _BranchRef(_parallelize3, 1)
        _parallelize4 = Parallelize(graph, K_split, parallelize_rank=K_split.stream.rank, num_consumers=2)
        K0 = _BranchRef(_parallelize4, 0)
        K1 = _BranchRef(_parallelize4, 1)
        _parallelize5 = Parallelize(graph, cos_split, parallelize_rank=cos_split.stream.rank, num_consumers=2)
        cos0 = _BranchRef(_parallelize5, 0)
        cos1 = _BranchRef(_parallelize5, 1)
        _parallelize6 = Parallelize(graph, sin_split, parallelize_rank=sin_split.stream.rank, num_consumers=2)
        sin0 = _BranchRef(_parallelize6, 0)
        sin1 = _BranchRef(_parallelize6, 1)
        q0_cos = BinaryMap(graph, Q0, cos0, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        q1_sin = BinaryMap(graph, Q1, sin0, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        q1_sin_neg = UnaryMap(graph, q1_sin, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        Q_out0 = BinaryMap(graph, q0_cos, q1_sin_neg, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        q1_cos = BinaryMap(graph, Q1, cos1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        q0_sin = BinaryMap(graph, Q0, sin1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q_out1 = BinaryMap(graph, q1_cos, q0_sin, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        k0_cos = BinaryMap(graph, K0, cos0, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        k1_sin = BinaryMap(graph, K1, sin0, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        k1_sin_neg = UnaryMap(graph, k1_sin, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        K_out0 = BinaryMap(graph, k0_cos, k1_sin_neg, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        k1_cos = BinaryMap(graph, K1, cos1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        k0_sin = BinaryMap(graph, K0, sin1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K_out1 = BinaryMap(graph, k1_cos, k0_sin, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        Q_comb = StaticReassemble(graph, inputs=[Q_out0, Q_out1], merge_rank=_dsl2step_stream([Q_out0, Q_out1][0]).rank)
        Q_comb = Reshape(graph, Q_comb, chunk_size=2, reshape_rank=0, write_back_mu=False)
        Q_final = Accum(graph, Q_comb, output_stream_dtype=_dsl2step_out_tile(Q_comb, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(Q_comb, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        K_comb = StaticReassemble(graph, inputs=[K_out0, K_out1], merge_rank=_dsl2step_stream([K_out0, K_out1][0]).rank)
        K_comb = Reshape(graph, K_comb, chunk_size=2, reshape_rank=0, write_back_mu=False)
        K_final = Accum(graph, K_comb, output_stream_dtype=_dsl2step_out_tile(K_comb, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(K_comb, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        return (Q_final, K_final, V)

    def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
        Q0, K0, V0 = pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, out_shapes=out_shapes, out_perms=out_perms)
        Q, K, V = per_head_norm_and_rope(Q0, K0, V0, cos, sin, out_shapes=out_shapes, out_perms=out_perms)
        return (Q, K, V)

    def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
        Qp = Promote(graph, Q, promote_rank=0)
        Qr = RetileStreamify(graph, Qp, split_row=True, chunk=1)
        Qr = PromoteOuter(graph, Qr)
        Qb = Bufferize(graph, Qr, rank=2)
        Qs = Streamify(graph, Qb, stride=tuple((4, 1, 16)), out_shape_tiled=tuple((4, 4, 64)))
        Qh = Accum(graph, Qs, output_stream_dtype=_dsl2step_out_tile(Qs, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(Qs, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Kp = Promote(graph, K, promote_rank=0)
        Kr = RetileStreamify(graph, Kp, split_row=True, chunk=1)
        Kr = PromoteOuter(graph, Kr)
        Kb = Bufferize(graph, Kr, rank=2)
        Ks = Streamify(graph, Kb, stride=tuple((1, 4)), out_shape_tiled=tuple((4, 64)))
        Kh_tmp = Accum(graph, Ks, output_stream_dtype=_dsl2step_out_tile(Ks, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(Ks, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Kh = Promote(graph, Kh_tmp, promote_rank=0)
        Vp = Promote(graph, V, promote_rank=0)
        Vr = RetileStreamify(graph, Vp, split_row=True, chunk=1)
        Vr = PromoteOuter(graph, Vr)
        Vb = Bufferize(graph, Vr, rank=2)
        Vs = Streamify(graph, Vb, stride=tuple((1, 4)), out_shape_tiled=tuple((4, 64)))
        Vh_tmp = Accum(graph, Vs, output_stream_dtype=_dsl2step_out_tile(Vs, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(Vs, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Vh = Promote(graph, Vh_tmp, promote_rank=0)
        return (Qh, Kh, Vh)

    def attention_compute(Qh, Kh, Vh, *, out_shapes, out_perms=None):
        Kh_exp = ExpandRef(graph, Kh, ref=Qh, expand_rank=1)
        scores = BinaryMap(graph, Qh, Kh_exp, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=4096)
        scores_p = Promote(graph, scores, promote_rank=0)
        scores_rs = RetileStreamify(graph, scores_p, split_row=False, chunk=1)
        row_max = Accum(graph, scores_rs, output_stream_dtype=_dsl2step_out_tile(scores_rs, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(scores_rs, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        neg_max = UnaryMap(graph, row_max, fn=map_fn.MulImmediate(-1), write_back_mu=False, compute_bw=4096)
        diff = BinaryMap(graph, scores, neg_max, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        e = UnaryMap(graph, diff, fn=map_fn.Exp(), write_back_mu=False, compute_bw=4096)
        denom = UnaryMap(graph, e, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        Vh_exp = ExpandRef(graph, Vh, ref=e, expand_rank=1)
        num = BinaryMap(graph, e, Vh_exp, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        attn = BinaryMap(graph, num, denom, fn=map_fn.Div(), write_back_mu=False, compute_bw=4096)
        attn_merged = Accum(graph, attn, output_stream_dtype=_dsl2step_out_tile(attn, 'col', 2), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn, 'col'), accum_rank=2, write_back_mu=False, compute_bw=4096)
        attn_stream = Promote(graph, attn_merged, promote_rank=0)
        final = RetileStreamify(graph, attn_stream, split_row=True, chunk=1)
        return final

    def attention(Q, K, V, *, out_shapes, out_perms=None):
        Qh, Kh, Vh = compute_qkv(Q, K, V, out_shapes=((4, 4, 64, 32), (4, 1, 64, 32), (4, 1, 64, 32)), out_perms=(None, None, None))
        attn = attention_compute(Qh, Kh, Vh, out_shapes=out_shapes, out_perms=out_perms)
        return attn

    def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes, out_perms=None):
        attn = attention(Q, K, V, out_shapes=((64, 1, 512),), out_perms=(None,))
        weight = _offchip_load_or_restream(graph, o_proj_weight, stride=tuple((0,)), out_shape_tiled=tuple((64,)), tile_row=512, tile_col=512, par_dispatch=8)
        resid = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((64,)), tile_row=1, tile_col=512, par_dispatch=8)
        proj = BinaryMap(graph, attn, weight, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        summed = BinaryMap(graph, proj, resid, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return summed

    def rms_norm(res_add_0, *, out_shapes, out_perms=None):
        hidden_dim = int(res_add_0.shape[-1])
        eps = 1e-06
        sq = UnaryMap(graph, res_add_0, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        mean = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / hidden_dim), write_back_mu=False, compute_bw=4096)
        mean_eps = UnaryMap(graph, mean, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        scale = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        out = BinaryMap(graph, res_add_0, scale, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        return out

    def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        """
    MoE dispatch & aggregation using only DSL operations.

    The algorithm follows the reference implementation but stays completely
    within the DSL surface:
      1. Split the on‑chip activation tensor into a per‑token stream.
      2. Duplicate each token for the two activated‑expert slots and prepend
         a leading singleton stream dimension.
      3. Build a MultiHot selector from ``expert_onehot``.
      4. Partition the duplicated token stream and the per‑token scalar
         routing weights into per‑expert sub‑streams.
      5. For each routed expert:
           * Broadcast the expert weight matrices to the token sub‑stream
             (using ``offchip_load_ref`` + ``flatten`` to drop the extra
             singleton stream dim).
           * Perform the expert forward‑pass (gate → up → SiLU → down).
           * Multiply by the per‑token routing scalar and collect the result.
      6. Re‑assemble the per‑expert results, sum over the two activated‑expert
         slots, and absorb the token stream into the tile row dimension.
      7. Return the final ``StepTensor`` whose shape is (1, 64, 512), matching
         the declared output shape.
    """
        seq_len = normed_2.shape[-3]
        dim = normed_2.shape[-1]
        inter_dim = w_gate.shape[2]
        n_activated_experts = expert_onehot.shape[1]
        n_routed_experts = w_gate.shape[0]
        token_stream = RetileStreamify(graph, normed_2, split_row=True, chunk=1)
        token_dup = RepeatStatic(graph, token_stream, repeat_factor=n_activated_experts)
        selector = SelectGen(is_multihot=True, tensor=expert_onehot, n=n_routed_experts)
        graph.add_node(selector)
        _flat_partition7 = FlatPartition(graph, token_dup, control=selector, partition_rank=0, switch_cycles=[1] * n_routed_experts, write_back_mu=False, num_consumers=n_routed_experts)
        token_per_expert = [_BranchRef(_flat_partition7, _i) for _i in range(n_routed_experts)]
        weight_stream = _offchip_load_or_restream(graph, expert_weights, stride=tuple((n_activated_experts, 1)), out_shape_tiled=tuple((seq_len, n_activated_experts)), tile_row=1, tile_col=1, par_dispatch=8)
        _flat_partition8 = FlatPartition(graph, weight_stream, control=selector, partition_rank=0, switch_cycles=[1] * n_routed_experts, write_back_mu=False, num_consumers=n_routed_experts)
        weight_per_expert = [_BranchRef(_flat_partition8, _i) for _i in range(n_routed_experts)]
        expert_outputs = []
        for e_idx in range(n_routed_experts):
            wg_raw = LinearOffChipLoadRef(graph, ref=token_per_expert[e_idx], underlying=w_gate[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=dim, tile_col=inter_dim, par_dispatch=8)
            wg = Flatten(graph, wg_raw, min_rank=0, max_rank=1)
            wu_raw = LinearOffChipLoadRef(graph, ref=token_per_expert[e_idx], underlying=w_up[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=dim, tile_col=inter_dim, par_dispatch=8)
            wu = Flatten(graph, wu_raw, min_rank=0, max_rank=1)
            wd_raw = LinearOffChipLoadRef(graph, ref=token_per_expert[e_idx], underlying=w_down[e_idx], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=inter_dim, tile_col=dim, par_dispatch=8)
            wd = Flatten(graph, wd_raw, min_rank=0, max_rank=1)
            gate = BinaryMap(graph, token_per_expert[e_idx], wg, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            up = BinaryMap(graph, token_per_expert[e_idx], wu, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            _tmp1 = UnaryMap(graph, gate, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
            hidden = BinaryMap(graph, _tmp1, up, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            down = BinaryMap(graph, hidden, wd, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            weighted = BinaryMap(graph, down, weight_per_expert[e_idx], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            expert_outputs.append(weighted)
        reassembled = FlatReassemble(graph, inputs=expert_outputs, control=selector, reassemble_rank=0, switch_cycles=[1] * len(expert_outputs), write_back_mu=False)
        summed = Accum(graph, reassembled, output_stream_dtype=_dsl2step_out_tile(reassembled, 'elem', 2), fn=accum_fn.Add(), init_fn=_dsl2step_init(reassembled, 'elem'), accum_rank=2, write_back_mu=False, compute_bw=4096)
        return summed

    def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        normed_2 = rms_norm(res_add_0, out_shapes=out_shapes, out_perms=out_perms)
        return moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes, out_perms=out_perms)

    def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes, out_perms=None):
        moe_out = moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes, out_perms=out_perms)
        _tmp2 = BinaryMap(graph, res_add_0, moe_out, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return _tmp2
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
    res_add_0 = attention_o_proj(Q, K, V, o_proj_weight, input_tensor, out_shapes=((1, 64, 512),))
    out = moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=((1, 64, 512),))
    _store9 = OffChipStore(graph, out, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store9)