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
    B = 64
    D = 512
    HEAD_DIM = 32
    NUM_HEADS = 16
    NUM_KV = 4
    QPK = 4
    N_EXPERTS = 8
    N_ACTIVE = 2
    FFN_DIM = 1792
    FFN_TILE = 32
    FFN_CHUNKS = FFN_DIM // FFN_TILE
    TILE_N = 32
    W_CHUNKS = D // TILE_N
    HALF = HEAD_DIM // 2

    def rms_norm_stream(x, dim):
        sq = UnaryMap(graph, x, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        s = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        m = UnaryMap(graph, s, fn=map_fn.MulImmediate(1.0 / dim), write_back_mu=False, compute_bw=4096)
        e = UnaryMap(graph, m, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        r = UnaryMap(graph, e, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        rr = RepeatStatic(graph, r, repeat_factor=dim)
        rt = Accum(graph, rr, output_stream_dtype=_dsl2step_out_tile(rr, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(rr, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        _tmp1 = BinaryMap(graph, x, rt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        return _tmp1
    inp_raw = LinearOffChipLoad(tensors['input_tensor'], stride=tuple((1,)), out_shape_tiled=tuple((B,)), tile_row=1, tile_col=D, par_dispatch=8, start_tile_idx=0)
    inp = Flatten(graph, inp_raw, min_rank=0, max_rank=1)
    normed = rms_norm_stream(inp, D)
    _tmp2 = LinearOffChipLoad(tensors['q_proj'], stride=tuple((0, 1)), out_shape_tiled=tuple((B, NUM_HEADS)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
    q_proj_w = Flatten(graph, _tmp2, min_rank=1, max_rank=2)
    normed_q = RepeatRef(graph, normed, ref=q_proj_w)
    Q_raw = BinaryMap(graph, normed_q, q_proj_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
    _tmp3 = LinearOffChipLoad(tensors['k_proj'], stride=tuple((0, 1)), out_shape_tiled=tuple((B, NUM_KV)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
    k_proj_w = Flatten(graph, _tmp3, min_rank=1, max_rank=2)
    normed_k = RepeatRef(graph, normed, ref=k_proj_w)
    K_raw = BinaryMap(graph, normed_k, k_proj_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
    _tmp4 = LinearOffChipLoad(tensors['v_proj'], stride=tuple((0, 1)), out_shape_tiled=tuple((B, NUM_KV)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
    v_proj_w = Flatten(graph, _tmp4, min_rank=1, max_rank=2)
    normed_v = RepeatRef(graph, normed, ref=v_proj_w)
    V_raw = BinaryMap(graph, normed_v, v_proj_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
    Q_normed = rms_norm_stream(Q_raw, HEAD_DIM)
    K_normed = rms_norm_stream(K_raw, HEAD_DIM)
    _tmp5 = LinearOffChipLoad(tensors['cos'], stride=tuple((1, 0)), out_shape_tiled=tuple((B, 1)), tile_row=1, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
    cos_s = Flatten(graph, _tmp5, min_rank=0, max_rank=2)
    _tmp6 = LinearOffChipLoad(tensors['sin'], stride=tuple((1, 0)), out_shape_tiled=tuple((B, 1)), tile_row=1, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
    sin_s = Flatten(graph, _tmp6, min_rank=0, max_rank=2)

    def apply_rope(x_heads, cos_1d, sin_1d, n_heads):
        """
        x_heads: stream(B, n_heads), tile(1, HEAD_DIM)
        cos_1d, sin_1d: stream(B,), tile(1, HEAD_DIM)
        Returns: stream(B, n_heads), tile(1, HEAD_DIM)
        
        Fix: flatten to 1D then parallelize by 2 to correctly split
        first-half and second-half tiles.
        """
        cos_exp = RepeatRef(graph, cos_1d, ref=x_heads)
        sin_exp = RepeatRef(graph, sin_1d, ref=x_heads)
        x_split = RetileStreamify(graph, x_heads, split_row=False, chunk=HALF)
        cos_split = RetileStreamify(graph, cos_exp, split_row=False, chunk=HALF)
        sin_split = RetileStreamify(graph, sin_exp, split_row=False, chunk=HALF)
        x_flat = Flatten(graph, x_split, min_rank=0, max_rank=1)
        cos_flat = Flatten(graph, cos_split, min_rank=0, max_rank=1)
        sin_flat = Flatten(graph, sin_split, min_rank=0, max_rank=1)
        _parallelize14 = Parallelize(graph, x_flat, parallelize_rank=x_flat.stream.rank, num_consumers=2)
        x_halves = [_BranchRef(_parallelize14, _i) for _i in range(2)]
        _parallelize15 = Parallelize(graph, cos_flat, parallelize_rank=cos_flat.stream.rank, num_consumers=2)
        cos_halves = [_BranchRef(_parallelize15, _i) for _i in range(2)]
        _parallelize16 = Parallelize(graph, sin_flat, parallelize_rank=sin_flat.stream.rank, num_consumers=2)
        sin_halves = [_BranchRef(_parallelize16, _i) for _i in range(2)]
        x1, x2 = (x_halves[0], x_halves[1])
        c1, c2 = (cos_halves[0], cos_halves[1])
        s1, s2 = (sin_halves[0], sin_halves[1])
        neg_x2 = UnaryMap(graph, x2, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        _tmp7 = BinaryMap(graph, x1, c1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp8 = BinaryMap(graph, neg_x2, s1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        out1 = BinaryMap(graph, _tmp7, _tmp8, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        _tmp9 = BinaryMap(graph, x2, c2, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        _tmp10 = BinaryMap(graph, x1, s2, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        out2 = BinaryMap(graph, _tmp9, _tmp10, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        interleaved = StaticReassemble(graph, inputs=[out1, out2], merge_rank=_dsl2step_stream([out1, out2][0]).rank)
        inter_2d = Reshape(graph, interleaved, chunk_size=2, reshape_rank=0, write_back_mu=False)
        merged = Accum(graph, inter_2d, output_stream_dtype=_dsl2step_out_tile(inter_2d, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(inter_2d, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        result = Reshape(graph, merged, chunk_size=n_heads, reshape_rank=0, write_back_mu=False)
        return result
    Q_rope = apply_rope(Q_normed, cos_s, sin_s, NUM_HEADS)
    K_rope = apply_rope(K_normed, cos_s, sin_s, NUM_KV)
    K_batched = Accum(graph, K_rope, output_stream_dtype=_dsl2step_out_tile(K_rope, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(K_rope, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    V_batched = Accum(graph, V_raw, output_stream_dtype=_dsl2step_out_tile(V_raw, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(V_raw, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    _parallelize17 = Parallelize(graph, K_batched, parallelize_rank=K_batched.stream.rank, num_consumers=B)
    K_by_batch = [_BranchRef(_parallelize17, _i) for _i in range(B)]
    _parallelize18 = Parallelize(graph, V_batched, parallelize_rank=V_batched.stream.rank, num_consumers=B)
    V_by_batch = [_BranchRef(_parallelize18, _i) for _i in range(B)]
    _parallelize19 = Parallelize(graph, Q_rope, parallelize_rank=Q_rope.stream.rank, num_consumers=B)
    Q_by_batch = [_BranchRef(_parallelize19, _i) for _i in range(B)]
    attn_outputs_per_batch = []
    for b in range(B):
        idx_b_meta = MetadataGen(tensor=tensors['idx'][b])
        graph.add_node(idx_b_meta)
        seq_b_meta = MetadataGen(tensor=tensors['seq_len'][b])
        graph.add_node(seq_b_meta)
        idx_zero_b = UnaryMap(graph, idx_b_meta, fn=map_fn.ToConstInt(0), write_back_mu=False, compute_bw=4096)
        raddr_b = CacheReadAddrGen(graph, idx_zero_b, seq_b_meta, 1)
        k_b_cache = RandomOffChipLoad(graph, underlying=tensors['k_cache'][b], raddr=raddr_b, tile_row=NUM_KV, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
        v_b_cache = RandomOffChipLoad(graph, underlying=tensors['v_cache'][b], raddr=raddr_b, tile_row=NUM_KV, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
        K_b_new = K_by_batch[b]
        V_b_new = V_by_batch[b]
        last_sel_b = FilterLastTile(graph, seq_b_meta)
        _flat_partition20 = FlatPartition(graph, k_b_cache, control=last_sel_b, partition_rank=0, switch_cycles=[1] * 2, write_back_mu=False, num_consumers=2)
        k_last_b = _BranchRef(_flat_partition20, 0)
        k_prefix_b = _BranchRef(_flat_partition20, 1)
        _flat_partition21 = FlatPartition(graph, v_b_cache, control=last_sel_b, partition_rank=0, switch_cycles=[1] * 2, write_back_mu=False, num_consumers=2)
        v_last_b = _BranchRef(_flat_partition21, 0)
        v_prefix_b = _BranchRef(_flat_partition21, 1)
        k_last_off = BinaryMap(graph, k_last_b, idx_zero_b, fn=map_fn.SetOffset(), write_back_mu=False, compute_bw=4096)
        k_last_updated = BinaryMap(graph, k_last_off, K_b_new, fn=map_fn.RowWiseAppend(), write_back_mu=False, compute_bw=4096)
        v_last_off = BinaryMap(graph, v_last_b, idx_zero_b, fn=map_fn.SetOffset(), write_back_mu=False, compute_bw=4096)
        v_last_updated = BinaryMap(graph, v_last_off, V_b_new, fn=map_fn.RowWiseAppend(), write_back_mu=False, compute_bw=4096)
        k_full_raw = FlatReassemble(graph, inputs=[k_last_updated, k_prefix_b], control=last_sel_b, reassemble_rank=0, switch_cycles=[1] * len([k_last_updated, k_prefix_b]), write_back_mu=False)
        k_full_b = Flatten(graph, k_full_raw, min_rank=0, max_rank=1)
        v_full_raw = FlatReassemble(graph, inputs=[v_last_updated, v_prefix_b], control=last_sel_b, reassemble_rank=0, switch_cycles=[1] * len([v_last_updated, v_prefix_b]), write_back_mu=False)
        v_full_b = Flatten(graph, v_full_raw, min_rank=0, max_rank=1)
        v_b_rows = RetileStreamify(graph, v_full_b, split_row=True, chunk=1)
        v_b_flat = Flatten(graph, v_b_rows, min_rank=0, max_rank=1)
        _parallelize22 = Parallelize(graph, v_b_flat, parallelize_rank=v_b_flat.stream.rank, num_consumers=NUM_KV)
        v_b_by_head = [_BranchRef(_parallelize22, _i) for _i in range(NUM_KV)]
        Q_b = Q_by_batch[b]
        Q_b_flat16 = Flatten(graph, Q_b, min_rank=0, max_rank=1)
        Q_b_grouped = Reshape(graph, Q_b_flat16, chunk_size=QPK, reshape_rank=0, write_back_mu=False)
        Q_b_kv = Accum(graph, Q_b_grouped, output_stream_dtype=_dsl2step_out_tile(Q_b_grouped, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(Q_b_grouped, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        _parallelize23 = Parallelize(graph, Q_b_kv, parallelize_rank=Q_b_kv.stream.rank, num_consumers=NUM_KV)
        Q_b_by_kv = [_BranchRef(_parallelize23, _i) for _i in range(NUM_KV)]
        kv_results = []
        for kv_h in range(NUM_KV):
            Q_kv_h = Q_b_by_kv[kv_h]
            v_kv = v_b_by_head[kv_h]
            Q_kv_h_exp = RepeatRef(graph, Q_kv_h, ref=k_full_b)
            scores_4x4 = BinaryMap(graph, Q_kv_h_exp, k_full_b, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=4096)
            scores_cols = RetileStreamify(graph, scores_4x4, split_row=False, chunk=1)
            scores_flat = Flatten(graph, scores_cols, min_rank=0, max_rank=1)
            _parallelize24 = Parallelize(graph, scores_flat, parallelize_rank=scores_flat.stream.rank, num_consumers=NUM_KV)
            scores_by_kv = [_BranchRef(_parallelize24, _i) for _i in range(NUM_KV)]
            scores_kv_h = scores_by_kv[kv_h]
            row_max = Accum(graph, scores_kv_h, output_stream_dtype=_dsl2step_out_tile(scores_kv_h, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(scores_kv_h, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            row_max_exp = RepeatRef(graph, row_max, ref=scores_kv_h)
            _tmp11 = UnaryMap(graph, row_max_exp, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
            scores_shifted = BinaryMap(graph, scores_kv_h, _tmp11, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
            exp_s = UnaryMap(graph, scores_shifted, fn=map_fn.Exp(), write_back_mu=False, compute_bw=4096)
            context_num = BinaryMap(graph, exp_s, v_kv, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            context_sum = Accum(graph, context_num, output_stream_dtype=_dsl2step_out_tile(context_num, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(context_num, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            denom = Accum(graph, exp_s, output_stream_dtype=_dsl2step_out_tile(exp_s, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(exp_s, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            denom_rep = RepeatStatic(graph, denom, repeat_factor=HEAD_DIM)
            denom_tiled = Accum(graph, denom_rep, output_stream_dtype=_dsl2step_out_tile(denom_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(denom_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            context_kv = BinaryMap(graph, context_sum, denom_tiled, fn=map_fn.Div(), write_back_mu=False, compute_bw=4096)
            context_kv_1d = Promote(graph, context_kv, promote_rank=0)
            heads_split = RetileStreamify(graph, context_kv_1d, split_row=True, chunk=1)
            kv_results.append(heads_split)
        _eager_merge25 = EagerMerge(graph, kv_results, input_rank=1)
        attn_b_data = _BranchRef(_eager_merge25, 0)
        _ = _BranchRef(_eager_merge25, 1)
        attn_outputs_per_batch.append(attn_b_data)
    _eager_merge26 = EagerMerge(graph, attn_outputs_per_batch, input_rank=1)
    attn_all = _BranchRef(_eager_merge26, 0)
    _ = _BranchRef(_eager_merge26, 1)
    attn_2d = Reshape(graph, attn_all, chunk_size=NUM_HEADS, reshape_rank=0, write_back_mu=False)
    attn_merged = Accum(graph, attn_2d, output_stream_dtype=_dsl2step_out_tile(attn_2d, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn_2d, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    _tmp12 = LinearOffChipLoad(tensors['o_proj_weight'], stride=tuple((0, 1)), out_shape_tiled=tuple((B, W_CHUNKS)), tile_row=D, tile_col=TILE_N, par_dispatch=8, start_tile_idx=0)
    o_proj_w = Flatten(graph, _tmp12, min_rank=1, max_rank=2)
    attn_for_proj = RepeatRef(graph, attn_merged, ref=o_proj_w)
    o_mm = BinaryMap(graph, attn_for_proj, o_proj_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
    o_proj_result = Accum(graph, o_mm, output_stream_dtype=_dsl2step_out_tile(o_mm, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(o_mm, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    res_add_0 = BinaryMap(graph, o_proj_result, inp, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
    normed_2 = rms_norm_stream(res_add_0, D)
    sel_moe = SelectGen(is_multihot=True, tensor=tensors['expert_onehot'], n=N_EXPERTS)
    graph.add_node(sel_moe)
    normed_2_outer = PromoteOuter(graph, normed_2)
    normed_2_rep = RepeatStatic(graph, normed_2_outer, repeat_factor=N_ACTIVE)
    _flat_partition27 = FlatPartition(graph, normed_2_rep, control=sel_moe, partition_rank=0, switch_cycles=[1] * N_EXPERTS, write_back_mu=False, num_consumers=N_EXPERTS)
    parts_moe = [_BranchRef(_flat_partition27, _i) for _i in range(N_EXPERTS)]
    expert_results = []
    for e in range(N_EXPERTS):
        xi = parts_moe[e]
        gate_w = LinearOffChipLoadRef(graph, ref=xi, underlying=tensors['w_gate_list'][e], stride=tuple((1,)), out_shape_tiled=tuple((FFN_CHUNKS,)), tile_row=D, tile_col=FFN_TILE, par_dispatch=8, start_tile_idx=0)
        up_w = LinearOffChipLoadRef(graph, ref=xi, underlying=tensors['w_up_list'][e], stride=tuple((1,)), out_shape_tiled=tuple((FFN_CHUNKS,)), tile_row=D, tile_col=FFN_TILE, par_dispatch=8, start_tile_idx=0)
        xi_exp = RepeatRef(graph, xi, ref=gate_w)
        gate_out = BinaryMap(graph, xi_exp, gate_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        up_out = BinaryMap(graph, xi_exp, up_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        _tmp13 = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
        hidden = BinaryMap(graph, _tmp13, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        down_w = LinearOffChipLoadRef(graph, ref=xi, underlying=tensors['w_down_list'][e], stride=tuple((1,)), out_shape_tiled=tuple((FFN_CHUNKS,)), tile_row=FFN_TILE, tile_col=D, par_dispatch=8, start_tile_idx=0)
        down_out = BinaryMapAccum(graph, hidden, down_w, fn=map_accum_fn.Matmul(), init_fn=Zero(shape=(_dsl2step_in_tile(hidden).shape[0], _dsl2step_in_tile(down_w).shape[0 if False else 1]), dtype=_dsl2step_in_tile(hidden).tile_dtype), rank=1, write_back_mu=False, compute_bw=4096)
        expert_results.append(down_out)
    moe_reassembled = FlatReassemble(graph, inputs=expert_results, control=sel_moe, reassemble_rank=0, switch_cycles=[1] * len(expert_results), write_back_mu=False)
    per_slot_sum = Accum(graph, moe_reassembled, output_stream_dtype=_dsl2step_out_tile(moe_reassembled, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(moe_reassembled, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    ew_stream = LinearOffChipLoad(tensors['expert_weights'], stride=tuple((N_ACTIVE, 1)), out_shape_tiled=tuple((B, N_ACTIVE)), tile_row=1, tile_col=1, par_dispatch=8, start_tile_idx=0)
    per_slot_flat = Flatten(graph, per_slot_sum, min_rank=1, max_rank=2)
    ew_flat = Flatten(graph, ew_stream, min_rank=1, max_rank=2)
    weighted_moe = BinaryMap(graph, per_slot_flat, ew_flat, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
    moe_out = Accum(graph, weighted_moe, output_stream_dtype=_dsl2step_out_tile(weighted_moe, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(weighted_moe, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    final = BinaryMap(graph, moe_out, res_add_0, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
    final_out = PromoteOuter(graph, final)
    _store28 = OffChipStore(graph, final_out, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store28)