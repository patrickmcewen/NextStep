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

    def attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, k_cache, v_cache, num_token_list, *, out_shapes):
        batch = 64
        hidden = 512
        head_dim = 32
        num_heads = 16
        num_kv_heads = 4
        query_per_kvhead = 4
        K_CHUNKS = hidden // head_dim
        HALF = head_dim // 2
        x_stream = LinearOffChipLoad(input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((batch,)), tile_row=1, tile_col=hidden, par_dispatch=8, start_tile_idx=0)
        x_sq = UnaryMap(graph, x_stream, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        x_sum = UnaryMap(graph, x_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        x_mean = UnaryMap(graph, x_sum, fn=map_fn.MulImmediate(1.0 / hidden), write_back_mu=False, compute_bw=4096)
        x_eps = UnaryMap(graph, x_mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        x_rsqrt = UnaryMap(graph, x_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        x_rsqrt_rep = RepeatStatic(graph, x_rsqrt, repeat_factor=hidden)
        x_rsqrt_tiled = Accum(graph, x_rsqrt_rep, output_stream_dtype=_dsl2step_out_tile(x_rsqrt_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(x_rsqrt_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        normed = BinaryMap(graph, x_stream, x_rsqrt_tiled, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        normed_chunked = RetileStreamify(graph, normed, split_row=False, chunk=head_dim)
        normed_bc = Reshape(graph, normed_chunked, chunk_size=K_CHUNKS, reshape_rank=0, write_back_mu=False)
        normed_buffered = Bufferize(graph, normed_bc, rank=2)
        normed_for_q = Streamify(graph, normed_buffered, stride=tuple((K_CHUNKS, 0, 1)), out_shape_tiled=tuple((batch, num_heads, K_CHUNKS)))
        normed_for_k = Streamify(graph, normed_buffered, stride=tuple((K_CHUNKS, 0, 1)), out_shape_tiled=tuple((batch, num_kv_heads, K_CHUNKS)))
        normed_for_v = Streamify(graph, normed_buffered, stride=tuple((K_CHUNKS, 0, 1)), out_shape_tiled=tuple((batch, num_kv_heads, K_CHUNKS)))
        q_proj_stream = LinearOffChipLoad(q_proj, stride=tuple((0, 1, num_heads)), out_shape_tiled=tuple((batch, num_heads, K_CHUNKS)), tile_row=head_dim, tile_col=head_dim, par_dispatch=8, start_tile_idx=0)
        Q_raw = BinaryMapAccum(graph, normed_for_q, q_proj_stream, fn=map_accum_fn.Matmul(), init_fn=Zero(shape=(_dsl2step_in_tile(normed_for_q).shape[0], _dsl2step_in_tile(q_proj_stream).shape[0 if False else 1]), dtype=_dsl2step_in_tile(normed_for_q).tile_dtype), rank=1, write_back_mu=False, compute_bw=4096)
        k_proj_stream = LinearOffChipLoad(k_proj, stride=tuple((0, 1, num_kv_heads)), out_shape_tiled=tuple((batch, num_kv_heads, K_CHUNKS)), tile_row=head_dim, tile_col=head_dim, par_dispatch=8, start_tile_idx=0)
        K_raw = BinaryMapAccum(graph, normed_for_k, k_proj_stream, fn=map_accum_fn.Matmul(), init_fn=Zero(shape=(_dsl2step_in_tile(normed_for_k).shape[0], _dsl2step_in_tile(k_proj_stream).shape[0 if False else 1]), dtype=_dsl2step_in_tile(normed_for_k).tile_dtype), rank=1, write_back_mu=False, compute_bw=4096)
        v_proj_stream = LinearOffChipLoad(v_proj, stride=tuple((0, 1, num_kv_heads)), out_shape_tiled=tuple((batch, num_kv_heads, K_CHUNKS)), tile_row=head_dim, tile_col=head_dim, par_dispatch=8, start_tile_idx=0)
        V_all = BinaryMapAccum(graph, normed_for_v, v_proj_stream, fn=map_accum_fn.Matmul(), init_fn=Zero(shape=(_dsl2step_in_tile(normed_for_v).shape[0], _dsl2step_in_tile(v_proj_stream).shape[0 if False else 1]), dtype=_dsl2step_in_tile(normed_for_v).tile_dtype), rank=1, write_back_mu=False, compute_bw=4096)

        def rms_norm_stream(x_in, dim):
            sq = UnaryMap(graph, x_in, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
            s = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
            m = UnaryMap(graph, s, fn=map_fn.MulImmediate(1.0 / dim), write_back_mu=False, compute_bw=4096)
            e = UnaryMap(graph, m, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
            r = UnaryMap(graph, e, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
            rr = RepeatStatic(graph, r, repeat_factor=dim)
            rt = Accum(graph, rr, output_stream_dtype=_dsl2step_out_tile(rr, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(rr, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            _tmp1 = BinaryMap(graph, x_in, rt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            return _tmp1
        Q_normed = rms_norm_stream(Q_raw, head_dim)
        K_normed = rms_norm_stream(K_raw, head_dim)
        cos_stream = LinearOffChipLoad(cos, stride=tuple((1, 0)), out_shape_tiled=tuple((batch, 1)), tile_row=1, tile_col=head_dim, par_dispatch=8, start_tile_idx=0)
        sin_stream = LinearOffChipLoad(sin, stride=tuple((1, 0)), out_shape_tiled=tuple((batch, 1)), tile_row=1, tile_col=head_dim, par_dispatch=8, start_tile_idx=0)
        cos_q = ExpandRef(graph, cos_stream, ref=Q_normed, expand_rank=1)
        sin_q = ExpandRef(graph, sin_stream, ref=Q_normed, expand_rank=1)
        cos_k = ExpandRef(graph, cos_stream, ref=K_normed, expand_rank=1)
        sin_k = ExpandRef(graph, sin_stream, ref=K_normed, expand_rank=1)

        def apply_rope(x_in, cos_in, sin_in, n_heads):
            x_flat = Flatten(graph, x_in, min_rank=0, max_rank=2)
            cos_flat = Flatten(graph, cos_in, min_rank=0, max_rank=2)
            sin_flat = Flatten(graph, sin_in, min_rank=0, max_rank=2)
            x_split = RetileStreamify(graph, x_flat, split_row=False, chunk=HALF)
            cos_split = RetileStreamify(graph, cos_flat, split_row=False, chunk=HALF)
            sin_split = RetileStreamify(graph, sin_flat, split_row=False, chunk=HALF)
            _parallelize8 = Parallelize(graph, x_split, parallelize_rank=x_split.stream.rank, num_consumers=2)
            x_parts = [_BranchRef(_parallelize8, _i) for _i in range(2)]
            _parallelize9 = Parallelize(graph, cos_split, parallelize_rank=cos_split.stream.rank, num_consumers=2)
            cos_parts = [_BranchRef(_parallelize9, _i) for _i in range(2)]
            _parallelize10 = Parallelize(graph, sin_split, parallelize_rank=sin_split.stream.rank, num_consumers=2)
            sin_parts = [_BranchRef(_parallelize10, _i) for _i in range(2)]
            x1, x2 = (x_parts[0], x_parts[1])
            c1, c2 = (cos_parts[0], cos_parts[1])
            s1, s2 = (sin_parts[0], sin_parts[1])
            neg_x2 = UnaryMap(graph, x2, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
            _tmp2 = BinaryMap(graph, x1, c1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            _tmp3 = BinaryMap(graph, neg_x2, s1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            rope1 = BinaryMap(graph, _tmp2, _tmp3, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
            _tmp4 = BinaryMap(graph, x2, c2, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            _tmp5 = BinaryMap(graph, x1, s2, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            rope2 = BinaryMap(graph, _tmp4, _tmp5, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
            interleaved = StaticReassemble(graph, inputs=[rope1, rope2], merge_rank=_dsl2step_stream([rope1, rope2][0]).rank)
            rope_2d = Reshape(graph, interleaved, chunk_size=2, reshape_rank=0, write_back_mu=False)
            rope_full = Accum(graph, rope_2d, output_stream_dtype=_dsl2step_out_tile(rope_2d, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(rope_2d, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            rope_3d = Reshape(graph, rope_full, chunk_size=n_heads, reshape_rank=0, write_back_mu=False)
            _tmp6 = PromoteOuter(graph, rope_3d)
            return _tmp6
        Q_rope = apply_rope(Q_normed, cos_q, sin_q, num_heads)
        K_rope = apply_rope(K_normed, cos_k, sin_k, num_kv_heads)
        Q_64x16 = Flatten(graph, Q_rope, min_rank=1, max_rank=2)
        K_64x4 = Flatten(graph, K_rope, min_rank=1, max_rank=2)
        V_64x4 = Flatten(graph, V_all, min_rank=1, max_rank=2)
        _parallelize11 = Parallelize(graph, Q_64x16, parallelize_rank=Q_64x16.stream.rank, num_consumers=batch)
        Q_batched = [_BranchRef(_parallelize11, _i) for _i in range(batch)]
        _parallelize12 = Parallelize(graph, K_64x4, parallelize_rank=K_64x4.stream.rank, num_consumers=batch)
        K_batched = [_BranchRef(_parallelize12, _i) for _i in range(batch)]
        _parallelize13 = Parallelize(graph, V_64x4, parallelize_rank=V_64x4.stream.rank, num_consumers=batch)
        V_batched = [_BranchRef(_parallelize13, _i) for _i in range(batch)]
        batch_outputs = []
        for b in range(batch):
            Q_b = Q_batched[b]
            K_b = K_batched[b]
            V_b = V_batched[b]
            seq_meta = MetadataGen(tensor=num_token_list[b])
            graph.add_node(seq_meta)
            ones_u = UnaryMap(graph, seq_meta, fn=map_fn.ToConstInt(1), write_back_mu=False, compute_bw=4096)
            NTL_plus_1 = BinaryMap(graph, seq_meta, ones_u, fn=map_fn.CacheWriteAddrGen(row_offset=1), write_back_mu=False, compute_bw=4096)
            zeros_u = UnaryMap(graph, seq_meta, fn=map_fn.ToConstInt(0), write_back_mu=False, compute_bw=4096)
            ragged_addrs = CacheReadAddrGen(graph, zeros_u, NTL_plus_1, 1)
            k_zeros = RandomOffChipLoad(graph, underlying=k_cache[b], raddr=ragged_addrs, tile_row=num_kv_heads, tile_col=head_dim, base_addr_byte=0, par_dispatch=8)
            v_zeros = RandomOffChipLoad(graph, underlying=v_cache[b], raddr=ragged_addrs, tile_row=num_kv_heads, tile_col=head_dim, base_addr_byte=0, par_dispatch=8)
            last_sel = FilterLastTile(graph, NTL_plus_1)
            _flat_partition14 = FlatPartition(graph, k_zeros, control=last_sel, partition_rank=0, switch_cycles=[1] * 2, write_back_mu=False, num_consumers=2)
            k_last = _BranchRef(_flat_partition14, 0)
            k_prefix = _BranchRef(_flat_partition14, 1)
            _flat_partition15 = FlatPartition(graph, v_zeros, control=last_sel, partition_rank=0, switch_cycles=[1] * 2, write_back_mu=False, num_consumers=2)
            v_last = _BranchRef(_flat_partition15, 0)
            v_prefix = _BranchRef(_flat_partition15, 1)
            K_b_4x32 = Accum(graph, K_b, output_stream_dtype=_dsl2step_out_tile(K_b, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(K_b, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            V_b_4x32 = Accum(graph, V_b, output_stream_dtype=_dsl2step_out_tile(V_b, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(V_b, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            k_reassembled = FlatReassemble(graph, inputs=[K_b_4x32, k_prefix], control=last_sel, reassemble_rank=0, switch_cycles=[1] * len([K_b_4x32, k_prefix]), write_back_mu=False)
            k_combined = Flatten(graph, k_reassembled, min_rank=0, max_rank=1)
            v_reassembled = FlatReassemble(graph, inputs=[V_b_4x32, v_prefix], control=last_sel, reassemble_rank=0, switch_cycles=[1] * len([V_b_4x32, v_prefix]), write_back_mu=False)
            v_combined = Flatten(graph, v_reassembled, min_rank=0, max_rank=1)
            v_rows = RetileStreamify(graph, v_combined, split_row=True, chunk=1)
            v_flat = Flatten(graph, v_rows, min_rank=0, max_rank=1)
            _parallelize16 = Parallelize(graph, v_flat, parallelize_rank=v_flat.stream.rank, num_consumers=num_kv_heads)
            v_kv_streams = [_BranchRef(_parallelize16, _i) for _i in range(num_kv_heads)]
            Q_b_flat = Flatten(graph, Q_b, min_rank=0, max_rank=1)
            _parallelize17 = Parallelize(graph, Q_b_flat, parallelize_rank=Q_b_flat.stream.rank, num_consumers=num_heads)
            Q_heads = [_BranchRef(_parallelize17, _i) for _i in range(num_heads)]
            head_outputs = []
            for nh in range(num_heads):
                h = nh // query_per_kvhead
                Q_nh = Q_heads[nh]
                Q_nh_2d = Reshape(graph, Q_nh, chunk_size=1, reshape_rank=0, write_back_mu=False)
                Q_nh_exp = ExpandRef(graph, Q_nh_2d, ref=k_combined, expand_rank=1)
                scores_all = BinaryMap(graph, Q_nh_exp, k_combined, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=4096)
                scores_split = RetileStreamify(graph, scores_all, split_row=False, chunk=1)
                scores_flat = Flatten(graph, scores_split, min_rank=0, max_rank=1)
                _parallelize18 = Parallelize(graph, scores_flat, parallelize_rank=scores_flat.stream.rank, num_consumers=num_kv_heads)
                scores_by_kv = [_BranchRef(_parallelize18, _i) for _i in range(num_kv_heads)]
                scores_h = scores_by_kv[h]
                v_h = v_kv_streams[h]
                _broadcast19 = Broadcast(graph, scores_h, num_consumers=3)
                scores_bc = [_BranchRef(_broadcast19, _i) for _i in range(3)]
                max_s = Accum(graph, scores_bc[0], output_stream_dtype=_dsl2step_out_tile(scores_bc[0], 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(scores_bc[0], 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                max_exp = RepeatRef(graph, max_s, ref=scores_bc[1])
                neg_max = UnaryMap(graph, max_exp, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
                scores_shifted = BinaryMap(graph, scores_bc[2], neg_max, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
                exp_s = UnaryMap(graph, scores_shifted, fn=map_fn.Exp(), write_back_mu=False, compute_bw=4096)
                _broadcast20 = Broadcast(graph, exp_s, num_consumers=2)
                exp_bc = [_BranchRef(_broadcast20, _i) for _i in range(2)]
                denom = Accum(graph, exp_bc[0], output_stream_dtype=_dsl2step_out_tile(exp_bc[0], 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(exp_bc[0], 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                weighted = BinaryMap(graph, v_h, exp_bc[1], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
                context = Accum(graph, weighted, output_stream_dtype=_dsl2step_out_tile(weighted, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(weighted, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                denom_rep = RepeatStatic(graph, denom, repeat_factor=head_dim)
                denom_tiled = Accum(graph, denom_rep, output_stream_dtype=_dsl2step_out_tile(denom_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(denom_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                out_nh = BinaryMap(graph, context, denom_tiled, fn=map_fn.Div(), write_back_mu=False, compute_bw=4096)
                out_nh_streamed = RepeatStatic(graph, out_nh, repeat_factor=1)
                head_outputs.append(out_nh_streamed)
            batch_out = StaticReassemble(graph, inputs=head_outputs, merge_rank=_dsl2step_stream(head_outputs[0]).rank)
            batch_outputs.append(batch_out)
        _eager_merge21 = EagerMerge(graph, batch_outputs, input_rank=1)
        merged_data = _BranchRef(_eager_merge21, 0)
        _ = _BranchRef(_eager_merge21, 1)
        result = Reshape(graph, merged_data, chunk_size=num_heads, reshape_rank=0, write_back_mu=False)
        return result

    def moe(normed_input, expert_onehot, expert_weights, w_gate_list, w_up_list, w_down_list, *, out_shapes):
        batch = 64
        n_experts = 8
        n_active = 2
        D = 512
        F_dim = 1792
        TILE_F = 32
        F_TILES = F_dim // TILE_F
        sel = SelectGen(is_multihot=True, tensor=expert_onehot, n=n_experts)
        graph.add_node(sel)
        ni_promoted = PromoteOuter(graph, normed_input)
        ni_repeated = RepeatStatic(graph, ni_promoted, repeat_factor=n_active)
        _flat_partition22 = FlatPartition(graph, ni_repeated, control=sel, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
        partitioned = [_BranchRef(_flat_partition22, _i) for _i in range(n_experts)]
        expert_outputs = []
        for i in range(n_experts):
            xi = partitioned[i]
            gate_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_gate_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=D, tile_col=TILE_F, par_dispatch=8, start_tile_idx=0)
            up_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_up_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=D, tile_col=TILE_F, par_dispatch=8, start_tile_idx=0)
            xi_exp = RepeatRef(graph, xi, ref=gate_w)
            gate_out = BinaryMap(graph, xi_exp, gate_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            up_out = BinaryMap(graph, xi_exp, up_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            _tmp7 = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
            hidden = BinaryMap(graph, _tmp7, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            down_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_down_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=TILE_F, tile_col=D, par_dispatch=8, start_tile_idx=0)
            down_out = BinaryMapAccum(graph, hidden, down_w, fn=map_accum_fn.Matmul(), init_fn=Zero(shape=(_dsl2step_in_tile(hidden).shape[0], _dsl2step_in_tile(down_w).shape[0 if False else 1]), dtype=_dsl2step_in_tile(hidden).tile_dtype), rank=1, write_back_mu=False, compute_bw=4096)
            expert_outputs.append(down_out)
        reassembled = FlatReassemble(graph, inputs=expert_outputs, control=sel, reassemble_rank=0, switch_cycles=[1] * len(expert_outputs), write_back_mu=False)
        per_slot = Accum(graph, reassembled, output_stream_dtype=_dsl2step_out_tile(reassembled, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(reassembled, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        ew_stream = LinearOffChipLoad(expert_weights, stride=tuple((n_active, 1)), out_shape_tiled=tuple((batch, n_active)), tile_row=1, tile_col=1, par_dispatch=8, start_tile_idx=0)
        weighted = BinaryMap(graph, per_slot, ew_stream, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        output = Accum(graph, weighted, output_stream_dtype=_dsl2step_out_tile(weighted, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(weighted, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        output_flat = Flatten(graph, output, min_rank=0, max_rank=1)
        output_final = Promote(graph, output_flat, promote_rank=0)
        return output_final
    B = 64
    N_HEADS = 16
    HEAD_DIM = 32
    D = 512
    TILE_N = 32
    W_CHUNKS = D // TILE_N
    attn_out = attention(tensors['input_tensor'], tensors['q_proj'], tensors['k_proj'], tensors['v_proj'], tensors['cos'], tensors['sin'], tensors['k_cache'], tensors['v_cache'], tensors['num_token_list'], out_shapes=((B, N_HEADS, 1, HEAD_DIM),))
    attn_merged = Accum(graph, attn_out, output_stream_dtype=_dsl2step_out_tile(attn_out, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn_out, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    o_proj_w = LinearOffChipLoadRef(graph, ref=attn_merged, underlying=tensors['o_proj_weight'], stride=tuple((1,)), out_shape_tiled=tuple((W_CHUNKS,)), tile_row=D, tile_col=TILE_N, par_dispatch=8, start_tile_idx=0)
    attn_exp = RepeatRef(graph, attn_merged, ref=o_proj_w)
    o_mm = BinaryMap(graph, attn_exp, o_proj_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
    o_proj_result = Accum(graph, o_mm, output_stream_dtype=_dsl2step_out_tile(o_mm, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(o_mm, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    inp_stream_raw = LinearOffChipLoad(tensors['input_tensor'], stride=tuple((1,)), out_shape_tiled=tuple((B,)), tile_row=1, tile_col=D, par_dispatch=8, start_tile_idx=0)
    inp_stream = Flatten(graph, inp_stream_raw, min_rank=0, max_rank=1)
    res_add_0 = BinaryMap(graph, o_proj_result, inp_stream, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
    x_sq = UnaryMap(graph, res_add_0, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
    x_sum = UnaryMap(graph, x_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
    x_mean = UnaryMap(graph, x_sum, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=4096)
    x_eps = UnaryMap(graph, x_mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
    x_rsqrt = UnaryMap(graph, x_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
    x_rsqrt_rep = RepeatStatic(graph, x_rsqrt, repeat_factor=D)
    x_rsqrt_tiled = Accum(graph, x_rsqrt_rep, output_stream_dtype=_dsl2step_out_tile(x_rsqrt_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(x_rsqrt_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    normed_2 = BinaryMap(graph, res_add_0, x_rsqrt_tiled, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
    moe_out_raw = moe(normed_2, tensors['expert_onehot'], tensors['expert_weights'], tensors['w_gate_list'], tensors['w_up_list'], tensors['w_down_list'], out_shapes=((B, 1, 1, D),))
    moe_out = Flatten(graph, moe_out_raw, min_rank=0, max_rank=1)
    final = BinaryMap(graph, moe_out, res_add_0, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
    final_out = PromoteOuter(graph, final)
    _store23 = OffChipStore(graph, final_out, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store23)
