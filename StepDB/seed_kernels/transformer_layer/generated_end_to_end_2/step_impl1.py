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

    def attention_block_1(input_tensor, q_proj, k_proj, v_proj, cos, sin, k_cache, v_cache, num_token_list, o_proj_weight, *, out_shapes):
        B = 64
        D = 512
        HEAD_DIM = 32
        HALF = 16
        NUM_HEADS = 16
        NUM_KV_HEADS = 4
        QPK = 4
        MAX_N = 4096
        D_CHUNKS = D // HEAD_DIM
        KV_D_CHUNKS = NUM_KV_HEADS * HEAD_DIM // HEAD_DIM
        PAR_FACTOR = 16
        CBW = 16
        x_raw = LinearOffChipLoad(input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((B,)), tile_row=1, tile_col=D, par_dispatch=8, start_tile_idx=0)
        x = Flatten(graph, x_raw, min_rank=0, max_rank=1)
        x_sq = UnaryMap(graph, x, fn=map_fn.Square(), write_back_mu=False, compute_bw=1)
        x_rsum = UnaryMap(graph, x_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
        x_mean = UnaryMap(graph, x_rsum, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=CBW)
        x_eps = UnaryMap(graph, x_mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=CBW)
        x_inv = UnaryMap(graph, x_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=CBW)
        x_inv_rep = RepeatStatic(graph, x_inv, repeat_factor=D)
        x_inv_tiled = Accum(graph, x_inv_rep, output_stream_dtype=_dsl2step_out_tile(x_inv_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(x_inv_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
        normed = BinaryMap(graph, x, x_inv_tiled, fn=map_fn.Mul(), write_back_mu=False, compute_bw=CBW)
        q_w = LinearOffChipLoadRef(graph, ref=normed, underlying=q_proj, stride=tuple((1,)), out_shape_tiled=tuple((D_CHUNKS,)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        kv_w_k = LinearOffChipLoadRef(graph, ref=normed, underlying=k_proj, stride=tuple((1,)), out_shape_tiled=tuple((KV_D_CHUNKS,)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        kv_w_v = LinearOffChipLoadRef(graph, ref=normed, underlying=v_proj, stride=tuple((1,)), out_shape_tiled=tuple((KV_D_CHUNKS,)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        _parallelize15 = Parallelize(graph, normed, parallelize_rank=normed.stream.rank, num_consumers=PAR_FACTOR)
        normed_par = [_BranchRef(_parallelize15, _i) for _i in range(PAR_FACTOR)]
        _parallelize16 = Parallelize(graph, q_w, parallelize_rank=q_w.stream.rank, num_consumers=PAR_FACTOR)
        q_w_par = [_BranchRef(_parallelize16, _i) for _i in range(PAR_FACTOR)]
        _parallelize17 = Parallelize(graph, kv_w_k, parallelize_rank=kv_w_k.stream.rank, num_consumers=PAR_FACTOR)
        kv_k_par = [_BranchRef(_parallelize17, _i) for _i in range(PAR_FACTOR)]
        _parallelize18 = Parallelize(graph, kv_w_v, parallelize_rank=kv_w_v.stream.rank, num_consumers=PAR_FACTOR)
        kv_v_par = [_BranchRef(_parallelize18, _i) for _i in range(PAR_FACTOR)]
        Q_parts = []
        K_parts = []
        V_parts = []
        for i in range(PAR_FACTOR):
            n_q = RepeatRef(graph, normed_par[i], ref=q_w_par[i])
            _tmp1 = BinaryMap(graph, n_q, q_w_par[i], fn=map_fn.Matmul(), write_back_mu=False, compute_bw=CBW)
            Q_parts.append(_tmp1)
            n_k = RepeatRef(graph, normed_par[i], ref=kv_k_par[i])
            _tmp2 = BinaryMap(graph, n_k, kv_k_par[i], fn=map_fn.Matmul(), write_back_mu=False, compute_bw=CBW)
            K_parts.append(_tmp2)
            n_v = RepeatRef(graph, normed_par[i], ref=kv_v_par[i])
            _tmp3 = BinaryMap(graph, n_v, kv_v_par[i], fn=map_fn.Matmul(), write_back_mu=False, compute_bw=CBW)
            V_parts.append(_tmp3)
        Q_full = StaticReassemble(graph, inputs=Q_parts, merge_rank=_dsl2step_stream(Q_parts[0]).rank)
        K_full = StaticReassemble(graph, inputs=K_parts, merge_rank=_dsl2step_stream(K_parts[0]).rank)
        V_full = StaticReassemble(graph, inputs=V_parts, merge_rank=_dsl2step_stream(V_parts[0]).rank)

        def per_head_rms(h_stream, dim):
            sq = UnaryMap(graph, h_stream, fn=map_fn.Square(), write_back_mu=False, compute_bw=CBW)
            rs = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
            m = UnaryMap(graph, rs, fn=map_fn.MulImmediate(1.0 / dim), write_back_mu=False, compute_bw=CBW)
            e = UnaryMap(graph, m, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=CBW)
            inv = UnaryMap(graph, e, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=CBW)
            irep = RepeatStatic(graph, inv, repeat_factor=dim)
            itil = Accum(graph, irep, output_stream_dtype=_dsl2step_out_tile(irep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(irep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
            _tmp4 = BinaryMap(graph, h_stream, itil, fn=map_fn.Mul(), write_back_mu=False, compute_bw=CBW)
            return _tmp4
        Q_normed = per_head_rms(Q_full, HEAD_DIM)
        K_normed = per_head_rms(K_full, HEAD_DIM)
        cos_raw = LinearOffChipLoad(cos, stride=tuple((1, 0)), out_shape_tiled=tuple((B, 1)), tile_row=1, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        sin_raw = LinearOffChipLoad(sin, stride=tuple((1, 0)), out_shape_tiled=tuple((B, 1)), tile_row=1, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        cos_s = Flatten(graph, cos_raw, min_rank=0, max_rank=2)
        sin_s = Flatten(graph, sin_raw, min_rank=0, max_rank=2)

        def apply_rope(heads, n_heads):
            cos_bc = RepeatStatic(graph, cos_s, repeat_factor=n_heads)
            sin_bc = RepeatStatic(graph, sin_s, repeat_factor=n_heads)
            h_sp = RetileStreamify(graph, heads, split_row=False, chunk=HALF)
            c_sp = RetileStreamify(graph, cos_bc, split_row=False, chunk=HALF)
            s_sp = RetileStreamify(graph, sin_bc, split_row=False, chunk=HALF)
            h_f = Flatten(graph, h_sp, min_rank=0, max_rank=1)
            c_f = Flatten(graph, c_sp, min_rank=0, max_rank=1)
            s_f = Flatten(graph, s_sp, min_rank=0, max_rank=1)
            _parallelize19 = Parallelize(graph, h_f, parallelize_rank=h_f.stream.rank, num_consumers=2)
            hh = [_BranchRef(_parallelize19, _i) for _i in range(2)]
            _parallelize20 = Parallelize(graph, c_f, parallelize_rank=c_f.stream.rank, num_consumers=2)
            ch = [_BranchRef(_parallelize20, _i) for _i in range(2)]
            _parallelize21 = Parallelize(graph, s_f, parallelize_rank=s_f.stream.rank, num_consumers=2)
            sh = [_BranchRef(_parallelize21, _i) for _i in range(2)]
            x1, x2 = (hh[0], hh[1])
            c1, c2 = (ch[0], ch[1])
            s1, s2 = (sh[0], sh[1])
            neg_x2 = UnaryMap(graph, x2, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=CBW)
            _tmp5 = BinaryMap(graph, x1, c1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=CBW)
            _tmp6 = BinaryMap(graph, neg_x2, s1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=CBW)
            r1 = BinaryMap(graph, _tmp5, _tmp6, fn=map_fn.Add(), write_back_mu=False, compute_bw=CBW)
            _tmp7 = BinaryMap(graph, x2, c2, fn=map_fn.Mul(), write_back_mu=False, compute_bw=CBW)
            _tmp8 = BinaryMap(graph, x1, s2, fn=map_fn.Mul(), write_back_mu=False, compute_bw=CBW)
            r2 = BinaryMap(graph, _tmp7, _tmp8, fn=map_fn.Add(), write_back_mu=False, compute_bw=CBW)
            reassembled = StaticReassemble(graph, inputs=[r1, r2], merge_rank=_dsl2step_stream([r1, r2][0]).rank)
            rs2 = Reshape(graph, reassembled, chunk_size=2, reshape_rank=0, write_back_mu=False)
            merged = Accum(graph, rs2, output_stream_dtype=_dsl2step_out_tile(rs2, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(rs2, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
            restored = Reshape(graph, merged, chunk_size=n_heads, reshape_rank=0, write_back_mu=False)
            return restored
        Q_rope = apply_rope(Q_normed, NUM_HEADS)
        K_rope = apply_rope(K_normed, NUM_KV_HEADS)
        K_new = Accum(graph, K_rope, output_stream_dtype=_dsl2step_out_tile(K_rope, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(K_rope, 'row'), accum_rank=1, write_back_mu=False, compute_bw=1)
        V_new = Accum(graph, V_full, output_stream_dtype=_dsl2step_out_tile(V_full, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(V_full, 'row'), accum_rank=1, write_back_mu=False, compute_bw=1)
        ntl_meta = MetadataGen(tensor=num_token_list)
        graph.add_node(ntl_meta)
        ntl = Flatten(graph, ntl_meta, min_rank=0, max_rank=1)
        _parallelize22 = Parallelize(graph, K_new, parallelize_rank=K_new.stream.rank, num_consumers=B)
        K_bwr = [_BranchRef(_parallelize22, _i) for _i in range(B)]
        _parallelize23 = Parallelize(graph, V_new, parallelize_rank=V_new.stream.rank, num_consumers=B)
        V_bwr = [_BranchRef(_parallelize23, _i) for _i in range(B)]
        _parallelize24 = Parallelize(graph, ntl, parallelize_rank=ntl.stream.rank, num_consumers=B)
        ntl_wr = [_BranchRef(_parallelize24, _i) for _i in range(B)]
        for b in range(B):
            nb = ntl_wr[b]
            bidx = UnaryMap(graph, nb, fn=map_fn.ToConstInt(b), write_back_mu=False, compute_bw=1)
            kwa = BinaryMap(graph, bidx, nb, fn=map_fn.CacheWriteAddrGen(row_offset=MAX_N), write_back_mu=False, compute_bw=1)
            _tmp9 = RandomOffChipStore(graph, underlying=k_cache, wdata=K_bwr[b], waddr=kwa, tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
            vwa = BinaryMap(graph, bidx, nb, fn=map_fn.CacheWriteAddrGen(row_offset=MAX_N), write_back_mu=False, compute_bw=1)
            _tmp10 = RandomOffChipStore(graph, underlying=v_cache, wdata=V_bwr[b], waddr=vwa, tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
        Q_for_attn = Accum(graph, Q_rope, output_stream_dtype=_dsl2step_out_tile(Q_rope, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(Q_rope, 'row'), accum_rank=1, write_back_mu=False, compute_bw=1)
        Q_grp_flat = RetileStreamify(graph, Q_for_attn, split_row=True, chunk=QPK)
        Q_groups = Reshape(graph, Q_grp_flat, chunk_size=NUM_KV_HEADS, reshape_rank=0, write_back_mu=False)
        _parallelize25 = Parallelize(graph, Q_groups, parallelize_rank=Q_groups.stream.rank, num_consumers=B)
        Q_batched = [_BranchRef(_parallelize25, _i) for _i in range(B)]
        _parallelize26 = Parallelize(graph, K_new, parallelize_rank=K_new.stream.rank, num_consumers=B)
        K_new_bat = [_BranchRef(_parallelize26, _i) for _i in range(B)]
        _parallelize27 = Parallelize(graph, V_new, parallelize_rank=V_new.stream.rank, num_consumers=B)
        V_new_bat = [_BranchRef(_parallelize27, _i) for _i in range(B)]
        _parallelize28 = Parallelize(graph, ntl, parallelize_rank=ntl.stream.rank, num_consumers=B)
        ntl_bat = [_BranchRef(_parallelize28, _i) for _i in range(B)]
        attn_out_list = []
        for b in range(B):
            Q_kv = Q_batched[b]
            K_b = K_new_bat[b]
            V_b = V_new_bat[b]
            nb = ntl_bat[b]
            one = UnaryMap(graph, nb, fn=map_fn.ToConstInt(1), write_back_mu=False, compute_bw=1)
            sl = BinaryMap(graph, nb, one, fn=map_fn.CacheWriteAddrGen(row_offset=1), write_back_mu=False, compute_bw=1)
            bidx2 = UnaryMap(graph, nb, fn=map_fn.ToConstInt(b), write_back_mu=False, compute_bw=1)
            raddr = CacheReadAddrGen(graph, bidx2, sl, MAX_N)
            k_full = RandomOffChipLoad(graph, underlying=k_cache, raddr=raddr, tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
            v_full = RandomOffChipLoad(graph, underlying=v_cache, raddr=raddr, tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
            last_sel = FilterLastTile(graph, sl)
            _flat_partition29 = FlatPartition(graph, k_full, control=last_sel, partition_rank=0, switch_cycles=[1] * 2, write_back_mu=False, num_consumers=2)
            k_last = _BranchRef(_flat_partition29, 0)
            k_notlast = _BranchRef(_flat_partition29, 1)
            _flat_partition30 = FlatPartition(graph, v_full, control=last_sel, partition_rank=0, switch_cycles=[1] * 2, write_back_mu=False, num_consumers=2)
            v_last = _BranchRef(_flat_partition30, 0)
            v_notlast = _BranchRef(_flat_partition30, 1)
            offset_val = UnaryMap(graph, nb, fn=map_fn.ToConstInt(0), write_back_mu=False, compute_bw=1)
            k_last_off = BinaryMap(graph, k_last, offset_val, fn=map_fn.SetOffset(), write_back_mu=False, compute_bw=1)
            k_updated = BinaryMap(graph, k_last_off, K_b, fn=map_fn.RowWiseAppend(), write_back_mu=False, compute_bw=1)
            offset_val_v = UnaryMap(graph, nb, fn=map_fn.ToConstInt(0), write_back_mu=False, compute_bw=1)
            v_last_off = BinaryMap(graph, v_last, offset_val_v, fn=map_fn.SetOffset(), write_back_mu=False, compute_bw=1)
            v_updated = BinaryMap(graph, v_last_off, V_b, fn=map_fn.RowWiseAppend(), write_back_mu=False, compute_bw=1)
            k_seq_raw = FlatReassemble(graph, inputs=[k_updated, k_notlast], control=last_sel, reassemble_rank=0, switch_cycles=[1] * len([k_updated, k_notlast]), write_back_mu=False)
            v_seq_raw = FlatReassemble(graph, inputs=[v_updated, v_notlast], control=last_sel, reassemble_rank=0, switch_cycles=[1] * len([v_updated, v_notlast]), write_back_mu=False)
            k_seq = Flatten(graph, k_seq_raw, min_rank=0, max_rank=1)
            v_seq = Flatten(graph, v_seq_raw, min_rank=0, max_rank=1)
            Q_kv_flat = Flatten(graph, Q_kv, min_rank=0, max_rank=1)
            _parallelize31 = Parallelize(graph, Q_kv_flat, parallelize_rank=Q_kv_flat.stream.rank, num_consumers=NUM_KV_HEADS)
            Q_per_head = [_BranchRef(_parallelize31, _i) for _i in range(NUM_KV_HEADS)]
            k_rows = RetileStreamify(graph, k_seq, split_row=True, chunk=1)
            k_rows_3d = ReshapePadStream(graph, k_rows, chunk_size=NUM_KV_HEADS, reshape_rank=0, write_back_mu=False, have_pad_stream=False, pad_fn=None)
            k_buf = Bufferize(graph, k_rows_3d, rank=1)
            v_rows = RetileStreamify(graph, v_seq, split_row=True, chunk=1)
            v_rows_3d = ReshapePadStream(graph, v_rows, chunk_size=NUM_KV_HEADS, reshape_rank=0, write_back_mu=False, have_pad_stream=False, pad_fn=None)
            v_buf = Bufferize(graph, v_rows_3d, rank=1)
            group_outputs = []
            for h in range(NUM_KV_HEADS):
                Q_h = Q_per_head[h]
                if h == 0:
                    k_h_st = Streamify(graph, k_buf, stride=tuple((1,)), out_shape_tiled=tuple((1,)))
                    k_row_h = Accum(graph, k_h_st, output_stream_dtype=_dsl2step_out_tile(k_h_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(k_h_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                else:
                    k_h_st = Streamify(graph, k_buf, stride=tuple((1,)), out_shape_tiled=tuple((h + 1,)))
                    k_h_acc = Accum(graph, k_h_st, output_stream_dtype=_dsl2step_out_tile(k_h_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(k_h_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                    k_hm1_st = Streamify(graph, k_buf, stride=tuple((1,)), out_shape_tiled=tuple((h,)))
                    k_hm1_acc = Accum(graph, k_hm1_st, output_stream_dtype=_dsl2step_out_tile(k_hm1_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(k_hm1_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                    _tmp11 = UnaryMap(graph, k_hm1_acc, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=CBW)
                    k_row_h = BinaryMap(graph, k_h_acc, _tmp11, fn=map_fn.Add(), write_back_mu=False, compute_bw=CBW)
                if h == 0:
                    v_h_st = Streamify(graph, v_buf, stride=tuple((1,)), out_shape_tiled=tuple((1,)))
                    v_row_h = Accum(graph, v_h_st, output_stream_dtype=_dsl2step_out_tile(v_h_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(v_h_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                else:
                    v_h_st = Streamify(graph, v_buf, stride=tuple((1,)), out_shape_tiled=tuple((h + 1,)))
                    v_h_acc = Accum(graph, v_h_st, output_stream_dtype=_dsl2step_out_tile(v_h_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(v_h_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                    v_hm1_st = Streamify(graph, v_buf, stride=tuple((1,)), out_shape_tiled=tuple((h,)))
                    v_hm1_acc = Accum(graph, v_hm1_st, output_stream_dtype=_dsl2step_out_tile(v_hm1_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(v_hm1_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                    _tmp12 = UnaryMap(graph, v_hm1_acc, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=CBW)
                    v_row_h = BinaryMap(graph, v_h_acc, _tmp12, fn=map_fn.Add(), write_back_mu=False, compute_bw=CBW)
                Q_h_2d = Reshape(graph, Q_h, chunk_size=1, reshape_rank=0, write_back_mu=False)
                Q_h_exp = ExpandRef(graph, Q_h_2d, ref=k_row_h, expand_rank=1)
                scores = BinaryMap(graph, Q_h_exp, k_row_h, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=CBW)
                row_max = Accum(graph, scores, output_stream_dtype=_dsl2step_out_tile(scores, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(scores, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                rm_2d = Reshape(graph, row_max, chunk_size=1, reshape_rank=0, write_back_mu=False)
                rm_bc = ExpandRef(graph, rm_2d, ref=scores, expand_rank=1)
                _tmp13 = UnaryMap(graph, rm_bc, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=CBW)
                sh_s = BinaryMap(graph, scores, _tmp13, fn=map_fn.Add(), write_back_mu=False, compute_bw=CBW)
                exp_s = UnaryMap(graph, sh_s, fn=map_fn.Exp(), write_back_mu=False, compute_bw=CBW)
                denom = Accum(graph, exp_s, output_stream_dtype=_dsl2step_out_tile(exp_s, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(exp_s, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                context_raw = BinaryMap(graph, exp_s, v_row_h, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=CBW)
                context = Accum(graph, context_raw, output_stream_dtype=_dsl2step_out_tile(context_raw, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(context_raw, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
                d_rep = RepeatStatic(graph, denom, repeat_factor=HEAD_DIM)
                d_tiled = Accum(graph, d_rep, output_stream_dtype=_dsl2step_out_tile(d_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(d_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
                out_h = BinaryMap(graph, context, d_tiled, fn=map_fn.Div(), write_back_mu=False, compute_bw=CBW)
                group_outputs.append(out_h)
            batch_out = StaticReassemble(graph, inputs=group_outputs, merge_rank=_dsl2step_stream(group_outputs[0]).rank)
            batch_out2 = PromoteOuter(graph, batch_out)
            attn_out_list.append(batch_out2)
        all_attn = StaticReassemble(graph, inputs=attn_out_list, merge_rank=_dsl2step_stream(attn_out_list[0]).rank)
        attn_hd = Accum(graph, all_attn, output_stream_dtype=_dsl2step_out_tile(all_attn, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(all_attn, 'row'), accum_rank=1, write_back_mu=False, compute_bw=1)
        attn_rows = RetileStreamify(graph, attn_hd, split_row=True, chunk=1)
        attn_2d = Reshape(graph, attn_rows, chunk_size=NUM_HEADS, reshape_rank=0, write_back_mu=False)
        attn_512 = Accum(graph, attn_2d, output_stream_dtype=_dsl2step_out_tile(attn_2d, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn_2d, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
        ow = LinearOffChipLoadRef(graph, ref=attn_512, underlying=o_proj_weight, stride=tuple((1,)), out_shape_tiled=tuple((D_CHUNKS,)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        _parallelize32 = Parallelize(graph, attn_512, parallelize_rank=attn_512.stream.rank, num_consumers=PAR_FACTOR)
        attn_512_par = [_BranchRef(_parallelize32, _i) for _i in range(PAR_FACTOR)]
        _parallelize33 = Parallelize(graph, ow, parallelize_rank=ow.stream.rank, num_consumers=PAR_FACTOR)
        ow_par = [_BranchRef(_parallelize33, _i) for _i in range(PAR_FACTOR)]
        o_parts = []
        for i in range(PAR_FACTOR):
            o_rep = RepeatRef(graph, attn_512_par[i], ref=ow_par[i])
            o_mm = BinaryMap(graph, o_rep, ow_par[i], fn=map_fn.Matmul(), write_back_mu=False, compute_bw=CBW)
            o_parts.append(o_mm)
        o_mm_full = StaticReassemble(graph, inputs=o_parts, merge_rank=_dsl2step_stream(o_parts[0]).rank)
        o_out = Accum(graph, o_mm_full, output_stream_dtype=_dsl2step_out_tile(o_mm_full, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(o_mm_full, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
        res_add_0 = BinaryMap(graph, o_out, x, fn=map_fn.Add(), write_back_mu=False, compute_bw=CBW)
        r_sq = UnaryMap(graph, res_add_0, fn=map_fn.Square(), write_back_mu=False, compute_bw=CBW)
        r_sum = UnaryMap(graph, r_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
        r_mean = UnaryMap(graph, r_sum, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=CBW)
        r_eps = UnaryMap(graph, r_mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=CBW)
        r_inv = UnaryMap(graph, r_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=CBW)
        r_irep = RepeatStatic(graph, r_inv, repeat_factor=D)
        r_itil = Accum(graph, r_irep, output_stream_dtype=_dsl2step_out_tile(r_irep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(r_irep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
        normed_2 = BinaryMap(graph, res_add_0, r_itil, fn=map_fn.Mul(), write_back_mu=False, compute_bw=CBW)
        normed_2_out = Promote(graph, normed_2, promote_rank=0)
        res_add_0_out = Promote(graph, res_add_0, promote_rank=0)
        return (normed_2_out, res_add_0_out)

    def moe_block_0(normed_2, res_add_0, expert_onehot, expert_weights, w_gate_list, w_up_list, w_down_list, *, out_shapes):
        batch = 64
        n_experts = 8
        n_active = 2
        D = 512
        F_dim = 1792
        TILE_F = 64
        F_TILES = F_dim // TILE_F
        normed_flat = Flatten(graph, normed_2, min_rank=0, max_rank=1)
        res_flat = Flatten(graph, res_add_0, min_rank=0, max_rank=1)
        normed_po = PromoteOuter(graph, normed_flat)
        normed_rep = RepeatStatic(graph, normed_po, repeat_factor=n_active)
        sel = SelectGen(is_multihot=True, tensor=expert_onehot, n=n_experts)
        graph.add_node(sel)
        _flat_partition34 = FlatPartition(graph, normed_rep, control=sel, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
        partitioned = [_BranchRef(_flat_partition34, _i) for _i in range(n_experts)]
        expert_outputs = []
        for i in range(n_experts):
            xi = partitioned[i]
            gate_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_gate_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=D, tile_col=TILE_F, par_dispatch=32, start_tile_idx=0)
            up_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_up_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=D, tile_col=TILE_F, par_dispatch=32, start_tile_idx=0)
            down_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_down_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=TILE_F, tile_col=D, par_dispatch=32, start_tile_idx=0)
            xi_exp = RepeatRef(graph, xi, ref=gate_w)
            gate_out = BinaryMap(graph, xi_exp, gate_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=32)
            up_out = BinaryMap(graph, xi_exp, up_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=32)
            _tmp14 = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=32)
            hidden = BinaryMap(graph, _tmp14, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=32)
            down_out = BinaryMapAccum(graph, hidden, down_w, fn=map_accum_fn.Matmul(), init_fn=Zero(shape=(_dsl2step_in_tile(hidden).shape[0], _dsl2step_in_tile(down_w).shape[0 if False else 1]), dtype=_dsl2step_in_tile(hidden).tile_dtype), rank=1, write_back_mu=False, compute_bw=32)
            expert_outputs.append(down_out)
        reassembled = FlatReassemble(graph, inputs=expert_outputs, control=sel, reassemble_rank=0, switch_cycles=[1] * len(expert_outputs), write_back_mu=False)
        per_slot = Accum(graph, reassembled, output_stream_dtype=_dsl2step_out_tile(reassembled, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(reassembled, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
        ew_stream = LinearOffChipLoad(expert_weights, stride=tuple((n_active, 1)), out_shape_tiled=tuple((batch, n_active)), tile_row=1, tile_col=1, par_dispatch=32, start_tile_idx=0)
        weighted = BinaryMap(graph, per_slot, ew_stream, fn=map_fn.Mul(), write_back_mu=False, compute_bw=32)
        moe_sum = Accum(graph, weighted, output_stream_dtype=_dsl2step_out_tile(weighted, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(weighted, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
        moe_flat = Flatten(graph, moe_sum, min_rank=0, max_rank=1)
        final = BinaryMap(graph, moe_flat, res_flat, fn=map_fn.Add(), write_back_mu=False, compute_bw=32)
        result = Promote(graph, final, promote_rank=0)
        return result
    normed_2, res_add_0 = attention_block_1(tensors['input_tensor'], tensors['q_proj'], tensors['k_proj'], tensors['v_proj'], tensors['cos'], tensors['sin'], tensors['k_cache'], tensors['v_cache'], tensors['num_token_list'], tensors['o_proj_weight'], out_shapes=((64, 1, 1, 512), (64, 1, 1, 512)))
    moe_out = moe_block_0(normed_2, res_add_0, tensors['expert_onehot'], tensors['expert_weights'], tensors['w_gate_list'], tensors['w_up_list'], tensors['w_down_list'], out_shapes=((64, 1, 1, 512),))
    _store35 = OffChipStore(graph, moe_out, par_dispatch=1)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store35)