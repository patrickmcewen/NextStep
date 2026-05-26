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

    def attention_block(input_tensor, q_proj, k_proj, v_proj, cos, sin, k_cache, v_cache, num_token_list, o_proj_weight, *, out_shapes):
        B = 64
        D = 512
        HEAD_DIM = 32
        HALF = 16
        NUM_HEADS = 16
        NUM_KV_HEADS = 4
        QPK = 4
        MAX_N = 4096
        D_CHUNKS = D // HEAD_DIM
        KV_CHUNKS = NUM_KV_HEADS * HEAD_DIM // HEAD_DIM
        k_cache = k_cache.flatten(0, -2)
        v_cache = v_cache.flatten(0, -2)
        x_raw = LinearOffChipLoad(input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((B,)), tile_row=1, tile_col=D, par_dispatch=8, start_tile_idx=0)
        x = Flatten(graph, x_raw, min_rank=0, max_rank=1)
        x_sq = UnaryMap(graph, x, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        x_rsum = UnaryMap(graph, x_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        x_mean = UnaryMap(graph, x_rsum, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=4096)
        x_eps = UnaryMap(graph, x_mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        x_inv = UnaryMap(graph, x_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        x_inv_rep = RepeatStatic(graph, x_inv, repeat_factor=D)
        x_inv_tiled = Accum(graph, x_inv_rep, output_stream_dtype=_dsl2step_out_tile(x_inv_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(x_inv_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        normed = BinaryMap(graph, x, x_inv_tiled, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        q_w = LinearOffChipLoadRef(graph, ref=normed, underlying=q_proj, stride=tuple((1,)), out_shape_tiled=tuple((D_CHUNKS,)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        normed_q = RepeatRef(graph, normed, ref=q_w)
        Q_full = BinaryMap(graph, normed_q, q_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        k_w = LinearOffChipLoadRef(graph, ref=normed, underlying=k_proj, stride=tuple((1,)), out_shape_tiled=tuple((KV_CHUNKS,)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        normed_k = RepeatRef(graph, normed, ref=k_w)
        K_full = BinaryMap(graph, normed_k, k_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        v_w = LinearOffChipLoadRef(graph, ref=normed, underlying=v_proj, stride=tuple((1,)), out_shape_tiled=tuple((KV_CHUNKS,)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        normed_v = RepeatRef(graph, normed, ref=v_w)
        V_full = BinaryMap(graph, normed_v, v_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)

        def per_head_rms(h_stream, dim):
            sq = UnaryMap(graph, h_stream, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
            rs = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
            m = UnaryMap(graph, rs, fn=map_fn.MulImmediate(1.0 / dim), write_back_mu=False, compute_bw=4096)
            e = UnaryMap(graph, m, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
            inv = UnaryMap(graph, e, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
            irep = RepeatStatic(graph, inv, repeat_factor=dim)
            itil = Accum(graph, irep, output_stream_dtype=_dsl2step_out_tile(irep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(irep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            _tmp1 = BinaryMap(graph, h_stream, itil, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            return _tmp1
        Q_normed = per_head_rms(Q_full, HEAD_DIM)
        K_normed = per_head_rms(K_full, HEAD_DIM)
        cos_raw = LinearOffChipLoad(cos, stride=tuple((1, 0)), out_shape_tiled=tuple((B, 1)), tile_row=1, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        sin_raw = LinearOffChipLoad(sin, stride=tuple((1, 0)), out_shape_tiled=tuple((B, 1)), tile_row=1, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        cos_b = Flatten(graph, cos_raw, min_rank=0, max_rank=2)
        sin_b = Flatten(graph, sin_raw, min_rank=0, max_rank=2)

        def apply_rope(x_heads, n_heads):
            cos_bc = RepeatStatic(graph, cos_b, repeat_factor=n_heads)
            sin_bc = RepeatStatic(graph, sin_b, repeat_factor=n_heads)
            x_sp = RetileStreamify(graph, x_heads, split_row=False, chunk=HALF)
            c_sp = RetileStreamify(graph, cos_bc, split_row=False, chunk=HALF)
            s_sp = RetileStreamify(graph, sin_bc, split_row=False, chunk=HALF)
            x_f = Flatten(graph, x_sp, min_rank=0, max_rank=1)
            c_f = Flatten(graph, c_sp, min_rank=0, max_rank=1)
            s_f = Flatten(graph, s_sp, min_rank=0, max_rank=1)
            _parallelize13 = Parallelize(graph, x_f, parallelize_rank=x_f.stream.rank, num_consumers=2)
            xh = [_BranchRef(_parallelize13, _i) for _i in range(2)]
            _parallelize14 = Parallelize(graph, c_f, parallelize_rank=c_f.stream.rank, num_consumers=2)
            ch = [_BranchRef(_parallelize14, _i) for _i in range(2)]
            _parallelize15 = Parallelize(graph, s_f, parallelize_rank=s_f.stream.rank, num_consumers=2)
            sh = [_BranchRef(_parallelize15, _i) for _i in range(2)]
            x1, x2 = (xh[0], xh[1])
            c1, c2 = (ch[0], ch[1])
            s1, s2 = (sh[0], sh[1])
            nx2 = UnaryMap(graph, x2, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
            _tmp2 = BinaryMap(graph, x1, c1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            _tmp3 = BinaryMap(graph, nx2, s1, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            r1 = BinaryMap(graph, _tmp2, _tmp3, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
            _tmp4 = BinaryMap(graph, x2, c2, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            _tmp5 = BinaryMap(graph, x1, s2, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            r2 = BinaryMap(graph, _tmp4, _tmp5, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
            il = StaticReassemble(graph, inputs=[r1, r2], merge_rank=_dsl2step_stream([r1, r2][0]).rank)
            il2 = Reshape(graph, il, chunk_size=2, reshape_rank=0, write_back_mu=False)
            hd = Accum(graph, il2, output_stream_dtype=_dsl2step_out_tile(il2, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(il2, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            _tmp6 = Reshape(graph, hd, chunk_size=n_heads, reshape_rank=0, write_back_mu=False)
            return _tmp6
        Q_rope = apply_rope(Q_normed, NUM_HEADS)
        K_rope = apply_rope(K_normed, NUM_KV_HEADS)
        K_new = Accum(graph, K_rope, output_stream_dtype=_dsl2step_out_tile(K_rope, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(K_rope, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        V_new = Accum(graph, V_full, output_stream_dtype=_dsl2step_out_tile(V_full, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(V_full, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        ntl_meta = MetadataGen(tensor=num_token_list)
        graph.add_node(ntl_meta)
        ntl = Flatten(graph, ntl_meta, min_rank=0, max_rank=1)
        _parallelize16 = Parallelize(graph, K_new, parallelize_rank=K_new.stream.rank, num_consumers=B)
        K_bwr = [_BranchRef(_parallelize16, _i) for _i in range(B)]
        _parallelize17 = Parallelize(graph, V_new, parallelize_rank=V_new.stream.rank, num_consumers=B)
        V_bwr = [_BranchRef(_parallelize17, _i) for _i in range(B)]
        _parallelize18 = Parallelize(graph, ntl, parallelize_rank=ntl.stream.rank, num_consumers=B)
        ntl_wr = [_BranchRef(_parallelize18, _i) for _i in range(B)]
        for b in range(B):
            nb = ntl_wr[b]
            bidx = UnaryMap(graph, nb, fn=map_fn.ToConstInt(b), write_back_mu=False, compute_bw=4096)
            kwa = BinaryMap(graph, bidx, nb, fn=map_fn.CacheWriteAddrGen(row_offset=MAX_N), write_back_mu=False, compute_bw=4096)
            _tmp7 = RandomOffChipStore(graph, underlying=k_cache, wdata=K_bwr[b], waddr=kwa, tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
            vwa = BinaryMap(graph, bidx, nb, fn=map_fn.CacheWriteAddrGen(row_offset=MAX_N), write_back_mu=False, compute_bw=4096)
            _tmp8 = RandomOffChipStore(graph, underlying=v_cache, wdata=V_bwr[b], waddr=vwa, tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
        Q_for_attn = Accum(graph, Q_rope, output_stream_dtype=_dsl2step_out_tile(Q_rope, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(Q_rope, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Q_groups_flat = RetileStreamify(graph, Q_for_attn, split_row=True, chunk=QPK)
        Q_groups = Reshape(graph, Q_groups_flat, chunk_size=NUM_KV_HEADS, reshape_rank=0, write_back_mu=False)
        _parallelize19 = Parallelize(graph, Q_groups, parallelize_rank=Q_groups.stream.rank, num_consumers=B)
        Q_batched = [_BranchRef(_parallelize19, _i) for _i in range(B)]
        _parallelize20 = Parallelize(graph, K_new, parallelize_rank=K_new.stream.rank, num_consumers=B)
        K_new_bat = [_BranchRef(_parallelize20, _i) for _i in range(B)]
        _parallelize21 = Parallelize(graph, V_new, parallelize_rank=V_new.stream.rank, num_consumers=B)
        V_new_bat = [_BranchRef(_parallelize21, _i) for _i in range(B)]
        _parallelize22 = Parallelize(graph, ntl, parallelize_rank=ntl.stream.rank, num_consumers=B)
        ntl_bat = [_BranchRef(_parallelize22, _i) for _i in range(B)]
        attn_out_list = []
        for b in range(B):
            Q_kv = Q_batched[b]
            K_b = K_new_bat[b]
            V_b = V_new_bat[b]
            nb = ntl_bat[b]
            one = UnaryMap(graph, nb, fn=map_fn.ToConstInt(1), write_back_mu=False, compute_bw=4096)
            sl = BinaryMap(graph, nb, one, fn=map_fn.CacheWriteAddrGen(row_offset=1), write_back_mu=False, compute_bw=4096)
            bidx2 = UnaryMap(graph, nb, fn=map_fn.ToConstInt(b), write_back_mu=False, compute_bw=4096)
            raddr = CacheReadAddrGen(graph, bidx2, sl, MAX_N)
            k_full = RandomOffChipLoad(graph, underlying=k_cache, raddr=raddr, tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
            v_full = RandomOffChipLoad(graph, underlying=v_cache, raddr=raddr, tile_row=NUM_KV_HEADS, tile_col=HEAD_DIM, base_addr_byte=0, par_dispatch=8)
            last_sel = FilterLastTile(graph, sl)
            _flat_partition23 = FlatPartition(graph, k_full, control=last_sel, partition_rank=0, switch_cycles=[1] * 2, write_back_mu=False, num_consumers=2)
            k_last = _BranchRef(_flat_partition23, 0)
            k_notlast = _BranchRef(_flat_partition23, 1)
            _flat_partition24 = FlatPartition(graph, v_full, control=last_sel, partition_rank=0, switch_cycles=[1] * 2, write_back_mu=False, num_consumers=2)
            v_last = _BranchRef(_flat_partition24, 0)
            v_notlast = _BranchRef(_flat_partition24, 1)
            k_seq_raw = FlatReassemble(graph, inputs=[K_b, k_notlast], control=last_sel, reassemble_rank=0, switch_cycles=[1] * len([K_b, k_notlast]), write_back_mu=False)
            v_seq_raw = FlatReassemble(graph, inputs=[V_b, v_notlast], control=last_sel, reassemble_rank=0, switch_cycles=[1] * len([V_b, v_notlast]), write_back_mu=False)
            k_seq = Flatten(graph, k_seq_raw, min_rank=0, max_rank=1)
            v_seq = Flatten(graph, v_seq_raw, min_rank=0, max_rank=1)
            Q_kv_flat = Flatten(graph, Q_kv, min_rank=0, max_rank=1)
            _parallelize25 = Parallelize(graph, Q_kv_flat, parallelize_rank=Q_kv_flat.stream.rank, num_consumers=NUM_KV_HEADS)
            Q_kv_per_grp = [_BranchRef(_parallelize25, _i) for _i in range(NUM_KV_HEADS)]
            group_outputs = []
            for h in range(NUM_KV_HEADS):
                Q_h = Q_kv_per_grp[h]
                Q_h_2d = Reshape(graph, Q_h, chunk_size=1, reshape_rank=0, write_back_mu=False)
                Q_h_exp = ExpandRef(graph, Q_h_2d, ref=k_seq, expand_rank=1)
                scores = BinaryMap(graph, Q_h_exp, k_seq, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=4096)
                sc_cols = RetileStreamify(graph, scores, split_row=False, chunk=1)
                sc_3d = ReshapePadStream(graph, sc_cols, chunk_size=NUM_KV_HEADS, reshape_rank=0, write_back_mu=False, have_pad_stream=False, pad_fn=None)
                sc_buf = Bufferize(graph, sc_3d, rank=1)
                sum_h_st = Streamify(graph, sc_buf, stride=tuple((1,)), out_shape_tiled=tuple((h + 1,)))
                sum_h = Accum(graph, sum_h_st, output_stream_dtype=_dsl2step_out_tile(sum_h_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(sum_h_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                if h == 0:
                    col_h = sum_h
                else:
                    sc_buf_hm1 = Bufferize(graph, sc_3d, rank=1)
                    sum_hm1_st = Streamify(graph, sc_buf_hm1, stride=tuple((1,)), out_shape_tiled=tuple((h,)))
                    sum_hm1 = Accum(graph, sum_hm1_st, output_stream_dtype=_dsl2step_out_tile(sum_hm1_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(sum_hm1_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                    _tmp9 = UnaryMap(graph, sum_hm1, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
                    col_h = BinaryMap(graph, sum_h, _tmp9, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
                row_max = Accum(graph, col_h, output_stream_dtype=_dsl2step_out_tile(col_h, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(col_h, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                rm_2d = Reshape(graph, row_max, chunk_size=1, reshape_rank=0, write_back_mu=False)
                rm_bc = ExpandRef(graph, rm_2d, ref=col_h, expand_rank=1)
                _tmp10 = UnaryMap(graph, rm_bc, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
                sh = BinaryMap(graph, col_h, _tmp10, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
                exp_s = UnaryMap(graph, sh, fn=map_fn.Exp(), write_back_mu=False, compute_bw=4096)
                denom = Accum(graph, exp_s, output_stream_dtype=_dsl2step_out_tile(exp_s, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(exp_s, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                v_rows = RetileStreamify(graph, v_seq, split_row=True, chunk=1)
                v_3d = ReshapePadStream(graph, v_rows, chunk_size=NUM_KV_HEADS, reshape_rank=0, write_back_mu=False, have_pad_stream=False, pad_fn=None)
                v_buf = Bufferize(graph, v_3d, rank=1)
                vsum_h_st = Streamify(graph, v_buf, stride=tuple((1,)), out_shape_tiled=tuple((h + 1,)))
                vsum_h = Accum(graph, vsum_h_st, output_stream_dtype=_dsl2step_out_tile(vsum_h_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(vsum_h_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                if h == 0:
                    v_row_h = vsum_h
                else:
                    v_buf_hm1 = Bufferize(graph, v_3d, rank=1)
                    vsum_hm1_st = Streamify(graph, v_buf_hm1, stride=tuple((1,)), out_shape_tiled=tuple((h,)))
                    vsum_hm1 = Accum(graph, vsum_hm1_st, output_stream_dtype=_dsl2step_out_tile(vsum_hm1_st, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(vsum_hm1_st, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                    _tmp11 = UnaryMap(graph, vsum_hm1, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
                    v_row_h = BinaryMap(graph, vsum_h, _tmp11, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
                wtd_v = BinaryMap(graph, exp_s, v_row_h, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
                context = Accum(graph, wtd_v, output_stream_dtype=_dsl2step_out_tile(wtd_v, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(wtd_v, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                d_rep = RepeatStatic(graph, denom, repeat_factor=HEAD_DIM)
                d_tiled = Accum(graph, d_rep, output_stream_dtype=_dsl2step_out_tile(d_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(d_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
                out_h = BinaryMap(graph, context, d_tiled, fn=map_fn.Div(), write_back_mu=False, compute_bw=4096)
                group_outputs.append(out_h)
            batch_out = StaticReassemble(graph, inputs=group_outputs, merge_rank=_dsl2step_stream(group_outputs[0]).rank)
            attn_out_list.append(batch_out)
        _eager_merge26 = EagerMerge(graph, attn_out_list, input_rank=1)
        all_attn = _BranchRef(_eager_merge26, 0)
        _ = _BranchRef(_eager_merge26, 1)
        all_2d = Reshape(graph, all_attn, chunk_size=NUM_KV_HEADS, reshape_rank=0, write_back_mu=False)
        attn_hd = Accum(graph, all_2d, output_stream_dtype=_dsl2step_out_tile(all_2d, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(all_2d, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        attn_rows = RetileStreamify(graph, attn_hd, split_row=True, chunk=1)
        attn_2d = Reshape(graph, attn_rows, chunk_size=NUM_HEADS, reshape_rank=0, write_back_mu=False)
        attn_512 = Accum(graph, attn_2d, output_stream_dtype=_dsl2step_out_tile(attn_2d, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn_2d, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        ow = LinearOffChipLoadRef(graph, ref=attn_512, underlying=o_proj_weight, stride=tuple((1,)), out_shape_tiled=tuple((D_CHUNKS,)), tile_row=D, tile_col=HEAD_DIM, par_dispatch=8, start_tile_idx=0)
        attn_rep = RepeatRef(graph, attn_512, ref=ow)
        o_mm = BinaryMap(graph, attn_rep, ow, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        o_out = Accum(graph, o_mm, output_stream_dtype=_dsl2step_out_tile(o_mm, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(o_mm, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        result = BinaryMap(graph, o_out, x, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return result

    def moe_block(res_add_0, expert_onehot, expert_weights, w_gate_list, w_up_list, w_down_list, *, out_shapes):
        batch = 64
        n_experts = 8
        n_active = 2
        D = 512
        F_dim = 1792
        TILE_F = 32
        F_TILES = F_dim // TILE_F
        res_flat = res_add_0
        x_sq = UnaryMap(graph, res_flat, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        x_sum = UnaryMap(graph, x_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        x_mean = UnaryMap(graph, x_sum, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=4096)
        x_eps = UnaryMap(graph, x_mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        x_rsqrt = UnaryMap(graph, x_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        x_rsqrt_rep = RepeatStatic(graph, x_rsqrt, repeat_factor=D)
        x_rsqrt_tiled = Accum(graph, x_rsqrt_rep, output_stream_dtype=_dsl2step_out_tile(x_rsqrt_rep, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(x_rsqrt_rep, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        normed = BinaryMap(graph, res_flat, x_rsqrt_tiled, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        normed_rep_inner = RepeatStatic(graph, normed, repeat_factor=n_active)
        normed_rep = PromoteOuter(graph, normed_rep_inner)
        sel = SelectGen(is_multihot=True, tensor=expert_onehot, n=n_experts)
        graph.add_node(sel)
        _flat_partition27 = FlatPartition(graph, normed_rep, control=sel, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
        partitioned = [_BranchRef(_flat_partition27, _i) for _i in range(n_experts)]
        expert_outputs = []
        for i in range(n_experts):
            xi = partitioned[i]
            gate_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_gate_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=D, tile_col=TILE_F, par_dispatch=8, start_tile_idx=0)
            up_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_up_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=D, tile_col=TILE_F, par_dispatch=8, start_tile_idx=0)
            xi_exp = RepeatRef(graph, xi, ref=gate_w)
            gate_out = BinaryMap(graph, xi_exp, gate_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            up_out = BinaryMap(graph, xi_exp, up_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            _tmp12 = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
            hidden = BinaryMap(graph, _tmp12, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            down_w = LinearOffChipLoadRef(graph, ref=xi, underlying=w_down_list[i], stride=tuple((1,)), out_shape_tiled=tuple((F_TILES,)), tile_row=TILE_F, tile_col=D, par_dispatch=8, start_tile_idx=0)
            down_out = BinaryMapAccum(graph, hidden, down_w, fn=map_accum_fn.Matmul(), init_fn=Zero(shape=(_dsl2step_in_tile(hidden).shape[0], _dsl2step_in_tile(down_w).shape[0 if False else 1]), dtype=_dsl2step_in_tile(hidden).tile_dtype), rank=1, write_back_mu=False, compute_bw=4096)
            expert_outputs.append(down_out)
        reassembled = FlatReassemble(graph, inputs=expert_outputs, control=sel, reassemble_rank=0, switch_cycles=[1] * len(expert_outputs), write_back_mu=False)
        per_slot = Accum(graph, reassembled, output_stream_dtype=_dsl2step_out_tile(reassembled, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(reassembled, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        ew_stream = LinearOffChipLoad(expert_weights, stride=tuple((n_active, 1)), out_shape_tiled=tuple((batch, n_active)), tile_row=1, tile_col=1, par_dispatch=8, start_tile_idx=0)
        weighted = BinaryMap(graph, per_slot, ew_stream, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        moe_sum = Accum(graph, weighted, output_stream_dtype=_dsl2step_out_tile(weighted, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(weighted, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        moe_flat = Flatten(graph, moe_sum, min_rank=0, max_rank=1)
        final = BinaryMap(graph, moe_flat, res_flat, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        result = Promote(graph, final, promote_rank=0)
        return result
    res_add_0 = attention_block(tensors['input_tensor'], tensors['q_proj'], tensors['k_proj'], tensors['v_proj'], tensors['cos'], tensors['sin'], tensors['k_cache'], tensors['v_cache'], tensors['num_token_list'], tensors['o_proj_weight'], out_shapes=((64, 1, 1, 512),))
    moe_out = moe_block(res_add_0, tensors['expert_onehot'], tensors['expert_weights'], tensors['w_gate_list'], tensors['w_up_list'], tensors['w_down_list'], out_shapes=((64, 1, 1, 512),))
    _store28 = OffChipStore(graph, moe_out, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store28)