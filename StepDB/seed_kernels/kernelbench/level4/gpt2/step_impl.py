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

    def qkv_projection(h, block_c_attn_w_q, block_c_attn_w_k, block_c_attn_w_v, block_c_attn_b_q, block_c_attn_b_k, block_c_attn_b_v, n_head, *, out_shapes):
        B = h.shape[1]
        T = h.shape[2]
        D = h.shape[-1]
        D_head = D // n_head

        def load_and_repeat(raw_tensor, tile_row, tile_col):
            t = LinearOffChipLoad(raw_tensor, stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=tile_row, tile_col=tile_col, par_dispatch=1, start_tile_idx=0)
            t = RepeatStatic(graph, t, repeat_factor=B)
            t = RepeatStatic(graph, t, repeat_factor=T)
            t = Flatten(graph, t, min_rank=2, max_rank=3)
            return t
        w_q = load_and_repeat(block_c_attn_w_q, tile_row=D, tile_col=D)
        b_q = load_and_repeat(block_c_attn_b_q, tile_row=1, tile_col=D)
        w_k = load_and_repeat(block_c_attn_w_k, tile_row=D, tile_col=D)
        b_k = load_and_repeat(block_c_attn_b_k, tile_row=1, tile_col=D)
        w_v = load_and_repeat(block_c_attn_w_v, tile_row=D, tile_col=D)
        b_v = load_and_repeat(block_c_attn_b_v, tile_row=1, tile_col=D)
        q = BinaryMap(graph, h, w_q, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1)
        q = BinaryMap(graph, q, b_q, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        k = BinaryMap(graph, h, w_k, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1)
        k = BinaryMap(graph, k, b_k, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        v = BinaryMap(graph, h, w_v, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1)
        v = BinaryMap(graph, v, b_v, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)

        def to_multi_head(x):
            x = Promote(graph, x, promote_rank=0)
            x = RetileStreamify(graph, x, split_row=False, chunk=D_head)
            x = Bufferize(graph, x, rank=2)
            x = Streamify(graph, x, stride=tuple((1, n_head)), out_shape_tiled=tuple((n_head, T)))
            return x
        q_out = to_multi_head(q)
        k_out = to_multi_head(k)
        v_out = to_multi_head(v)
        return (q_out, k_out, v_out)

    def attention(q, k, v, causal_mask, *, out_shapes):
        B = q.shape[1]
        H = q.shape[2]
        S = q.shape[3]
        mask = LinearOffChipLoad(causal_mask, stride=tuple((0, 0, S, 1)), out_shape_tiled=tuple((B, H, S, S)), tile_row=1, tile_col=1, par_dispatch=1, start_tile_idx=0)
        q_buf = Bufferize(graph, q, rank=1)
        q_rep = Streamify(graph, q_buf, stride=tuple((1, 0)), out_shape_tiled=tuple((S, S)))
        k_buf = Bufferize(graph, k, rank=1)
        k_rep = Streamify(graph, k_buf, stride=tuple((0, 1)), out_shape_tiled=tuple((S, S)))
        v_buf = Bufferize(graph, v, rank=1)
        v_rep = Streamify(graph, v_buf, stride=tuple((0, 1)), out_shape_tiled=tuple((S, S)))
        scores = BinaryMap(graph, q_rep, k_rep, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=1)
        scores = UnaryMap(graph, scores, fn=map_fn.MulImmediate(0.25), write_back_mu=False, compute_bw=1)
        scores = BinaryMap(graph, scores, mask, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        exp_scores = UnaryMap(graph, scores, fn=map_fn.Exp(), write_back_mu=False, compute_bw=1)
        sum_exp = Accum(graph, exp_scores, output_stream_dtype=_dsl2step_out_tile(exp_scores, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(exp_scores, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
        sum_exp = RepeatRef(graph, sum_exp, ref=exp_scores)
        attn = BinaryMap(graph, exp_scores, sum_exp, fn=map_fn.Div(), write_back_mu=False, compute_bw=1)
        weighted = BinaryMap(graph, attn, v_rep, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1)
        ctx = Accum(graph, weighted, output_stream_dtype=_dsl2step_out_tile(weighted, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(weighted, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
        return ctx

    def attention_subroutine(h, n_head, block_c_attn_w_q, block_c_attn_w_k, block_c_attn_w_v, block_c_attn_b_q, block_c_attn_b_k, block_c_attn_b_v, causal_mask, *, out_shapes):
        B = h.shape[1]
        T = h.shape[2]
        D = h.shape[4]
        D_head = D // n_head
        out_shape = (1, B, n_head, T, 1, D_head)
        q, k, v = qkv_projection(h, block_c_attn_w_q, block_c_attn_w_k, block_c_attn_w_v, block_c_attn_b_q, block_c_attn_b_k, block_c_attn_b_v, n_head, out_shapes=(out_shape, out_shape, out_shape))
        ctx = attention(q, k, v, causal_mask, out_shapes=(out_shape,))
        ctx_buf = Bufferize(graph, ctx, rank=2)
        ctx_perm = Streamify(graph, ctx_buf, stride=tuple((1, T)), out_shape_tiled=tuple((T, n_head)))
        ctx_out = Accum(graph, ctx_perm, output_stream_dtype=_dsl2step_out_tile(ctx_perm, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(ctx_perm, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
        return ctx_out

    def attention_block(n_head, x, causal_mask, block_ln1_w, block_ln1_b, block_c_attn_w_q, block_c_attn_w_k, block_c_attn_w_v, block_c_attn_b_q, block_c_attn_b_k, block_c_attn_b_v, block_c_proj_w, block_c_proj_b, *, out_shapes):
        D = x.shape[-1]
        ln_w = LinearOffChipLoad(block_ln1_w, stride=tuple((0, 0)), out_shape_tiled=tuple((2, 32)), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
        ln_b = LinearOffChipLoad(block_ln1_b, stride=tuple((0, 0)), out_shape_tiled=tuple((2, 32)), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
        sum_x = UnaryMap(graph, x, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
        mean = UnaryMap(graph, sum_x, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=1)
        neg_mean = UnaryMap(graph, mean, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=1)
        x_centered = BinaryMap(graph, x, neg_mean, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        sq = UnaryMap(graph, x_centered, fn=map_fn.Square(), write_back_mu=False, compute_bw=1)
        sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
        var = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=1)
        eps = 1e-05
        var_eps = UnaryMap(graph, var, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=1)
        inv_sqrt = UnaryMap(graph, var_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=1)
        norm = BinaryMap(graph, x_centered, inv_sqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
        scaled = BinaryMap(graph, norm, ln_w, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
        h = BinaryMap(graph, scaled, ln_b, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        ctx = attention_subroutine(h, n_head, block_c_attn_w_q, block_c_attn_w_k, block_c_attn_w_v, block_c_attn_b_q, block_c_attn_b_k, block_c_attn_b_v, causal_mask, out_shapes=((1, 2, 32, 1, 64),))
        proj_w = LinearOffChipLoad(block_c_proj_w, stride=tuple((0, 0)), out_shape_tiled=tuple((2, 32)), tile_row=D, tile_col=D, par_dispatch=1, start_tile_idx=0)
        proj_b = LinearOffChipLoad(block_c_proj_b, stride=tuple((0, 0)), out_shape_tiled=tuple((2, 32)), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
        _tmp1 = BinaryMap(graph, ctx, proj_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1)
        proj = BinaryMap(graph, _tmp1, proj_b, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        out = BinaryMap(graph, x, proj, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        return out

    def mlp_block(x, block_ln2_w, block_ln2_b, block_mlp_fc_w, block_mlp_fc_b, block_mlp_proj_w, block_mlp_proj_b, *, out_shapes):
        stream_shape = tuple(x.shape[1:-2])
        zero_stride = tuple((0 for _ in stream_shape))
        D = x.shape[-1]
        I = block_mlp_fc_w.shape[-1]
        ln_w = LinearOffChipLoad(block_ln2_w, stride=tuple(zero_stride), out_shape_tiled=tuple(stream_shape), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
        ln_b = LinearOffChipLoad(block_ln2_b, stride=tuple(zero_stride), out_shape_tiled=tuple(stream_shape), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
        mean = UnaryMap(graph, x, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
        mean = UnaryMap(graph, mean, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=1)
        neg_mean = UnaryMap(graph, mean, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=1)
        x_centered = BinaryMap(graph, x, neg_mean, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        sq = BinaryMap(graph, x_centered, x_centered, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
        var_sum = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
        var = UnaryMap(graph, var_sum, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=1)
        eps = 1e-05
        var_eps = UnaryMap(graph, var, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=1)
        inv_std = UnaryMap(graph, var_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=1)
        x_norm = BinaryMap(graph, x_centered, inv_std, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
        x_norm = BinaryMap(graph, x_norm, ln_w, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
        x_norm = BinaryMap(graph, x_norm, ln_b, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        fc_w = LinearOffChipLoad(block_mlp_fc_w, stride=tuple(zero_stride), out_shape_tiled=tuple(stream_shape), tile_row=D, tile_col=I, par_dispatch=1, start_tile_idx=0)
        fc_b = LinearOffChipLoad(block_mlp_fc_b, stride=tuple(zero_stride), out_shape_tiled=tuple(stream_shape), tile_row=1, tile_col=I, par_dispatch=1, start_tile_idx=0)
        h = BinaryMap(graph, x_norm, fc_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1)
        h = BinaryMap(graph, h, fc_b, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        h_sq = BinaryMap(graph, h, h, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
        h_cu = BinaryMap(graph, h_sq, h, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
        term = UnaryMap(graph, h_cu, fn=map_fn.MulImmediate(0.044715), write_back_mu=False, compute_bw=1)
        term_plus_h = BinaryMap(graph, term, h, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        sqrt_2_over_pi = 0.7978845608028654
        scaled = UnaryMap(graph, term_plus_h, fn=map_fn.MulImmediate(sqrt_2_over_pi), write_back_mu=False, compute_bw=1)
        tanh_out = UnaryMap(graph, scaled, fn=map_fn.Tanh(), write_back_mu=False, compute_bw=1)
        add_one = UnaryMap(graph, tanh_out, fn=map_fn.AddImmediate(1.0), write_back_mu=False, compute_bw=1)
        half = UnaryMap(graph, add_one, fn=map_fn.MulImmediate(0.5), write_back_mu=False, compute_bw=1)
        gelu = BinaryMap(graph, h, half, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
        proj_w = LinearOffChipLoad(block_mlp_proj_w, stride=tuple(zero_stride), out_shape_tiled=tuple(stream_shape), tile_row=I, tile_col=D, par_dispatch=1, start_tile_idx=0)
        proj_b = LinearOffChipLoad(block_mlp_proj_b, stride=tuple(zero_stride), out_shape_tiled=tuple(stream_shape), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
        out_linear = BinaryMap(graph, gelu, proj_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1)
        out_linear = BinaryMap(graph, out_linear, proj_b, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        out = BinaryMap(graph, out_linear, x, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
        return out

    def gpt2_layer(n_head, x, causal_mask, block_ln1_w, block_ln1_b, block_c_attn_w_q, block_c_attn_w_k, block_c_attn_w_v, block_c_attn_b_q, block_c_attn_b_k, block_c_attn_b_v, block_c_proj_w, block_c_proj_b, block_ln2_w, block_ln2_b, block_mlp_fc_w, block_mlp_fc_b, block_mlp_proj_w, block_mlp_proj_b, *, out_shapes):
        attn_out = attention_block(n_head, x, causal_mask, block_ln1_w, block_ln1_b, block_c_attn_w_q, block_c_attn_w_k, block_c_attn_w_v, block_c_attn_b_q, block_c_attn_b_k, block_c_attn_b_v, block_c_proj_w, block_c_proj_b, out_shapes=out_shapes)
        mlp_out = mlp_block(attn_out, block_ln2_w, block_ln2_b, block_mlp_fc_w, block_mlp_fc_b, block_mlp_proj_w, block_mlp_proj_b, out_shapes=out_shapes)
        return mlp_out
    B = dims['B']
    T = dims['T']
    D = dims['D']
    V = dims['V']
    H = dims['H']
    L = dims['L']
    token_addr = MetadataGen(tensor=tensors['input_ids'])
    graph.add_node(token_addr)
    token_emb = RandomOffChipLoad(graph, underlying=tensors['wte'], raddr=token_addr, tile_row=1, tile_col=D, base_addr_byte=0, par_dispatch=1)
    pos_emb = LinearOffChipLoad(tensors['wpe'], stride=tuple((0, 1)), out_shape_tiled=tuple((B, T)), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
    x = BinaryMap(graph, token_emb, pos_emb, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
    for layer in range(L):
        x = gpt2_layer(H, x, tensors['causal_mask'], tensors['block_ln1_w'][layer], tensors['block_ln1_b'][layer], tensors['block_c_attn_w_q'][layer], tensors['block_c_attn_w_k'][layer], tensors['block_c_attn_w_v'][layer], tensors['block_c_attn_b_q'][layer], tensors['block_c_attn_b_k'][layer], tensors['block_c_attn_b_v'][layer], tensors['block_c_proj_w'][layer], tensors['block_c_proj_b'][layer], tensors['block_ln2_w'][layer], tensors['block_ln2_b'][layer], tensors['block_mlp_fc_w'][layer], tensors['block_mlp_fc_b'][layer], tensors['block_mlp_proj_w'][layer], tensors['block_mlp_proj_b'][layer], out_shapes=((1, B, T, 1, D),))
    sum_tile = UnaryMap(graph, x, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
    mean = UnaryMap(graph, sum_tile, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=1)
    neg_mean = UnaryMap(graph, mean, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=1)
    centered = BinaryMap(graph, x, neg_mean, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
    sq = UnaryMap(graph, centered, fn=map_fn.Square(), write_back_mu=False, compute_bw=1)
    sum_sq = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=1)
    var = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(1.0 / D), write_back_mu=False, compute_bw=1)
    var_eps = UnaryMap(graph, var, fn=map_fn.AddImmediate(1e-05), write_back_mu=False, compute_bw=1)
    rsqrt = UnaryMap(graph, var_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=1)
    normalized = BinaryMap(graph, centered, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
    scale = LinearOffChipLoad(tensors['ln_f_w'], stride=tuple((0, 0)), out_shape_tiled=tuple((B, T)), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
    bias = LinearOffChipLoad(tensors['ln_f_b'], stride=tuple((0, 0)), out_shape_tiled=tuple((B, T)), tile_row=1, tile_col=D, par_dispatch=1, start_tile_idx=0)
    scaled = BinaryMap(graph, normalized, scale, fn=map_fn.Mul(), write_back_mu=False, compute_bw=1)
    x_ln = BinaryMap(graph, scaled, bias, fn=map_fn.Add(), write_back_mu=False, compute_bw=1)
    wte_stream = LinearOffChipLoad(tensors['wte'], stride=tuple((0, 0)), out_shape_tiled=tuple((B, T)), tile_row=V, tile_col=D, par_dispatch=1, start_tile_idx=0)
    logits = BinaryMap(graph, x_ln, wte_stream, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=1)
    _store2 = OffChipStore(graph, logits, par_dispatch=1)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store2)