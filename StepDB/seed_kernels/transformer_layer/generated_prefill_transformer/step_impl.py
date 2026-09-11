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

    def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, *, out_shapes):
        seq_len = input_tensor.shape[0]
        hidden_dim = input_tensor.shape[1]
        head_dim = cos.shape[-1]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim
        hidden_chunk = head_dim
        hidden_chunks = hidden_dim // hidden_chunk
        tokens = _offchip_load_or_restream(graph, input_tensor, stride=tuple((hidden_chunks, 1)), out_shape_tiled=tuple((seq_len, hidden_chunks)), tile_row=1, tile_col=hidden_chunk, par_dispatch=8)
        sq = BinaryMap(graph, tokens, tokens, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        sum_chunk = UnaryMap(graph, sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        sum_total = Accum(graph, sum_chunk, output_stream_dtype=_dsl2step_out_tile(sum_chunk, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(sum_chunk, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        mean_sq = UnaryMap(graph, sum_total, fn=map_fn.MulImmediate(1.0 / hidden_dim), write_back_mu=False, compute_bw=4096)
        mean_eps = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        inv_root = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        inv_root_bc = RepeatStatic(graph, inv_root, repeat_factor=hidden_chunks)
        normed = BinaryMap(graph, tokens, inv_root_bc, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)

        def _proj(x, weight, proj_dim, n_heads):
            w = _offchip_load_or_restream(graph, weight, stride=tuple((0, 1)), out_shape_tiled=tuple((seq_len, hidden_chunks)), tile_row=hidden_chunk, tile_col=proj_dim, par_dispatch=8)
            mat = BinaryMap(graph, x, w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            acc = Accum(graph, mat, output_stream_dtype=_dsl2step_out_tile(mat, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(mat, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            split = RetileStreamify(graph, acc, split_row=False, chunk=head_dim)
            resh = Reshape(graph, split, chunk_size=n_heads, reshape_rank=0, write_back_mu=False)
            absorb = Accum(graph, resh, output_stream_dtype=_dsl2step_out_tile(resh, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(resh, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
            final = Flatten(graph, absorb, min_rank=0, max_rank=1)
            return final
        q = _proj(normed, q_proj, num_heads * head_dim, num_heads)
        k = _proj(normed, k_proj, num_kv_heads * head_dim, num_kv_heads)
        v = _proj(normed, v_proj, num_kv_heads * head_dim, num_kv_heads)
        return (q, k, v)

    def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes):
        """
    Implements the per‑head RMS‑norm followed by RoPE using only STeP DSL ops.

    * RMS‑norm is built from `binary_mul`, `unary_rowwise_sum`,
      `unary_mul_imm`, `unary_add_imm`, and `unary_rsqrt`.
    * The positional encodings `cos` and `sin` are off‑chip, so they are loaded
      with `offchip_load`.  Their stream has a leading singleton dimension
      (shape (1, seq_len)).  We flatten that extra dimension away with `flatten`
      so the stream shape becomes (seq_len,) and matches Q/K.
    * To apply the RoPE half‑rotate we split the head dimension into two halves
      using `retile_streamify(..., split_row=False)`.  This expands the stream
      dimension by a factor of two and reduces the tile‑column size to half.
    * `parallelize(..., 2)` separates the two halves into independent streams.
    * The RoPE formula is then expressed with element‑wise multiplies and adds,
      using a negative constant via `unary_mul_imm`.
    * The two halves are re‑assembled with `static_reassemble`, the temporary
      stream dimension is merged back into the tile‑column with `reshape_stream`
      (splitting the stream into (64, 2)) and `accum_retile_col`.
    * V is returned unchanged.
    """
        Q_sq = BinaryMap(graph, Q, Q, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q_sum = UnaryMap(graph, Q_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        head_dim = Q.shape[-1]
        Q_mean = UnaryMap(graph, Q_sum, fn=map_fn.MulImmediate(1.0 / head_dim), write_back_mu=False, compute_bw=4096)
        Q_mean_eps = UnaryMap(graph, Q_mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        Q_inv_rms = UnaryMap(graph, Q_mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        Q_norm = BinaryMap(graph, Q, Q_inv_rms, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K_sq = BinaryMap(graph, K, K, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K_sum = UnaryMap(graph, K_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        K_mean = UnaryMap(graph, K_sum, fn=map_fn.MulImmediate(1.0 / head_dim), write_back_mu=False, compute_bw=4096)
        K_mean_eps = UnaryMap(graph, K_mean, fn=map_fn.AddImmediate(1e-06), write_back_mu=False, compute_bw=4096)
        K_inv_rms = UnaryMap(graph, K_mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        K_norm = BinaryMap(graph, K, K_inv_rms, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        seq_len = Q.shape[0]
        cos_load = _offchip_load_or_restream(graph, cos, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=head_dim, par_dispatch=8)
        sin_load = _offchip_load_or_restream(graph, sin, stride=tuple((1,)), out_shape_tiled=tuple((seq_len,)), tile_row=1, tile_col=head_dim, par_dispatch=8)
        cos_flat = Flatten(graph, cos_load, min_rank=0, max_rank=1)
        sin_flat = Flatten(graph, sin_load, min_rank=0, max_rank=1)
        half = head_dim // 2
        Q_split = RetileStreamify(graph, Q_norm, split_row=False, chunk=half)
        K_split = RetileStreamify(graph, K_norm, split_row=False, chunk=half)
        cos_split = RetileStreamify(graph, cos_flat, split_row=False, chunk=half)
        sin_split = RetileStreamify(graph, sin_flat, split_row=False, chunk=half)
        _parallelize4 = Parallelize(graph, Q_split, parallelize_rank=Q_split.stream.rank, num_consumers=2)
        Q_par = [_BranchRef(_parallelize4, _i) for _i in range(2)]
        _parallelize5 = Parallelize(graph, K_split, parallelize_rank=K_split.stream.rank, num_consumers=2)
        K_par = [_BranchRef(_parallelize5, _i) for _i in range(2)]
        _parallelize6 = Parallelize(graph, cos_split, parallelize_rank=cos_split.stream.rank, num_consumers=2)
        cos_par = [_BranchRef(_parallelize6, _i) for _i in range(2)]
        _parallelize7 = Parallelize(graph, sin_split, parallelize_rank=sin_split.stream.rank, num_consumers=2)
        sin_par = [_BranchRef(_parallelize7, _i) for _i in range(2)]
        Q0_cos = BinaryMap(graph, Q_par[0], cos_par[0], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q1_sin = BinaryMap(graph, Q_par[1], sin_par[0], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q1_sin_neg = UnaryMap(graph, Q1_sin, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        Q_half0 = BinaryMap(graph, Q0_cos, Q1_sin_neg, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        Q1_cos = BinaryMap(graph, Q_par[1], cos_par[1], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q0_sin = BinaryMap(graph, Q_par[0], sin_par[1], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        Q_half1 = BinaryMap(graph, Q1_cos, Q0_sin, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        Q_comb = StaticReassemble(graph, inputs=[Q_half0, Q_half1], merge_rank=_dsl2step_stream([Q_half0, Q_half1][0]).rank)
        Q_resh = Reshape(graph, Q_comb, chunk_size=2, reshape_rank=0, write_back_mu=False)
        Q_out = Accum(graph, Q_resh, output_stream_dtype=_dsl2step_out_tile(Q_resh, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(Q_resh, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        K0_cos = BinaryMap(graph, K_par[0], cos_par[0], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K1_sin = BinaryMap(graph, K_par[1], sin_par[0], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K1_sin_neg = UnaryMap(graph, K1_sin, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        K_half0 = BinaryMap(graph, K0_cos, K1_sin_neg, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        K1_cos = BinaryMap(graph, K_par[1], cos_par[1], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K0_sin = BinaryMap(graph, K_par[0], sin_par[1], fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        K_half1 = BinaryMap(graph, K1_cos, K0_sin, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        K_comb = StaticReassemble(graph, inputs=[K_half0, K_half1], merge_rank=_dsl2step_stream([K_half0, K_half1][0]).rank)
        K_resh = Reshape(graph, K_comb, chunk_size=2, reshape_rank=0, write_back_mu=False)
        K_out = Accum(graph, K_resh, output_stream_dtype=_dsl2step_out_tile(K_resh, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(K_resh, 'col'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        return (Q_out, K_out, V)

    def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin, *, out_shapes):
        Q0, K0, V0 = pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, out_shapes=out_shapes)
        Q, K, V = per_head_norm_and_rope(Q0, K0, V0, cos, sin, out_shapes=out_shapes)
        return (Q, K, V)

    def compute_qkv(Q, K, V, *, out_shapes):
        q = RetileStreamify(graph, Q, split_row=True, chunk=1)
        q = PromoteOuter(graph, q)
        q = Bufferize(graph, q, rank=1)
        q = Streamify(graph, q, stride=tuple([4, 1, 16]), out_shape_tiled=tuple([4, 4, 64]))
        q = Flatten(graph, q, min_rank=2, max_rank=3)
        Qh = Accum(graph, q, output_stream_dtype=_dsl2step_out_tile(q, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(q, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        k = RetileStreamify(graph, K, split_row=True, chunk=1)
        k = PromoteOuter(graph, k)
        k = Bufferize(graph, k, rank=1)
        k = Streamify(graph, k, stride=tuple([1, 4]), out_shape_tiled=tuple([4, 64]))
        k = Flatten(graph, k, min_rank=1, max_rank=2)
        k = Accum(graph, k, output_stream_dtype=_dsl2step_out_tile(k, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(k, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Kh = Promote(graph, k, promote_rank=0)
        v = RetileStreamify(graph, V, split_row=True, chunk=1)
        v = PromoteOuter(graph, v)
        v = Bufferize(graph, v, rank=1)
        v = Streamify(graph, v, stride=tuple([1, 4]), out_shape_tiled=tuple([4, 64]))
        v = Flatten(graph, v, min_rank=1, max_rank=2)
        v = Accum(graph, v, output_stream_dtype=_dsl2step_out_tile(v, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(v, 'row'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        Vh = Promote(graph, v, promote_rank=0)
        return (Qh, Kh, Vh)

    def attention_compute(Qh, Kh, Vh, *, out_shapes):
        Kh_b = ExpandRef(graph, Kh, ref=Qh, expand_rank=1)
        Vh_b = ExpandRef(graph, Vh, ref=Qh, expand_rank=1)
        scores = BinaryMap(graph, Qh, Kh_b, fn=map_fn.Matmul(weight_transposed=True), write_back_mu=False, compute_bw=4096)
        scores_promoted = Promote(graph, scores, promote_rank=0)
        scores_split = RetileStreamify(graph, scores_promoted, split_row=False, chunk=1)
        row_max = Accum(graph, scores_split, output_stream_dtype=_dsl2step_out_tile(scores_split, 'elem', 1), fn=accum_fn.Max(), init_fn=_dsl2step_init(scores_split, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        _tmp1 = UnaryMap(graph, row_max, fn=map_fn.MulImmediate(-1.0), write_back_mu=False, compute_bw=4096)
        scores_centered = BinaryMap(graph, scores, _tmp1, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        e = UnaryMap(graph, scores_centered, fn=map_fn.Exp(), write_back_mu=False, compute_bw=4096)
        num = BinaryMap(graph, e, Vh_b, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        denom = UnaryMap(graph, e, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        attn = BinaryMap(graph, num, denom, fn=map_fn.Div(), write_back_mu=False, compute_bw=4096)
        attn_collapsed = Accum(graph, attn, output_stream_dtype=_dsl2step_out_tile(attn, 'col', 2), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(attn, 'col'), accum_rank=2, write_back_mu=False, compute_bw=4096)
        attn_promoted = Promote(graph, attn_collapsed, promote_rank=0)
        out = RetileStreamify(graph, attn_promoted, split_row=True, chunk=1)
        return out

    def attention(Q, K, V, *, out_shapes):
        Qh, Kh, Vh = compute_qkv(Q, K, V, out_shapes=((4, 4, 64, 32), (4, 1, 64, 32), (4, 1, 64, 32)))
        attn = attention_compute(Qh, Kh, Vh, out_shapes=out_shapes)
        return attn

    def attention_o_proj(Q, K, V, o_proj_weight, input_tensor, *, out_shapes):
        attn = attention(Q, K, V, out_shapes=((64, 1, 512),))
        w_raw = _offchip_load_or_restream(graph, o_proj_weight, stride=tuple((0,)), out_shape_tiled=tuple((64,)), tile_row=512, tile_col=512, par_dispatch=8)
        w = Flatten(graph, w_raw, min_rank=0, max_rank=1)
        proj = BinaryMap(graph, attn, w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        resid_raw = _offchip_load_or_restream(graph, input_tensor, stride=tuple((1,)), out_shape_tiled=tuple((64,)), tile_row=1, tile_col=512, par_dispatch=8)
        resid = Flatten(graph, resid_raw, min_rank=0, max_rank=1)
        out = BinaryMap(graph, proj, resid, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return out

    def rms_norm(res_add_0, *, out_shapes):
        x_sq = UnaryMap(graph, res_add_0, fn=map_fn.Square(), write_back_mu=False, compute_bw=4096)
        sum_sq = UnaryMap(graph, x_sq, fn=map_fn.RowWiseSum(), write_back_mu=False, compute_bw=4096)
        tile_col = res_add_0.stream_dtype.shape[1]
        inv_tile_col = 1.0 / tile_col
        mean_sq = UnaryMap(graph, sum_sq, fn=map_fn.MulImmediate(inv_tile_col), write_back_mu=False, compute_bw=4096)
        eps = 1e-06
        mean_eps = UnaryMap(graph, mean_sq, fn=map_fn.AddImmediate(eps), write_back_mu=False, compute_bw=4096)
        rsqrt = UnaryMap(graph, mean_eps, fn=map_fn.Rsqrt(), write_back_mu=False, compute_bw=4096)
        out = BinaryMap(graph, res_add_0, rsqrt, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        return out

    def moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes):
        top_k = expert_onehot.shape[1]
        num_experts = w_gate.shape[0]
        sel = SelectGen(is_multihot=True, tensor=expert_onehot, n=num_experts)
        graph.add_node(sel)
        tok_stream = RepeatStatic(graph, normed_2, repeat_factor=top_k)
        tok_stream = PromoteOuter(graph, tok_stream)
        w_scalar = _offchip_load_or_restream(graph, expert_weights, stride=tuple((top_k, 1)), out_shape_tiled=tuple((expert_weights.shape[0], top_k)), tile_row=1, tile_col=1, par_dispatch=8)
        w_scalar = Flatten(graph, w_scalar, min_rank=1, max_rank=2)
        w_scalar = PromoteOuter(graph, w_scalar)
        _flat_partition8 = FlatPartition(graph, tok_stream, control=sel, partition_rank=0, switch_cycles=[1] * num_experts, write_back_mu=False, num_consumers=num_experts)
        token_parts = [_BranchRef(_flat_partition8, _i) for _i in range(num_experts)]
        _flat_partition9 = FlatPartition(graph, w_scalar, control=sel, partition_rank=0, switch_cycles=[1] * num_experts, write_back_mu=False, num_consumers=num_experts)
        weight_parts = [_BranchRef(_flat_partition9, _i) for _i in range(num_experts)]
        expert_outputs = []
        for i in range(num_experts):
            gate_mat = LinearOffChipLoadRef(graph, ref=token_parts[i], underlying=w_gate[i], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=normed_2.shape[2], tile_col=w_gate.shape[2], par_dispatch=8)
            up_mat = LinearOffChipLoadRef(graph, ref=token_parts[i], underlying=w_up[i], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=normed_2.shape[2], tile_col=w_up.shape[2], par_dispatch=8)
            down_mat = LinearOffChipLoadRef(graph, ref=token_parts[i], underlying=w_down[i], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=w_down.shape[1], tile_col=w_down.shape[2], par_dispatch=8)
            tok_i = Promote(graph, token_parts[i], promote_rank=0)
            weight_i = Promote(graph, weight_parts[i], promote_rank=0)
            gate_out = BinaryMap(graph, tok_i, gate_mat, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            up_out = BinaryMap(graph, tok_i, up_mat, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            _tmp2 = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
            hidden = BinaryMap(graph, _tmp2, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            down_out = BinaryMap(graph, hidden, down_mat, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
            weighted = BinaryMap(graph, down_out, weight_i, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
            expert_outputs.append(weighted)
        merged = FlatReassemble(graph, inputs=expert_outputs, control=sel, reassemble_rank=0, switch_cycles=[1] * len(expert_outputs), write_back_mu=False)
        merged = Accum(graph, merged, output_stream_dtype=_dsl2step_out_tile(merged, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(merged, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        merged = Accum(graph, merged, output_stream_dtype=_dsl2step_out_tile(merged, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(merged, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
        result = Flatten(graph, merged, min_rank=0, max_rank=1)
        return result

    def moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes):
        normed_2 = rms_norm(res_add_0, out_shapes=out_shapes)
        result = moe_dispatch__root_moe_moe_dispatch(normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes)
        return result

    def moe(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, *, out_shapes):
        moe_out = moe_dispatch(res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot, out_shapes=out_shapes)
        _tmp3 = BinaryMap(graph, moe_out, res_add_0, fn=map_fn.Add(), write_back_mu=False, compute_bw=4096)
        return _tmp3
    '\n    Root DSL for the pre‑fill transformer (attention + MoE).\n\n    Variant choices:\n      - pre_attention : variant\u202f1 (smaller on‑chip buffers)\n      - attention_o_proj : variant\u202f0 (exact reference implementation, includes the\n                           post‑attention RMSNorm)\n      - moe : variant\u202f0 (reference MoE, includes the final residual addition)\n\n    A leading static stream dimension is inserted with `promote_outer` so that\n    `offchip_store` receives a tensor with at least two stream dimensions.\n    '
    seq_len = dims['seq_len']
    Q, K, V = pre_attention(tensors['input_tensor'], tensors['q_proj'], tensors['k_proj'], tensors['v_proj'], tensors['cos'], tensors['sin'], out_shapes=((seq_len, 16, 32), (seq_len, 4, 32), (seq_len, 4, 32)))
    res_add_0 = attention_o_proj(Q, K, V, tensors['o_proj_weight'], tensors['input_tensor'], out_shapes=((seq_len, 1, 512),))
    out = moe(res_add_0, tensors['w_gate'], tensors['w_up'], tensors['w_down'], tensors['expert_weights'], tensors['expert_onehot'], out_shapes=((seq_len, 1, 512),))
    out = PromoteOuter(graph, out)
    _store10 = OffChipStore(graph, out, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store10)