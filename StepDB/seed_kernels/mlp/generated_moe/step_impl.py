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

def _offchip_load_or_restream(graph, underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False, par_dispatch=8, start_tile_idx=0):
    if not isinstance(underlying, (_StepOps, _BranchRef)):
        node = LinearOffChipLoad(underlying, stride=tuple(stride), out_shape_tiled=tuple(out_shape_tiled), tile_row=tile_row, tile_col=tile_col, transposed=transposed, par_dispatch=par_dispatch, start_tile_idx=start_tile_idx)
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
    x = _offchip_load_or_restream(graph, tensors['x'], stride=tuple((1,)), out_shape_tiled=tuple((dims['B'],)), tile_row=1, tile_col=dims['D'], par_dispatch=8, start_tile_idx=0)
    x_rep = RepeatStatic(graph, x, repeat_factor=dims['n_active'])
    weight = _offchip_load_or_restream(graph, tensors['expert_weights'], stride=tuple((dims['n_active'], 1)), out_shape_tiled=tuple((dims['B'], dims['n_active'])), tile_row=1, tile_col=1, par_dispatch=8, start_tile_idx=0)
    selector = SelectGen(is_multihot=True, tensor=tensors['expert_onehot'], n=dims['n_experts'])
    graph.add_node(selector)
    _flat_partition2 = FlatPartition(graph, x_rep, control=selector, partition_rank=0, switch_cycles=[1] * dims['n_experts'], write_back_mu=False, num_consumers=dims['n_experts'])
    token_parts = [_BranchRef(_flat_partition2, _i) for _i in range(dims['n_experts'])]
    _flat_partition3 = FlatPartition(graph, weight, control=selector, partition_rank=0, switch_cycles=[1] * dims['n_experts'], write_back_mu=False, num_consumers=dims['n_experts'])
    weight_parts = [_BranchRef(_flat_partition3, _i) for _i in range(dims['n_experts'])]
    contributions = []
    for i in range(dims['n_experts']):
        token_i = token_parts[i]
        weight_i = weight_parts[i]
        gate_i = LinearOffChipLoadRef(graph, ref=token_i, underlying=tensors['gate_weights'][i], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=tensors['gate_weights'][i].shape[0], tile_col=tensors['gate_weights'][i].shape[1], par_dispatch=8, start_tile_idx=0)
        gate_i = Flatten(graph, gate_i, min_rank=0, max_rank=1)
        up_i = LinearOffChipLoadRef(graph, ref=token_i, underlying=tensors['up_weights'][i], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=tensors['up_weights'][i].shape[0], tile_col=tensors['up_weights'][i].shape[1], par_dispatch=8, start_tile_idx=0)
        up_i = Flatten(graph, up_i, min_rank=0, max_rank=1)
        gate_out = BinaryMap(graph, token_i, gate_i, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        up_out = BinaryMap(graph, token_i, up_i, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        _tmp1 = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
        proj = BinaryMap(graph, _tmp1, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        down_i = LinearOffChipLoadRef(graph, ref=token_i, underlying=tensors['down_weights'][i], stride=tuple((1,)), out_shape_tiled=tuple((1,)), tile_row=tensors['down_weights'][i].shape[0], tile_col=tensors['down_weights'][i].shape[1], par_dispatch=8, start_tile_idx=0)
        down_i = Flatten(graph, down_i, min_rank=0, max_rank=1)
        down_out = BinaryMap(graph, proj, down_i, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        contrib = BinaryMap(graph, down_out, weight_i, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        contributions.append(contrib)
    merged = FlatReassemble(graph, inputs=contributions, control=selector, reassemble_rank=0, switch_cycles=[1] * len(contributions), write_back_mu=False)
    merged = Accum(graph, merged, output_stream_dtype=_dsl2step_out_tile(merged, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(merged, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    merged = Accum(graph, merged, output_stream_dtype=_dsl2step_out_tile(merged, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(merged, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=4096)
    _store4 = OffChipStore(graph, merged, par_dispatch=8)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store4)