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

def _dsl2step_out_tile(x, mode, accum_rank):
    sd = x.stream.stream_dtype
    if mode == 'elem':
        return sd
    dims = x.stream.shape[-accum_rank:]
    if any((not isinstance(d, int) for d in dims)):
        return sd
    tr, tc = sd.shape
    mul = 1
    for d in dims:
        mul *= d
    if mode == 'row':
        return Tile(tile_dtype=sd.tile_dtype, shape=(tr * mul, tc))
    return Tile(tile_dtype=sd.tile_dtype, shape=(tr, tc * mul))

def _dsl2step_init(x):
    return Empty(shape=(1, 1), dtype=x.stream.stream_dtype.tile_dtype)

def _dsl2step_in_tile(x):
    if isinstance(x, tuple) and len(x) == 2:
        node, idx = x
        return node.stream_idx(idx).stream_dtype
    return x.stream.stream_dtype

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
    '\n    Faster tiled MoE routing: each expert’s weight matrix is loaded once,\n    broadcast via expand_ref+flatten, and the cheap element‑wise ops are\n    given a larger compute_bw so they no longer become a secondary bottleneck.\n    '
    B = dims['B']
    D = dims['D']
    F = dims['F']
    n_experts = dims['n_experts']
    n_active = dims['n_active']
    expert_multihot_ctrl = SelectGen(is_multihot=True, tensor=tensors['expert_multihot'], n=n_experts)
    graph.add_node(expert_multihot_ctrl)
    expert_onehot_ctrl = SelectGen(is_multihot=True, tensor=tensors['expert_onehot'], n=n_experts)
    graph.add_node(expert_onehot_ctrl)
    x = LinearOffChipLoad(tensors['x'], stride=(1,), out_shape_tiled=(B,), tile_row=1, tile_col=D, par_dispatch=8)
    graph.add_node(x)
    _flat_partition8 = FlatPartition(graph, x, control=expert_multihot_ctrl, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
    token_streams = [_BranchRef(_flat_partition8, _i) for _i in range(n_experts)]
    expert_outs = []
    for i in range(n_experts):
        xs = token_streams[i]
        gate_w_one = LinearOffChipLoad(tensors['gate_weights'][i], stride=(0,), out_shape_tiled=(1,), tile_row=D, tile_col=F, par_dispatch=16)
        graph.add_node(gate_w_one)
        _tmp1 = PromoteOuter(graph, xs)
        _tmp2 = ExpandRef(graph, gate_w_one, ref=_tmp1, expand_rank=2)
        gate_w = Flatten(graph, _tmp2, min_rank=0, max_rank=1)
        up_w_one = LinearOffChipLoad(tensors['up_weights'][i], stride=(0,), out_shape_tiled=(1,), tile_row=D, tile_col=F, par_dispatch=16)
        graph.add_node(up_w_one)
        _tmp3 = PromoteOuter(graph, xs)
        _tmp4 = ExpandRef(graph, up_w_one, ref=_tmp3, expand_rank=2)
        up_w = Flatten(graph, _tmp4, min_rank=0, max_rank=1)
        gate = BinaryMap(graph, xs, gate_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1000)
        gate = UnaryMap(graph, gate, fn=map_fn.Silu(), write_back_mu=False, compute_bw=10)
        up = BinaryMap(graph, xs, up_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1000)
        proj = BinaryMap(graph, gate, up, fn=map_fn.Mul(), write_back_mu=False, compute_bw=10)
        down_w_one = LinearOffChipLoad(tensors['down_weights'][i], stride=(0,), out_shape_tiled=(1,), tile_row=F, tile_col=D, par_dispatch=16)
        graph.add_node(down_w_one)
        _tmp5 = PromoteOuter(graph, proj)
        _tmp6 = ExpandRef(graph, down_w_one, ref=_tmp5, expand_rank=2)
        down_w = Flatten(graph, _tmp6, min_rank=0, max_rank=1)
        down = BinaryMap(graph, proj, down_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=1000)
        expert_outs.append(down)
    exp_w = LinearOffChipLoad(tensors['expert_weights'], stride=(n_active, 1), out_shape_tiled=(B, n_active), tile_row=1, tile_col=1, par_dispatch=8)
    graph.add_node(exp_w)
    _flat_partition9 = FlatPartition(graph, exp_w, control=expert_onehot_ctrl, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
    weight_streams = [_BranchRef(_flat_partition9, _i) for _i in range(n_experts)]
    weighted_expert_outs = []
    for i in range(n_experts):
        _tmp7 = BinaryMap(graph, expert_outs[i], weight_streams[i], fn=map_fn.Mul(), write_back_mu=False, compute_bw=10)
        weighted_expert_outs.append(_tmp7)
    y_stream = FlatReassemble(graph, inputs=weighted_expert_outs, control=expert_onehot_ctrl, reassemble_rank=0, switch_cycles=[1] * len(weighted_expert_outs), write_back_mu=False)
    y_sum = Accum(graph, y_stream, output_stream_dtype=_dsl2step_out_tile(y_stream, 'elem', 2), fn=accum_fn.Add(), init_fn=_dsl2step_init(y_stream), accum_rank=2, write_back_mu=False, compute_bw=10)
    y = OffChipStore(graph, y_sum, par_dispatch=1)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, y)