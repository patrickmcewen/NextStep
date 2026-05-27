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

def build_graph(dims: dict, tensors: dict):
    graph = Graph()
    B = dims['B']
    D = dims['D']
    F_dim = dims['F']
    n_experts = dims['n_experts']
    n_active = dims['n_active']
    tile_f = dims['tile_f']
    num_f_tiles = F_dim // tile_f
    x_stream = LinearOffChipLoad(tensors['x'], stride=(1,), out_shape_tiled=(B,), tile_row=1, tile_col=D, par_dispatch=16)
    graph.add_node(x_stream)
    weight_stream = LinearOffChipLoad(tensors['expert_weights'], stride=(n_active, 1), out_shape_tiled=(B, n_active), tile_row=1, tile_col=1, par_dispatch=16)
    graph.add_node(weight_stream)
    expert_multihot_ctrl = SelectGen(is_multihot=True, tensor=tensors['expert_multihot'], n=n_experts)
    graph.add_node(expert_multihot_ctrl)
    expert_onehot_ctrl = SelectGen(is_multihot=True, tensor=tensors['expert_onehot'], n=n_experts)
    graph.add_node(expert_onehot_ctrl)
    _flat_partition2 = FlatPartition(graph, x_stream, control=expert_multihot_ctrl, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
    x_parts = [_BranchRef(_flat_partition2, _i) for _i in range(n_experts)]
    _flat_partition3 = FlatPartition(graph, weight_stream, control=expert_onehot_ctrl, partition_rank=0, switch_cycles=[1] * n_experts, write_back_mu=False, num_consumers=n_experts)
    weight_parts = [_BranchRef(_flat_partition3, _i) for _i in range(n_experts)]
    expert_contributions = []
    for i in range(n_experts):
        xi = x_parts[i]
        wi = weight_parts[i]
        xi_rep = RepeatStatic(graph, xi, repeat_factor=num_f_tiles)
        wi_rep = RepeatStatic(graph, wi, repeat_factor=num_f_tiles)
        gate_w = LinearOffChipLoadRef(graph, ref=xi, underlying=tensors['gate_weights'][i], stride=(1,), out_shape_tiled=(num_f_tiles,), tile_row=D, tile_col=tile_f, par_dispatch=16)
        up_w = LinearOffChipLoadRef(graph, ref=xi, underlying=tensors['up_weights'][i], stride=(1,), out_shape_tiled=(num_f_tiles,), tile_row=D, tile_col=tile_f, par_dispatch=16)
        gate_out = BinaryMap(graph, xi_rep, gate_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        up_out = BinaryMap(graph, xi_rep, up_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        _tmp1 = UnaryMap(graph, gate_out, fn=map_fn.Silu(), write_back_mu=False, compute_bw=4096)
        proj = BinaryMap(graph, _tmp1, up_out, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        down_w = LinearOffChipLoadRef(graph, ref=xi, underlying=tensors['down_weights'][i], stride=(1,), out_shape_tiled=(num_f_tiles,), tile_row=tile_f, tile_col=D, par_dispatch=16)
        down_out = BinaryMap(graph, proj, down_w, fn=map_fn.Matmul(), write_back_mu=False, compute_bw=4096)
        weighted = BinaryMap(graph, down_out, wi_rep, fn=map_fn.Mul(), write_back_mu=False, compute_bw=4096)
        expert_sum = Accum(graph, weighted, output_stream_dtype=_dsl2step_out_tile(weighted, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(weighted), accum_rank=1, write_back_mu=False, compute_bw=4096)
        expert_contributions.append(expert_sum)
    y_flat = FlatReassemble(graph, inputs=expert_contributions, control=expert_multihot_ctrl, reassemble_rank=0, switch_cycles=[1] * len(expert_contributions), write_back_mu=False)
    y_sum = Accum(graph, y_flat, output_stream_dtype=_dsl2step_out_tile(y_flat, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(y_flat), accum_rank=1, write_back_mu=False, compute_bw=4096)
    y_out = OffChipStore(graph, y_sum, par_dispatch=16)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, y_out)