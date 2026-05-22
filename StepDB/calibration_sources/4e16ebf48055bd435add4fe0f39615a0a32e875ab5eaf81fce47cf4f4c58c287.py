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

def _offchip_load_or_restream(graph, underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False, par_dispatch=1, start_tile_idx=0):
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
    rcol = Accum(graph, sm, output_stream_dtype=_dsl2step_out_tile(sm, 'col', 1), fn=accum_fn.RetileCol(), init_fn=_dsl2step_init(sm, 'col'), accum_rank=1, write_back_mu=False, compute_bw=1)
    rrow = Accum(graph, rcol, output_stream_dtype=_dsl2step_out_tile(rcol, 'row', 1), fn=accum_fn.RetileRow(), init_fn=_dsl2step_init(rcol, 'row'), accum_rank=1, write_back_mu=False, compute_bw=1)
    return PromoteOuter(graph, rrow)

def build_graph(dims, tensors):
    graph = Graph()

    def downtile(x, sub_r, sub_c):
        """Split each (T_r, T_c) tile into a (T_r/sub_r) x (T_c/sub_c) grid of
    (sub_r, sub_c) sub-tiles. Adds two new innermost stream dims for the
    sub-grid (row-chunks outer, col-chunks inner).

        stream (..., D)             tile (T_r, T_c)
                      ->
        stream (..., D, n_row, n_col)   tile (sub_r, sub_c)
    """
        tile_r, tile_c = (int(x.shape[-2]), int(x.shape[-1]))
        assert tile_r % sub_r == 0 and tile_c % sub_c == 0, f'downtile: ({tile_r},{tile_c}) not divisible by ({sub_r},{sub_c})'
        n_row = tile_r // sub_r
        n_col = tile_c // sub_c
        y = RetileStreamify(graph, x, split_row=True, chunk=sub_r)
        y = RetileStreamify(graph, y, split_row=False, chunk=sub_c)
        y = Reshape(graph, y, chunk_size=n_col, reshape_rank=0, write_back_mu=False)
        y = Reshape(graph, y, chunk_size=n_row, reshape_rank=1, write_back_mu=False)
        return y
    "GEMM with big-tile DRAM loads, one-big-tile on-chip working set,\n    and a two-stage K reduction.\n\n    Each side keeps only ONE big tile on chip (256 KB) by using offchip_load\n    with stride-0 broadcasts:\n      - A iterates (m_l, n_l, k_l) with n_l stride 0 (A doesn't depend on N).\n      - B iterates (m_l, n_l, k_l) with m_l stride 0 (B doesn't depend on M).\n    Each (m_l, n_l, k_l) outer step loads one big tile; each big-tile pair is\n    downtiled, matmul'd on (sub, sub) sub-tiles, and accumulated.\n\n    Natural stream order from this layout is (m_l, n_l, k_l, sub_r, sub_n, sub_c).\n    K splits across positions 2 and 5 — non-adjacent — so K is reduced in two\n    stages:\n      stage 1: binary_map_accum(rank=1)  matmul + sum sub_c.\n      stage 2: bufferize(rank=3) + restream to put k_l innermost,\n               then accum_add(rank=1) to sum k_l.\n    After stage 2 the stream is (m_l, n_l, sub_r, sub_n), which can't be\n    flattened directly to (M_small, N_small) because (m_l, sub_r) and\n    (n_l, sub_n) aren't adjacent. One final bufferize+restream rotates\n    the per-m_l buffer so order becomes (m_l, sub_r, n_l, sub_n) and the\n    two flattens recover the (M_small, N_small) layout for offchip_store.\n\n    SRAM per side: ~256 KB (big tile) + 4 MB (k_l reorder) + 4 MB (output\n    reorder) ≈ 8.5 MB total — vs. 128 MB for full rank=4 bufferize.\n    DRAM cost: 16x amplification on each side (re-reads big tiles across\n    the broadcast axis).\n    "
    big = 256
    sub = 16
    n_row = n_col = big // sub
    M_tiles = dims['M'] // big
    K_tiles = dims['K'] // big
    N_tiles = dims['N'] // big
    sub_per_k = n_row * n_col
    A_load = _offchip_load_or_restream(graph, tensors['A'], stride=tuple((K_tiles, 0, 1)), out_shape_tiled=tuple((M_tiles, N_tiles, K_tiles)), tile_row=big, tile_col=big, par_dispatch=1, start_tile_idx=0)
    A_buf = Bufferize(graph, downtile(A_load, sub, sub), rank=2)
    A_stream = Streamify(graph, A_buf, stride=tuple((n_col, 0, 1)), out_shape_tiled=tuple((n_row, n_col, n_col)))
    B_load = _offchip_load_or_restream(graph, tensors['B'], stride=tuple((0, 1, N_tiles)), out_shape_tiled=tuple((M_tiles, N_tiles, K_tiles)), tile_row=big, tile_col=big, par_dispatch=1, start_tile_idx=0)
    B_buf = Bufferize(graph, downtile(B_load, sub, sub), rank=2)
    B_stream = Streamify(graph, B_buf, stride=tuple((0, 1, n_col)), out_shape_tiled=tuple((n_row, n_col, n_col)))
    C_partial = BinaryMapAccum(graph, A_stream, B_stream, fn=map_accum_fn.Matmul(), init_fn=_dsl2step_init(A_stream, 'elem'), rank=1, write_back_mu=False, compute_bw=1)
    C_buf = Bufferize(graph, C_partial, rank=3)
    C_re = Streamify(graph, C_buf, stride=tuple((n_col, 1, sub_per_k)), out_shape_tiled=tuple((n_row, n_col, K_tiles)))
    C_red = Accum(graph, C_re, output_stream_dtype=_dsl2step_out_tile(C_re, 'elem', 1), fn=accum_fn.Add(), init_fn=_dsl2step_init(C_re, 'elem'), accum_rank=1, write_back_mu=False, compute_bw=1)
    out_buf = Bufferize(graph, C_red, rank=3)
    out_re = Streamify(graph, out_buf, stride=tuple((n_col, sub_per_k, 1)), out_shape_tiled=tuple((n_row, N_tiles, n_col)))
    out_re = Flatten(graph, out_re, min_rank=0, max_rank=1)
    out_re = Flatten(graph, out_re, min_rank=1, max_rank=3)
    _store1 = OffChipStore(graph, out_re, par_dispatch=1)
    _seal_unused_branches(graph)
    graph = infer_broadcast(graph)
    return (graph, _store1)