"""Deterministic translator: DSL-refactored ``tiled_reference`` -> STeP ``build_graph``.

Replaces the LLM ``translate`` pass. Each DSL call in
``StepGenFlow8/src/step_dsl.py`` maps to one STeP IR node construction.

Public API:
    translate(dsl_code: str) -> str

The output is a Python source string containing two top-level definitions:
    * a small set of helper functions used by the generated graph builder
    * ``def build_graph(dims, tensors)`` returning ``(graph, output_op)``

The output is intended to be exec'd under ``IMPORT_SCAFFOLD`` and run via
``execute(graph, output_op)`` -- exactly the harness used by
``_run_graph_correctness`` in ``orchestrator.py``.
"""
import ast


# DSL fn -> (STeP node class, map_fn / accum_fn class name).
_BINARY_MAP = {
    "binary_matmul":   "Matmul",
    "binary_mul":      "Mul",
    "binary_add":      "Add",
    "binary_div":      "Div",
    "binary_is_equal": "IsEqual",
}
_UNARY_MAP = {
    "unary_silu":        "Silu",
    "unary_square":      "Square",
    "unary_exp":         "Exp",
    "unary_rsqrt":       "Rsqrt",
    "unary_pow2":        "Pow2",
    "unary_mul_imm":     "MulImmediate",
    "unary_add_imm":     "AddImmediate",
    "unary_sub_imm":     "SubImmediate",
    "unary_rowwise_sum": "RowWiseSum",
}
_ACCUM_MAP = {
    # accum_fn class name, output-tile mode
    "accum_add":        ("Add",       "elem"),
    "accum_mul":        ("Mul",       "elem"),
    "accum_retile_row": ("RetileRow", "row"),
    "accum_retile_col": ("RetileCol", "col"),
}
_MULTI_OUTPUT = {"broadcast", "parallelize", "flat_partition"}

# All DSL function names that this translator knows how to rewrite.
_DSL_NAMES: set = set()


# Helper function definitions injected at the top of the generated module.
# These compute the output tile dtype and init_fn for Accum nodes at
# graph-build time using STeP's own stream inference, instead of re-implementing
# STeP's shape algebra here.
_HELPER_PRELUDE = """\
# Expose `.shape` on StepOps nodes so DSL code that introspects tensor shapes
# (e.g. `len(x.shape)`) keeps working at graph-build time. At DSL evaluation
# time `x` is a torch.Tensor with `.shape == stream + tile`; we mirror that.
_StepOps = BinaryMap.__mro__[1]
if not hasattr(_StepOps, 'shape'):
    _StepOps.shape = property(
        lambda self: tuple(self.stream.shape) + tuple(self.stream.stream_dtype.shape)
    )


def _dsl2step_out_tile(x, mode, accum_rank):
    sd = x.stream.stream_dtype
    if mode == 'elem':
        return sd
    dims = x.stream.shape[-accum_rank:]
    if any(not isinstance(d, int) for d in dims):
        # Stream dim is symbolic (DynDim from FlatPartition / FlatReassemble);
        # fall back to the input tile dtype and let STeP resolve the size.
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
"""


def translate(dsl_code: str) -> str:
    """Convert DSL ``tiled_reference`` source into a STeP ``build_graph`` source."""
    tree = ast.parse(dsl_code)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "tiled_reference":
            fn = node
            break
    assert fn is not None, "translate: input must define def tiled_reference(...)"

    state = _State()
    body = _denest_block(state, fn.body)
    body = state.rewrite_block(body)
    body.insert(0, _stmt("graph = Graph()"))

    fn.body = body
    fn.name = "build_graph"
    fn.returns = None

    helpers = ast.parse(_HELPER_PRELUDE).body
    tree.body = helpers + [fn]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _stmt(src: str) -> ast.stmt:
    return ast.parse(src).body[0]


def _block(src: str) -> list:
    return ast.parse(src).body


def _src(node) -> str:
    assert node is not None, "_src: missing required argument"
    return ast.unparse(node)


def _arg(call: ast.Call, pos: int, name: str):
    """Get a Call argument by keyword name or by positional index."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    if pos < len(call.args):
        return call.args[pos]
    return None


def _arg_or_default(call, pos, name, default_src: str) -> str:
    node = _arg(call, pos, name)
    return _src(node) if node is not None else default_src


# ---------------------------------------------------------------------------
# Pre-pass: lift every nested DSL Call into a preceding tmp assignment so the
# rewriter only needs to handle one DSL Call per statement.
# ---------------------------------------------------------------------------

def _denest_block(state, body, is_outer=True):
    """Lift nested DSL Calls in each statement.

    ``is_outer`` distinguishes the top-level ``tiled_reference`` body (where a
    ``return dsl_call(...)`` is handled specially by the rewriter) from nested
    function bodies (where such a return must be lifted to a tmp).
    """
    out = []
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stmt.body = _denest_block(state, stmt.body, is_outer=False)
            out.append(stmt)
        elif isinstance(stmt, ast.For):
            stmt.body = _denest_block(state, stmt.body, is_outer=is_outer)
            stmt.orelse = _denest_block(state, stmt.orelse, is_outer=is_outer)
            out.append(stmt)
        elif isinstance(stmt, ast.If):
            stmt.body = _denest_block(state, stmt.body, is_outer=is_outer)
            stmt.orelse = _denest_block(state, stmt.orelse, is_outer=is_outer)
            out.append(stmt)
        elif isinstance(stmt, ast.While):
            stmt.body = _denest_block(state, stmt.body, is_outer=is_outer)
            stmt.orelse = _denest_block(state, stmt.orelse, is_outer=is_outer)
            out.append(stmt)
        elif isinstance(stmt, ast.Assign):
            pre = []
            stmt.value = _lift(state, stmt.value, pre, lift_self=False)
            out.extend(pre)
            out.append(stmt)
        elif isinstance(stmt, ast.Return):
            pre = []
            if stmt.value is not None:
                # Outer return: the rewriter handles offchip_store(...) specially,
                # so leave it in place. Inner return: lift any DSL Call to a tmp.
                stmt.value = _lift(state, stmt.value, pre,
                                   lift_self=not is_outer)
            out.extend(pre)
            out.append(stmt)
        elif isinstance(stmt, ast.Expr):
            pre = []
            stmt.value = _lift(state, stmt.value, pre, lift_self=True)
            out.extend(pre)
            # If the entire Expr was lifted, skip the now-empty stmt.
            if isinstance(stmt.value, ast.Name):
                continue
            out.append(stmt)
        else:
            out.append(stmt)
    return out


def _lift(state, expr, pre, lift_self: bool):
    """Recursively replace nested DSL Calls in ``expr`` with tmp Names.

    ``lift_self=True``  -> if ``expr`` itself is a DSL Call, lift it.
    ``lift_self=False`` -> keep ``expr`` in place (used for the RHS of an Assign).
    Children are always lifted with ``lift_self=True``.
    """
    if isinstance(expr, ast.Call):
        expr.args = [_lift(state, a, pre, True) for a in expr.args]
        for kw in expr.keywords:
            kw.value = _lift(state, kw.value, pre, True)
        if (lift_self
                and isinstance(expr.func, ast.Name)
                and expr.func.id in _DSL_NAMES):
            tmp = state.fresh("tmp")
            pre.append(ast.Assign(
                targets=[ast.Name(id=tmp, ctx=ast.Store())],
                value=expr,
            ))
            return ast.Name(id=tmp, ctx=ast.Load())
        return expr
    if isinstance(expr, ast.BinOp):
        expr.left = _lift(state, expr.left, pre, True)
        expr.right = _lift(state, expr.right, pre, True)
        return expr
    if isinstance(expr, ast.UnaryOp):
        expr.operand = _lift(state, expr.operand, pre, True)
        return expr
    if isinstance(expr, (ast.List, ast.Tuple)):
        expr.elts = [_lift(state, e, pre, True) for e in expr.elts]
        return expr
    if isinstance(expr, ast.Subscript):
        expr.value = _lift(state, expr.value, pre, True)
        return expr
    return expr


# ---------------------------------------------------------------------------
# Rewriter
# ---------------------------------------------------------------------------

class _State:
    def __init__(self):
        self.counter = 0
        self.select_gen_vars: set = set()
        self.fn_depth = 0  # 0 == top-level tiled_reference; >0 == nested def

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"_{prefix}{self.counter}"

    def rewrite_block(self, body):
        out = []
        for stmt in body:
            out.extend(self._rewrite_stmt(stmt))
        return out

    def _rewrite_stmt(self, stmt):
        if isinstance(stmt, ast.Assign):
            return self._rewrite_assign(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.fn_depth += 1
            stmt.body = self.rewrite_block(stmt.body)
            self.fn_depth -= 1
            return [stmt]
        if isinstance(stmt, ast.For):
            stmt.body = self.rewrite_block(stmt.body)
            stmt.orelse = self.rewrite_block(stmt.orelse)
            return [stmt]
        if isinstance(stmt, ast.If):
            stmt.body = self.rewrite_block(stmt.body)
            stmt.orelse = self.rewrite_block(stmt.orelse)
            return [stmt]
        if isinstance(stmt, ast.While):
            stmt.body = self.rewrite_block(stmt.body)
            stmt.orelse = self.rewrite_block(stmt.orelse)
            return [stmt]
        if isinstance(stmt, ast.Return):
            # Only the outermost return is the graph-emit point. Inner-function
            # returns just pass through (any DSL Calls were lifted in pre-pass).
            if self.fn_depth > 0:
                return [stmt]
            return self._rewrite_return(stmt)
        return [stmt]

    def _rewrite_assign(self, stmt):
        if not (len(stmt.targets) == 1
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)):
            return [stmt]
        fname = stmt.value.func.id
        if fname not in _DSL_NAMES:
            return [stmt]
        target = stmt.targets[0]

        if fname in _MULTI_OUTPUT:
            return self._rewrite_multi(target, stmt.value, fname)

        assert isinstance(target, ast.Name), (
            f"DSL call {fname!r} produces a single output but is assigned to "
            f"a non-Name target: {ast.dump(target)}"
        )
        return _DISPATCH[fname](self, target.id, stmt.value)

    def _rewrite_multi(self, target, call, fname):
        n_node = _arg(call, 1 if fname != "flat_partition" else 2,
                      "n" if fname != "flat_partition" else "n")
        assert n_node is not None, f"{fname}: missing 'n' argument"
        n_src = _src(n_node)
        node_var = self.fresh(fname)

        if fname == "broadcast":
            x = _src(_arg(call, 0, "x"))
            ctor = (f"{node_var} = Broadcast(graph, {x}, num_consumers={n_src})\n")
        elif fname == "parallelize":
            x = _src(_arg(call, 0, "x"))
            ctor = (
                f"{node_var} = Parallelize(graph, {x}, "
                f"parallelize_rank={x}.stream.rank, num_consumers={n_src})\n"
            )
        else:  # flat_partition
            x = _src(_arg(call, 0, "x"))
            ctrl = _src(_arg(call, 1, "control"))
            assert ctrl in self.select_gen_vars, (
                f"flat_partition: control argument {ctrl!r} must be assigned "
                f"from select_gen(...) earlier in the function"
            )
            ctor = (
                f"{node_var} = FlatPartition(graph, {x}, control={ctrl}, "
                f"partition_rank=0, switch_cycles=[1] * {n_src}, "
                f"write_back_mu=False, num_consumers={n_src})\n"
            )

        # Bind the multi-output to the user's target. Three forms:
        #   xs = parallelize(x, n)              -> Name target
        #   a, b = parallelize(x, 2)            -> Tuple target
        #   (a, b) = parallelize(x, 2)          -> same
        if isinstance(target, ast.Name):
            bind = (f"{target.id} = "
                    f"[({node_var}, _i) for _i in range({n_src})]\n")
        elif isinstance(target, (ast.Tuple, ast.List)):
            assert all(isinstance(e, ast.Name) for e in target.elts), (
                f"{fname}: multi-output unpack target must be plain names"
            )
            bind = "".join(
                f"{e.id} = ({node_var}, {i})\n"
                for i, e in enumerate(target.elts)
            )
        else:
            raise AssertionError(
                f"{fname}: unsupported assignment target {ast.dump(target)}"
            )

        return _block(ctor + bind)

    def _rewrite_return(self, stmt):
        # Two supported forms at the end of tiled_reference:
        #   return offchip_store(x)
        #   return out                      (out was assigned earlier)
        if (isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "offchip_store"):
            x = _src(stmt.value.args[0])
            store_var = self.fresh("store")
            return _block(
                f"{store_var} = OffChipStore(graph, {x}, par_dispatch=1)\n"
                f"graph = infer_broadcast(graph)\n"
                f"return graph, {store_var}\n"
            )
        out = _src(stmt.value)
        return _block(
            f"graph = infer_broadcast(graph)\n"
            f"return graph, {out}\n"
        )


# ---------------------------------------------------------------------------
# Per-DSL-fn handlers (single-output only; multi-output is handled in _State)
# ---------------------------------------------------------------------------

def _h_offchip_load(state, target, call):
    underlying = _src(_arg(call, 0, "underlying"))
    stride     = _src(_arg(call, 1, "stride"))
    out_shape  = _src(_arg(call, 2, "out_shape_tiled"))
    tile_row   = _src(_arg(call, 3, "tile_row"))
    tile_col   = _src(_arg(call, 4, "tile_col"))
    transposed = _arg(call, 5, "transposed")
    extra = f", transposed={_src(transposed)}" if transposed is not None else ""
    return _block(
        f"{target} = LinearOffChipLoad({underlying}, stride={stride}, "
        f"out_shape_tiled={out_shape}, tile_row={tile_row}, tile_col={tile_col}, "
        f"par_dispatch=1{extra})\n"
        f"graph.add_node({target})\n"
    )


def _h_offchip_load_ref(state, target, call):
    ref        = _src(_arg(call, 0, "ref"))
    underlying = _src(_arg(call, 1, "underlying"))
    stride     = _src(_arg(call, 2, "stride"))
    out_shape  = _src(_arg(call, 3, "out_shape_tiled"))
    tile_row   = _src(_arg(call, 4, "tile_row"))
    tile_col   = _src(_arg(call, 5, "tile_col"))
    transposed = _arg(call, 6, "transposed")
    extra = f", transposed={_src(transposed)}" if transposed is not None else ""
    return _block(
        f"{target} = LinearOffChipLoadRef(graph, ref={ref}, "
        f"underlying={underlying}, stride={stride}, out_shape_tiled={out_shape}, "
        f"tile_row={tile_row}, tile_col={tile_col}, par_dispatch=1{extra})\n"
    )


def _h_random_offchip_load(state, target, call):
    underlying = _src(_arg(call, 0, "underlying"))
    raddr      = _src(_arg(call, 1, "raddr"))
    tile_row   = _src(_arg(call, 2, "tile_row"))
    tile_col   = _src(_arg(call, 3, "tile_col"))
    transposed = _arg(call, 4, "transposed")
    extra = f", transposed={_src(transposed)}" if transposed is not None else ""
    return _block(
        f"{target} = RandomOffChipLoad(graph, underlying={underlying}, "
        f"raddr={raddr}, tile_row={tile_row}, tile_col={tile_col}{extra})\n"
    )


def _h_select_gen(state, target, call):
    underlying  = _src(_arg(call, 0, "underlying"))
    is_multihot = _src(_arg(call, 1, "is_multihot"))
    n           = _src(_arg(call, 2, "n"))
    state.select_gen_vars.add(target)
    return _block(
        f"{target} = SelectGen(is_multihot={is_multihot}, "
        f"tensor={underlying}, n={n})\n"
        f"graph.add_node({target})\n"
    )


def _h_metadata_gen(state, target, call):
    tensor = _src(_arg(call, 0, "tensor"))
    return _block(
        f"{target} = MetadataGen(tensor={tensor})\n"
        f"graph.add_node({target})\n"
    )


def _h_cache_read_addr_gen(state, target, call):
    idx        = _src(_arg(call, 0, "idx"))
    seq_len    = _src(_arg(call, 1, "seq_len"))
    row_offset = _src(_arg(call, 2, "row_offset"))
    return _block(
        f"{target} = CacheReadAddrGen(graph, {idx}, {seq_len}, {row_offset})\n"
    )


def _h_filter_last_tile(state, target, call):
    seq_len = _src(_arg(call, 0, "seq_len"))
    return _block(f"{target} = FilterLastTile(graph, {seq_len})\n")


def _make_binary_map(map_class):
    def handler(state, target, call):
        a = _src(_arg(call, 0, "a"))
        b = _src(_arg(call, 1, "b"))
        if map_class == "Matmul":
            wt = _arg(call, 2, "weight_transposed")
            wt_s = f"weight_transposed={_src(wt)}" if wt is not None else ""
            fn_str = f"map_fn.Matmul({wt_s})"
        else:
            fn_str = f"map_fn.{map_class}()"
        return _block(
            f"{target} = BinaryMap(graph, {a}, {b}, fn={fn_str}, "
            f"write_back_mu=False)\n"
        )
    return handler


def _make_unary_map(map_class):
    def handler(state, target, call):
        x = _src(_arg(call, 0, "x"))
        if map_class in ("MulImmediate", "AddImmediate", "SubImmediate"):
            c = _src(_arg(call, 1, "constant"))
            fn_str = f"map_fn.{map_class}({c})"
        else:
            fn_str = f"map_fn.{map_class}()"
        return _block(
            f"{target} = UnaryMap(graph, {x}, fn={fn_str}, "
            f"write_back_mu=False)\n"
        )
    return handler


def _make_accum(accum_class, mode):
    def handler(state, target, call):
        x = _src(_arg(call, 0, "x"))
        rank = _arg_or_default(call, 1, "rank", "1")
        return _block(
            f"{target} = Accum(graph, {x}, "
            f"output_stream_dtype=_dsl2step_out_tile({x}, {mode!r}, {rank}), "
            f"fn=accum_fn.{accum_class}(), init_fn=_dsl2step_init({x}), "
            f"accum_rank={rank}, write_back_mu=False)\n"
        )
    return handler


def _h_binary_map_accum(state, target, call):
    a = _src(_arg(call, 0, "a"))
    b = _src(_arg(call, 1, "b"))
    rank = _arg_or_default(call, 2, "rank", "1")
    wt = _arg(call, 3, "weight_transposed")
    wt_s = f"weight_transposed={_src(wt)}" if wt is not None else ""
    return _block(
        f"{target} = BinaryMapAccum(graph, {a}, {b}, "
        f"fn=map_accum_fn.Matmul({wt_s}), init_fn=_dsl2step_init({a}), "
        f"rank={rank}, write_back_mu=False)\n"
    )


def _h_promote(state, target, call):
    x = _src(_arg(call, 0, "x"))
    rank = _arg_or_default(call, 1, "rank", "1")
    return _block(f"{target} = Promote(graph, {x}, promote_rank={rank})\n")


def _h_promote_outer(state, target, call):
    x = _src(_arg(call, 0, "x"))
    return _block(f"{target} = PromoteOuter(graph, {x})\n")


def _h_flatten(state, target, call):
    x        = _src(_arg(call, 0, "x"))
    min_rank = _src(_arg(call, 1, "min_rank"))
    max_rank = _src(_arg(call, 2, "max_rank"))
    return _block(
        f"{target} = Flatten(graph, {x}, min_rank={min_rank}, "
        f"max_rank={max_rank})\n"
    )


def _h_reshape_stream(state, target, call):
    x          = _src(_arg(call, 0, "x"))
    chunk_size = _src(_arg(call, 1, "chunk_size"))
    rank       = _arg_or_default(call, 2, "rank", "0")
    return _block(
        f"{target} = Reshape(graph, {x}, chunk_size={chunk_size}, "
        f"reshape_rank={rank}, write_back_mu=False)\n"
    )


def _h_reshape_pad_stream(state, target, call):
    x          = _src(_arg(call, 0, "x"))
    chunk_size = _src(_arg(call, 1, "chunk_size"))
    rank       = _arg_or_default(call, 2, "reshape_rank", "0")
    return _block(
        f"{target} = ReshapePadStream(graph, {x}, chunk_size={chunk_size}, "
        f"reshape_rank={rank}, write_back_mu=False, "
        f"have_pad_stream=False, pad_fn=None)\n"
    )


def _h_expand_ref(state, target, call):
    x           = _src(_arg(call, 0, "x"))
    ref         = _src(_arg(call, 1, "ref"))
    expand_rank = _src(_arg(call, 2, "expand_rank"))
    return _block(
        f"{target} = ExpandRef(graph, {x}, ref={ref}, "
        f"expand_rank={expand_rank})\n"
    )


def _h_repeat_ref(state, target, call):
    x   = _src(_arg(call, 0, "x"))
    ref = _src(_arg(call, 1, "ref"))
    return _block(f"{target} = RepeatRef(graph, {x}, ref={ref})\n")


def _h_repeat_static(state, target, call):
    x      = _src(_arg(call, 0, "x"))
    factor = _src(_arg(call, 1, "factor"))
    return _block(
        f"{target} = RepeatStatic(graph, {x}, repeat_factor={factor})\n"
    )


def _h_streamify(state, target, call):
    x       = _src(_arg(call, 0, "x"))
    factors = _src(_arg(call, 1, "repeat_factors"))
    rank    = _arg_or_default(call, 2, "rank", "0")
    return _block(
        f"{target} = Streamify(graph, {x}, repeat_factors=list({factors}), "
        f"rank={rank})\n"
    )


def _h_retile_streamify(state, target, call):
    x         = _src(_arg(call, 0, "x"))
    chunk     = _src(_arg(call, 1, "chunk"))
    split_row = _arg_or_default(call, 2, "split_row", "True")
    return _block(
        f"{target} = RetileStreamify(graph, {x}, split_row={split_row}, "
        f"chunk={chunk})\n"
    )


def _h_static_reassemble(state, target, call):
    inputs = _src(_arg(call, 0, "inputs"))
    return _block(
        f"{target} = StaticReassemble(graph, inputs={inputs}, "
        f"merge_rank=({inputs})[0].stream.rank)\n"
    )


def _h_flat_reassemble(state, target, call):
    inputs  = _src(_arg(call, 0, "inputs"))
    control = _src(_arg(call, 1, "control"))
    assert control in state.select_gen_vars, (
        f"flat_reassemble: control argument {control!r} must be assigned "
        f"from select_gen(...) earlier in the function"
    )
    return _block(
        f"{target} = FlatReassemble(graph, inputs={inputs}, control={control}, "
        f"reassemble_rank=0, switch_cycles=[1] * len({inputs}), "
        f"write_back_mu=False)\n"
    )


def _h_offchip_store(state, target, call):
    x = _src(_arg(call, 0, "x"))
    return _block(f"{target} = OffChipStore(graph, {x}, par_dispatch=1)\n")


def _h_random_offchip_store(state, target, call):
    underlying = _src(_arg(call, 0, "underlying"))
    wdata      = _src(_arg(call, 1, "wdata"))
    waddr      = _src(_arg(call, 2, "waddr"))
    tile_row   = _src(_arg(call, 3, "tile_row"))
    tile_col   = _src(_arg(call, 4, "tile_col"))
    return _block(
        f"{target} = RandomOffChipStore(graph, underlying={underlying}, "
        f"wdata={wdata}, waddr={waddr}, tile_row={tile_row}, "
        f"tile_col={tile_col})\n"
    )


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    "offchip_load":         _h_offchip_load,
    "offchip_load_ref":     _h_offchip_load_ref,
    "random_offchip_load":  _h_random_offchip_load,
    "select_gen":           _h_select_gen,
    "metadata_gen":         _h_metadata_gen,
    "cache_read_addr_gen":  _h_cache_read_addr_gen,
    "filter_last_tile":     _h_filter_last_tile,
    "binary_map_accum":     _h_binary_map_accum,
    "promote":              _h_promote,
    "promote_outer":        _h_promote_outer,
    "flatten":              _h_flatten,
    "reshape_stream":       _h_reshape_stream,
    "reshape_pad_stream":   _h_reshape_pad_stream,
    "expand_ref":           _h_expand_ref,
    "repeat_ref":           _h_repeat_ref,
    "repeat_static":        _h_repeat_static,
    "streamify":            _h_streamify,
    "retile_streamify":     _h_retile_streamify,
    "static_reassemble":    _h_static_reassemble,
    "flat_reassemble":      _h_flat_reassemble,
    "offchip_store":        _h_offchip_store,
    "random_offchip_store": _h_random_offchip_store,
    # broadcast / parallelize / flat_partition: handled by _State._rewrite_multi
    "broadcast":            None,
    "parallelize":          None,
    "flat_partition":       None,
}
for _name, _cls in _BINARY_MAP.items():
    _DISPATCH[_name] = _make_binary_map(_cls)
for _name, _cls in _UNARY_MAP.items():
    _DISPATCH[_name] = _make_unary_map(_cls)
for _name, (_cls, _mode) in _ACCUM_MAP.items():
    _DISPATCH[_name] = _make_accum(_cls, _mode)

_DSL_NAMES.update(_DISPATCH.keys())
