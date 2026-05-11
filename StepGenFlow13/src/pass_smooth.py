"""Smoothing pass — inline pass1_independent children into their parents,
inserting layout bridges across boundaries.

In independent pass1 every node is verified standalone: each input is RAW
(off-chip) and is produced on-chip via a single ``offchip_load``; each
output is returned as a tile-stream tensor whose flattened form equals
the vanilla pytorch output flattened.

The smoother composes a multi-node tree into one fused root DSL by
walking the tree in post-order and, for each non-leaf, replacing every
``child_name(args)`` placeholder call site with the child's verified
DSL body, with the boundary loads stripped and (if the parent's on-chip
layout for an arg differs from the child's expected layout) an
off-chip round-trip bridge inserted::

    parent_value          # parent's on-chip tile-stream value
    -> offchip_store(...) # back to a vanilla off-chip tensor
    -> offchip_load(stride=child_stride, out_shape_tiled=child_O,
                    tile_row=child_Tr, tile_col=child_Tc)
                          # the child's preferred layout

When the parent's tile-stream shape equals the child's offchip_load output
shape (exact match), the bridge is a no-op rename — the parent's value is
substituted directly.

The terminal ``offchip_store(...)`` of the child (if any) is stripped so
its on-chip output flows back into the parent's body. The substitution is
purely textual at the AST level; no runtime evaluation happens here.

Constraints (v1):
  * Each non-leaf child must define a single function
    ``def <child_name>(arg1, arg2, ...): ...`` with the
    ``_validate_load_once`` invariant already enforced upstream.
  * Each tensor arg in the child must be loaded with exactly one
    ``offchip_load(arg, ...)`` call whose result is bound to a
    distinct local name (single Assign target).
  * Local names in the child's body are renamed during inlining to
    avoid collisions with parent locals.
  * Child returns either a single expression or a tuple of expressions;
    if the body's last statement is ``return offchip_store(expr)`` or
    ``return offchip_store(expr_0), offchip_store(expr_1), ...``, the
    ``offchip_store`` calls are stripped so the on-chip values flow back.
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# AST inspection helpers
# ---------------------------------------------------------------------------


def _get_function(tree: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(
        f"_get_function: no `def {function_name}(...)` found at module top level")


def _call_name(call: ast.Call) -> str | None:
    """Return the bare name of the callable, or None if not a Name call."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


@dataclass(frozen=True)
class _LoadParams:
    """Extracted offchip_load parameters for one tensor arg."""
    bound_name: str                       # the variable the load result is bound to
    stride: ast.expr                      # AST expr for stride tuple
    out_shape_tiled: ast.expr             # AST expr for out_shape_tiled tuple
    tile_row: ast.expr                    # AST expr for tile_row int
    tile_col: ast.expr                    # AST expr for tile_col int
    transposed: ast.expr | None = None    # None or an AST expr
    par_dispatch: ast.expr | None = None  # None or an AST expr


def _kw(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _arg_or_kw(call: ast.Call, pos: int, name: str) -> ast.expr | None:
    if pos < len(call.args):
        return call.args[pos]
    return _kw(call, name)


def _extract_offchip_loads(child_fn: ast.FunctionDef,
                            arg_names: tuple[str, ...]) -> dict[str, _LoadParams]:
    """For each ``arg_name`` in ``arg_names``, find the single
    ``<bound> = offchip_load(<arg>, ...)`` assignment in the child's
    body and extract its layout parameters.

    Returns ``{arg_name: _LoadParams}``. Asserts loudly if an arg isn't
    loaded exactly once (the load-once validator should have already
    caught this upstream, so a failure here indicates either a bypassed
    validator or unsupported syntax).
    """
    arg_set = set(arg_names)
    loads: dict[str, _LoadParams] = {}

    for stmt in child_fn.body:
        # Match `<bound> = offchip_load(<arg>, ...)`.
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call)):
            continue
        call = stmt.value
        if _call_name(call) != "offchip_load":
            continue
        if not call.args:
            continue
        first = call.args[0]
        # We only handle the bare-name form here — list-of-tensors loads
        # use `arg[i]` and are inside a list comprehension or for loop;
        # those are handled separately (see _extract_list_loads).
        if not (isinstance(first, ast.Name) and first.id in arg_set):
            continue
        arg = first.id
        bound = stmt.targets[0].id

        stride = _arg_or_kw(call, 1, "stride")
        out_shape_tiled = _arg_or_kw(call, 2, "out_shape_tiled")
        tile_row = _arg_or_kw(call, 3, "tile_row")
        tile_col = _arg_or_kw(call, 4, "tile_col")
        transposed = _arg_or_kw(call, 5, "transposed")
        par_dispatch = _kw(call, "par_dispatch")

        assert stride is not None, (
            f"_extract_offchip_loads: offchip_load for arg {arg!r} missing stride")
        assert out_shape_tiled is not None, (
            f"_extract_offchip_loads: offchip_load for arg {arg!r} missing out_shape_tiled")
        assert tile_row is not None, (
            f"_extract_offchip_loads: offchip_load for arg {arg!r} missing tile_row")
        assert tile_col is not None, (
            f"_extract_offchip_loads: offchip_load for arg {arg!r} missing tile_col")

        if arg in loads:
            raise AssertionError(
                f"_extract_offchip_loads: arg {arg!r} loaded more than once "
                f"in child function — load-once validator should have caught "
                f"this")
        loads[arg] = _LoadParams(
            bound_name=bound,
            stride=stride,
            out_shape_tiled=out_shape_tiled,
            tile_row=tile_row,
            tile_col=tile_col,
            transposed=transposed,
            par_dispatch=par_dispatch,
        )

    missing = arg_set - set(loads.keys())
    assert not missing, (
        f"_extract_offchip_loads: args {sorted(missing)} are not loaded by "
        f"any single `offchip_load(<arg>, ...)` call in the child function. "
        f"List-of-tensor args are not supported by the smoother's v1 inliner; "
        f"add support if needed.")
    return loads


def _strip_terminal_offchip_store(child_fn: ast.FunctionDef) -> ast.expr:
    """Find the child's terminal ``return ...`` and, if it contains
    ``offchip_store(expr)`` calls (single or tuple), strip them so the
    inlined return value is the on-chip tile-stream expression.

    Returns a single ``ast.expr`` representing the stripped return value
    (a single Name/Call/Tuple/etc.). The caller is responsible for
    binding it to a fresh name in the inlined body.
    """
    last = child_fn.body[-1]
    assert isinstance(last, ast.Return), (
        f"_strip_terminal_offchip_store: child function's last statement "
        f"must be `return ...`, got {type(last).__name__}")
    val = last.value
    assert val is not None, (
        "_strip_terminal_offchip_store: bare `return` is not allowed")

    def _strip_one(expr: ast.expr) -> ast.expr:
        if isinstance(expr, ast.Call) and _call_name(expr) == "offchip_store":
            assert expr.args, (
                "_strip_terminal_offchip_store: offchip_store call has no positional arg")
            return expr.args[0]
        return expr

    if isinstance(val, ast.Tuple):
        return ast.Tuple(elts=[_strip_one(e) for e in val.elts], ctx=ast.Load())
    return _strip_one(val)


# ---------------------------------------------------------------------------
# Renaming helpers (avoid name collisions between parent and child locals)
# ---------------------------------------------------------------------------


class _Renamer(ast.NodeTransformer):
    """Rename selected ast.Name nodes in-place using a mapping."""
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if node.id in self.mapping:
            return ast.copy_location(
                ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node


def _collect_assigned_names(fn: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.Assign):
            for t in sub.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                elif isinstance(t, ast.Tuple):
                    for el in t.elts:
                        if isinstance(el, ast.Name):
                            names.add(el.id)
    return names


# ---------------------------------------------------------------------------
# Bridge construction
# ---------------------------------------------------------------------------


def _make_bridge(parent_var: str, bridge_var: str,
                 store_var: str, params: _LoadParams) -> list[ast.stmt]:
    """Construct the AST for a parent->child arg bridge:

        <store_var>  = offchip_store(<parent_var>)
        <bridge_var> = offchip_load(<store_var>,
                                     stride=<params.stride>,
                                     out_shape_tiled=<params.out_shape_tiled>,
                                     tile_row=<params.tile_row>,
                                     tile_col=<params.tile_col>,
                                     transposed=<params.transposed or False>)
    """
    store_call = ast.Call(
        func=ast.Name(id="offchip_store", ctx=ast.Load()),
        args=[ast.Name(id=parent_var, ctx=ast.Load())],
        keywords=[],
    )
    store_assign = ast.Assign(
        targets=[ast.Name(id=store_var, ctx=ast.Store())],
        value=store_call,
    )

    load_kwargs = [
        ast.keyword(arg="stride", value=params.stride),
        ast.keyword(arg="out_shape_tiled", value=params.out_shape_tiled),
        ast.keyword(arg="tile_row", value=params.tile_row),
        ast.keyword(arg="tile_col", value=params.tile_col),
    ]
    if params.transposed is not None:
        load_kwargs.append(ast.keyword(arg="transposed", value=params.transposed))

    load_call = ast.Call(
        func=ast.Name(id="offchip_load", ctx=ast.Load()),
        args=[ast.Name(id=store_var, ctx=ast.Load())],
        keywords=load_kwargs,
    )
    load_assign = ast.Assign(
        targets=[ast.Name(id=bridge_var, ctx=ast.Store())],
        value=load_call,
    )

    return [
        ast.fix_missing_locations(store_assign),
        ast.fix_missing_locations(load_assign),
    ]


# ---------------------------------------------------------------------------
# Inliner
# ---------------------------------------------------------------------------


@dataclass
class _InlineState:
    parent_locals: set[str] = field(default_factory=set)
    counter: int = 0

    def fresh(self, base: str) -> str:
        self.counter += 1
        name = f"_smooth_{base}_{self.counter}"
        while name in self.parent_locals:
            self.counter += 1
            name = f"_smooth_{base}_{self.counter}"
        self.parent_locals.add(name)
        return name


def _inline_child_call(
    *,
    parent_call: ast.Call,
    parent_assign_targets: list[ast.expr] | None,
    child_fn: ast.FunctionDef,
    state: _InlineState,
) -> list[ast.stmt]:
    """Replace one ``<targets> = child_name(parent_args)`` call site (or
    bare-call statement) with the inlined child body + bridges.

    ``parent_call``: the AST Call node at the call site.
    ``parent_assign_targets``: the target list of the surrounding Assign,
        or ``None`` if the call is an Expression statement (rare in DSL —
        children always return values that the parent uses).
    ``child_fn``: the child's parsed FunctionDef.
    ``state``: name-collision tracker.

    Returns a list of replacement statements (in source order).
    """
    arg_names = tuple(a.arg for a in child_fn.args.args)
    assert len(parent_call.args) == len(arg_names), (
        f"_inline_child_call: parent passed {len(parent_call.args)} args to "
        f"child {child_fn.name!r} but child declares {len(arg_names)} "
        f"({arg_names})")
    assert not parent_call.keywords, (
        f"_inline_child_call: parent called child {child_fn.name!r} with "
        f"keyword args {[kw.arg for kw in parent_call.keywords]} — "
        f"independent-mode child placeholders take only positional args")

    # Extract child's offchip_load layouts per arg.
    loads = _extract_offchip_loads(child_fn, arg_names)

    # For each arg: insert a bridge that converts the parent's value to
    # the child's expected layout. Always round-trip via offchip_store +
    # offchip_load (v1: correctness-first; an exact-match optimization
    # can be added later).
    bridge_stmts: list[ast.stmt] = []
    arg_to_bridge_var: dict[str, str] = {}
    for arg_name, parent_arg_expr in zip(arg_names, parent_call.args):
        params = loads[arg_name]
        # Bind the parent's expression to a temp name so we can pass a Name
        # into offchip_store (keeping the generated source readable).
        parent_tmp = state.fresh(f"{arg_name}_parent")
        bridge_stmts.append(ast.fix_missing_locations(ast.Assign(
            targets=[ast.Name(id=parent_tmp, ctx=ast.Store())],
            value=parent_arg_expr,
        )))
        store_var = state.fresh(f"{arg_name}_offchip")
        bridge_var = state.fresh(f"{arg_name}_bridged")
        bridge_stmts.extend(_make_bridge(parent_tmp, bridge_var, store_var, params))
        arg_to_bridge_var[arg_name] = bridge_var

    # Build the renamed child body.
    # 1) Drop the offchip_load assignment for each arg (its result is
    #    now the bridge's output bound to bridge_var).
    # 2) Rename child locals to avoid collisions with parent locals,
    #    and substitute each child's arg name and each child's loaded
    #    bound_name with the bridge_var.
    rename_map: dict[str, str] = {}
    for arg_name, params in loads.items():
        rename_map[arg_name] = arg_to_bridge_var[arg_name]
        rename_map[params.bound_name] = arg_to_bridge_var[arg_name]

    # Other child locals: rename to avoid collisions.
    child_locals = _collect_assigned_names(child_fn)
    for local in child_locals:
        if local in rename_map:
            continue
        rename_map[local] = state.fresh(f"{child_fn.name}_{local}")

    # Walk the child's body, applying renames + dropping offchip_load lines.
    drop_load_bound_names = {p.bound_name for p in loads.values()}
    new_body: list[ast.stmt] = []
    for stmt in child_fn.body[:-1]:
        # Drop `<bound> = offchip_load(<arg>, ...)` for boundary args.
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in drop_load_bound_names
                and isinstance(stmt.value, ast.Call)
                and _call_name(stmt.value) == "offchip_load"):
            continue
        renamed = _Renamer(rename_map).visit(stmt)
        ast.fix_missing_locations(renamed)
        new_body.append(renamed)

    # Terminal return: strip offchip_store + assign to parent's targets.
    return_value = _strip_terminal_offchip_store(child_fn)
    return_value = _Renamer(rename_map).visit(return_value)
    ast.fix_missing_locations(return_value)

    if parent_assign_targets is None:
        # Rare: the call is an expression statement. Just keep the body
        # and emit the value as an expression (no binding).
        new_body.append(ast.fix_missing_locations(ast.Expr(value=return_value)))
    else:
        # Bind the stripped return value to the parent's assign targets.
        new_body.append(ast.fix_missing_locations(ast.Assign(
            targets=parent_assign_targets,
            value=return_value,
        )))

    return bridge_stmts + new_body


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def smooth_compose(*, root_dsl: str, root_function_name: str,
                    children_dsls: dict[str, str]) -> str:
    """Inline every ``<targets> = child_name(args)`` call site in the
    parent's DSL with the child's body, inserting layout bridges. The
    parent here is the ``root_function_name`` function defined in
    ``root_dsl``; ``children_dsls`` maps each child function name to its
    own pass1 DSL source.

    For multi-level trees, call ``smooth_compose`` bottom-up: each
    non-leaf node's DSL is composed against its already-smoothed
    children, then the result becomes a child for that node's parent.

    Returns the parent's DSL with the children inlined and bridged.
    """
    parent_tree = ast.parse(root_dsl)
    parent_fn = _get_function(parent_tree, root_function_name)

    # Parse each child once.
    parsed_children: dict[str, ast.FunctionDef] = {}
    for child_name, child_src in children_dsls.items():
        ctree = ast.parse(child_src)
        parsed_children[child_name] = _get_function(ctree, child_name)

    state = _InlineState(parent_locals=_collect_assigned_names(parent_fn))

    new_body: list[ast.stmt] = []
    for stmt in parent_fn.body:
        # Match either:
        #   <targets> = <child_name>(args)
        #   <child_name>(args)        (rare expression statement)
        # ``<targets>`` may be a Name or a Tuple of Names. Anything else
        # is left untouched.
        replaced = None
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.value, ast.Call)
                and _call_name(stmt.value) in parsed_children):
            child_name = _call_name(stmt.value)
            tgt = stmt.targets[0]
            if isinstance(tgt, (ast.Name, ast.Tuple)):
                replaced = _inline_child_call(
                    parent_call=stmt.value,
                    parent_assign_targets=[tgt],
                    child_fn=parsed_children[child_name],
                    state=state,
                )
        elif (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
              and _call_name(stmt.value) in parsed_children):
            child_name = _call_name(stmt.value)
            replaced = _inline_child_call(
                parent_call=stmt.value,
                parent_assign_targets=None,
                child_fn=parsed_children[child_name],
                state=state,
            )

        if replaced is not None:
            new_body.extend(replaced)
        else:
            new_body.append(stmt)

    parent_fn.body = new_body
    ast.fix_missing_locations(parent_tree)

    # Render. Drop any `from ... import` of the children at the top of
    # the parent's module (independent-mode pass1 doesn't emit imports
    # but in case the LLM did, they'd reference no-longer-defined names).
    pruned: list[ast.stmt] = []
    for top in parent_tree.body:
        if isinstance(top, (ast.Import, ast.ImportFrom)):
            # Drop child-name re-imports; keep everything else.
            names_in_import = {n.name.rsplit(".", 1)[-1] for n in top.names}
            if names_in_import & set(parsed_children.keys()):
                continue
        pruned.append(top)
    parent_tree.body = pruned

    return ast.unparse(parent_tree)


def smooth_compose_tree(*, tree, pass1_dsls: dict[str, str],
                         root_function_name: str = "tiled_reference",
                         log: Callable[[str], None] = print) -> str:
    """Walk ``tree`` (a planner ``Tree``) bottom-up, smoothing each
    non-leaf against its (already-smoothed) children. Returns the fully
    inlined root DSL.

    ``pass1_dsls`` maps each ``node.path`` to the pass1_independent
    verified DSL string.
    """
    smoothed: dict[str, str] = {}

    for node in tree.iter_topological():  # post-order: children first
        node_dsl = pass1_dsls[node.path]
        if not node.children:
            # Leaf: nothing to inline.
            smoothed[node.path] = node_dsl
            continue
        # Non-leaf: compose this node against its smoothed children.
        children_dsls = {c.name: smoothed[c.path] for c in node.children}
        # The function name in node_dsl is `node.name` (per
        # build_pass1_independent_user_prompt's signature).
        function_name = node.name if node.path != "root" else root_function_name
        log(f"[smoother] inlining children {list(children_dsls)} into "
            f"node {node.path!r} (fn={function_name})")
        smoothed[node.path] = smooth_compose(
            root_dsl=node_dsl,
            root_function_name=function_name,
            children_dsls=children_dsls,
        )

    return smoothed[tree.root.path]
