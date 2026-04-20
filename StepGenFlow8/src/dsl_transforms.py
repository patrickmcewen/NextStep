"""Deterministic AST transforms on DSL code produced by the refactor_final pass.

These run between refactor_final and translate to canonicalize patterns that
the LLM writes naturally but that don't translate well to STeP IR.
"""

import ast
from typing import Optional

# Parameter order for offchip_load / offchip_load_ref
_LOAD_PARAMS = ["underlying", "stride", "out_shape_tiled",
                "tile_row", "tile_col", "transposed"]
_LOAD_REF_PARAMS = ["ref", "underlying", "stride", "out_shape_tiled",
                    "tile_row", "tile_col", "transposed"]


def _extract_arg(call: ast.Call, param_names: list[str],
                 name: str) -> Optional[ast.expr]:
    """Get an argument from a Call node by keyword name or positional index."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    pos = param_names.index(name)
    if pos < len(call.args):
        return call.args[pos]
    return None


def _is_all_ones_tuple(node: ast.expr) -> bool:
    """True if node is a tuple literal whose elements are all the integer 1."""
    if not isinstance(node, ast.Tuple):
        return False
    return (len(node.elts) > 0
            and all(isinstance(e, ast.Constant) and e.value == 1
                    for e in node.elts))


def _name_read_in(name: str, node: ast.AST) -> bool:
    """True if `name` appears in Load context anywhere in the subtree."""
    for child in ast.walk(node):
        if (isinstance(child, ast.Name)
                and child.id == name
                and isinstance(child.ctx, ast.Load)):
            return True
    return False


def _name_written_in(name: str, node: ast.AST) -> bool:
    """True if `name` appears in Store context anywhere in the subtree."""
    for child in ast.walk(node):
        if (isinstance(child, ast.Name)
                and child.id == name
                and isinstance(child.ctx, ast.Store)):
            return True
    return False


def _build_load_ref_call(ref_node: ast.expr,
                         load_call: ast.Call) -> ast.Call:
    """Build an offchip_load_ref(...) call from a ref node and an offchip_load call.

    Preserves the original stride and out_shape_tiled from the offchip_load call.
    STeP IR needs these values even when they're all-ones — stripping them to ()
    breaks LinearOffChipLoadRef which requires non-empty out_shape_tiled.
    """
    underlying = _extract_arg(load_call, _LOAD_PARAMS, "underlying")
    stride = _extract_arg(load_call, _LOAD_PARAMS, "stride")
    out_shape_tiled = _extract_arg(load_call, _LOAD_PARAMS, "out_shape_tiled")
    tile_row = _extract_arg(load_call, _LOAD_PARAMS, "tile_row")
    tile_col = _extract_arg(load_call, _LOAD_PARAMS, "tile_col")
    transposed = _extract_arg(load_call, _LOAD_PARAMS, "transposed")

    assert underlying is not None and tile_row is not None and tile_col is not None
    assert stride is not None and out_shape_tiled is not None

    keywords = [
        ast.keyword(arg="stride", value=stride),
        ast.keyword(arg="out_shape_tiled", value=out_shape_tiled),
        ast.keyword(arg="tile_row", value=tile_row),
        ast.keyword(arg="tile_col", value=tile_col),
    ]
    if transposed is not None:
        keywords.append(ast.keyword(arg="transposed", value=transposed))

    return ast.Call(
        func=ast.Name(id="offchip_load_ref", ctx=ast.Load()),
        args=[ref_node, underlying],
        keywords=keywords,
    )


def fuse_load_ref(code: str) -> tuple[str, list[str]]:
    """Replace offchip_load + repeat_ref/expand_ref with offchip_load_ref.

    Detects patterns where:
      1. var = offchip_load(..., out_shape_tiled=(1,...,1), ...)
      2. var = repeat_ref(var, ref)   OR   var = expand_ref(var, ref)
    and the loaded value isn't used between (1) and (2).

    Replaces with:
      var = offchip_load_ref(ref, underlying, stride=(), out_shape_tiled=(),
                             tile_row=..., tile_col=...)

    This is safe because all-ones out_shape_tiled loads exactly one tile
    (same as empty out_shape_tiled), and offchip_load_ref expands to match
    the ref's stream shape — identical to what repeat_ref/expand_ref did.

    Returns (transformed_code, list_of_fusion_descriptions).
    """
    tree = ast.parse(code)
    fusions: list[str] = []

    def _process_block(stmts: list[ast.stmt]):
        # pending_loads: var_name -> (stmt_index, load Call node, Assign node)
        pending: dict[str, tuple[int, ast.Call, ast.Assign]] = {}
        removals: set[int] = set()

        for i, stmt in enumerate(stmts):
            # Recurse into nested blocks
            for block_attr in ("body", "orelse", "handlers", "finalbody"):
                block = getattr(stmt, block_attr, None)
                if isinstance(block, list) and block:
                    _process_block(block)

            # --- Pattern part 1: var = offchip_load(...) with all-ones out_shape ---
            if (isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id == "offchip_load"):

                out_shape = _extract_arg(stmt.value, _LOAD_PARAMS, "out_shape_tiled")
                if out_shape is not None and _is_all_ones_tuple(out_shape):
                    var = stmt.targets[0].id
                    pending[var] = (i, stmt.value, stmt)
                continue

            # --- Pattern part 2: var = repeat_ref(var, ref) / expand_ref(var, ref) ---
            if (isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Call)
                    and isinstance(stmt.value.func, ast.Name)
                    and stmt.value.func.id in ("repeat_ref", "expand_ref")):

                call = stmt.value
                var = stmt.targets[0].id

                if (var in pending
                        and len(call.args) >= 2
                        and isinstance(call.args[0], ast.Name)
                        and call.args[0].id == var):

                    load_idx, load_call, load_stmt = pending[var]
                    ref_node = call.args[1]

                    # Verify var isn't read or overwritten between load and here
                    safe = True
                    for j in range(load_idx + 1, i):
                        if j in removals:
                            continue
                        if (_name_read_in(var, stmts[j])
                                or _name_written_in(var, stmts[j])):
                            safe = False
                            break

                    if safe:
                        # Rewrite the load assignment to offchip_load_ref
                        load_stmt.value = _build_load_ref_call(ref_node, load_call)
                        removals.add(i)
                        fusions.append(
                            f"{var}: offchip_load + {call.func.id} "
                            f"-> offchip_load_ref"
                        )
                        del pending[var]
                        continue

            # --- Invalidate pending loads if var is touched by this statement ---
            for var in list(pending):
                if _name_read_in(var, stmt) or _name_written_in(var, stmt):
                    del pending[var]

        # Remove fused repeat_ref/expand_ref statements (reverse order)
        for idx in sorted(removals, reverse=True):
            del stmts[idx]

    # Process all function definitions in the module
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            _process_block(node.body)

    ast.fix_missing_locations(tree)
    return ast.unparse(tree), fusions
