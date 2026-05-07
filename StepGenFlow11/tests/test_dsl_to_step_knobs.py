"""Translator forwards compute_bw / par_dispatch from DSL kwargs to STeP ctor kwargs."""

import ast

from src.dsl_to_step import translate


def _find_call(tree, fn_name):
    """First Call node whose func.id == fn_name (depth-first)."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == fn_name):
            return node
    return None


def _kwarg_value(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return ast.unparse(kw.value)
    return None


_BASE_TILED_REF = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4{a_kwargs})
    b = offchip_load(tensors["B"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4)
    c = binary_matmul(a, b{matmul_kwargs})
    return offchip_store(c{store_kwargs})
'''


def _render(a_kwargs="", matmul_kwargs="", store_kwargs=""):
    return _BASE_TILED_REF.format(
        a_kwargs=a_kwargs,
        matmul_kwargs=matmul_kwargs,
        store_kwargs=store_kwargs,
    )


def test_binary_matmul_forwards_compute_bw():
    out = translate(_render(matmul_kwargs=", compute_bw=8"))
    tree = ast.parse(out)
    call = _find_call(tree, "BinaryMap")
    assert call is not None, f"BinaryMap not found in:\n{out}"
    assert _kwarg_value(call, "compute_bw") == "8"


def test_offchip_load_forwards_par_dispatch():
    out = translate(_render(a_kwargs=", par_dispatch=4"))
    tree = ast.parse(out)
    call = _find_call(tree, "LinearOffChipLoad")
    assert call is not None, f"LinearOffChipLoad not found in:\n{out}"
    assert _kwarg_value(call, "par_dispatch") == "4"


def test_offchip_store_forwards_par_dispatch():
    out = translate(_render(store_kwargs=", par_dispatch=2"))
    tree = ast.parse(out)
    call = _find_call(tree, "OffChipStore")
    assert call is not None, f"OffChipStore not found in:\n{out}"
    assert _kwarg_value(call, "par_dispatch") == "2"


def test_default_compute_bw_is_one_when_omitted():
    """Backwards compat: DSL without new kwargs translates with default of 1."""
    out = translate(_render())
    tree = ast.parse(out)
    matmul = _find_call(tree, "BinaryMap")
    assert _kwarg_value(matmul, "compute_bw") == "1"
    load = _find_call(tree, "LinearOffChipLoad")
    assert _kwarg_value(load, "par_dispatch") == "1"
    store = _find_call(tree, "OffChipStore")
    assert _kwarg_value(store, "par_dispatch") == "1"


def test_unary_map_forwards_compute_bw():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4)
    b = unary_silu(a, compute_bw=4)
    return offchip_store(b)
'''
    out = translate(src)
    tree = ast.parse(out)
    call = _find_call(tree, "UnaryMap")
    assert _kwarg_value(call, "compute_bw") == "4"


def test_accum_forwards_compute_bw():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(3,),
                      tile_row=4, tile_col=4)
    b = accum_add(a, rank=1, compute_bw=2)
    return offchip_store(b)
'''
    out = translate(src)
    tree = ast.parse(out)
    call = _find_call(tree, "Accum")
    assert _kwarg_value(call, "compute_bw") == "2"
