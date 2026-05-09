import torch
from src.orchestrator import _pass2_compose_namespace


def test_compose_rebinds_child_to_verified_dsl():
    # Parent calls "double" as if it were a blackbox.
    parent_dsl = '''
def tiled_reference(dims, tensors):
    return double(tensors["x"], out_shapes=(tuple(tensors["x"].shape),))
'''
    # Child's verified DSL implements "double" properly.
    child_dsl = '''
def double(x, *, out_shapes, out_perms=None):
    return (x * 2).reshape(out_shapes[0])
'''
    ns = _pass2_compose_namespace(
        parent_dsl=parent_dsl,
        child_dsls_in_order=[child_dsl],
    )
    x = torch.arange(8, dtype=torch.float32)
    out = ns["tiled_reference"]({}, {"x": x})
    assert torch.equal(out, x * 2)


def test_compose_handles_grandchild_recursion():
    parent_dsl = '''
def tiled_reference(dims, tensors):
    return outer(tensors["x"], out_shapes=(tuple(tensors["x"].shape),))
'''
    middle_dsl = '''
def outer(x, *, out_shapes, out_perms=None):
    return inner(x, out_shapes=out_shapes) + 1
'''
    leaf_dsl = '''
def inner(x, *, out_shapes, out_perms=None):
    return (x * 10).reshape(out_shapes[0])
'''
    # Post-order: leaf (inner) before parent (outer).
    ns = _pass2_compose_namespace(
        parent_dsl=parent_dsl,
        child_dsls_in_order=[leaf_dsl, middle_dsl],
    )
    x = torch.arange(4, dtype=torch.float32)
    out = ns["tiled_reference"]({}, {"x": x})
    assert torch.equal(out, x * 10 + 1)


def test_compose_handles_same_name_parent_and_leaf():
    """Regression: planner can produce nested nodes that share a function
    name (e.g., a parent ``moe_dispatch`` whose child is also a leaf
    ``moe_dispatch``). Pass-1 may emit a parent that captures the leaf via
    ``_<name>_child = <name>`` before redefining the function. This works
    iff the leaf's DSL has been exec'd in the namespace before the
    parent's. Keying descendants by ``node.name`` (the old behaviour)
    silently dropped the leaf and produced ``NameError`` at parent-exec
    time."""
    leaf_dsl = '''
def moe_dispatch(x, *, out_shapes, out_perms=None):
    return (x * 100).reshape(out_shapes[0])
'''
    # Parent shares the leaf's name and uses the capture-then-shadow trick.
    parent_dsl = '''
_moe_dispatch_child = moe_dispatch
def moe_dispatch(x, *, out_shapes, out_perms=None):
    return _moe_dispatch_child(x, out_shapes=out_shapes) + 1
'''
    root_dsl = '''
def tiled_reference(dims, tensors):
    return moe_dispatch(tensors["x"],
                         out_shapes=(tuple(tensors["x"].shape),))
'''
    # Post-order: leaf first, then parent.
    ns = _pass2_compose_namespace(
        parent_dsl=root_dsl,
        child_dsls_in_order=[leaf_dsl, parent_dsl],
    )
    x = torch.arange(4, dtype=torch.float32)
    out = ns["tiled_reference"]({}, {"x": x})
    # Parent calls leaf (* 100), then adds 1 → 100x + 1
    assert torch.equal(out, x * 100 + 1)


def test_compose_handles_multi_output_child():
    """Child returns 3 tensors; parent destructures the tuple at call site
    — the namespace-rebinding path must transparently support tuple returns."""
    parent_dsl = '''
def tiled_reference(dims, tensors):
    x = tensors["x"]
    a, b, c = split3(x, out_shapes=((4,), (4,), (4,)))
    return a + b * 2 + c * 3
'''
    child_dsl = '''
def split3(x, *, out_shapes, out_perms=None):
    return x.reshape(-1), (x * 2).reshape(-1), (x * 4).reshape(-1)
'''
    ns = _pass2_compose_namespace(
        parent_dsl=parent_dsl,
        child_dsls_in_order=[child_dsl],
    )
    x = torch.arange(4, dtype=torch.float32)
    out = ns["tiled_reference"]({}, {"x": x})
    # a + 2*b + 3*c = x + 2*(2x) + 3*(4x) = x + 4x + 12x = 17x
    assert torch.equal(out, x * 17)
