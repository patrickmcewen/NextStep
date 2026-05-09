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
        child_name_to_dsl={"double": child_dsl},
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
    ns = _pass2_compose_namespace(
        parent_dsl=parent_dsl,
        child_name_to_dsl={
            "outer": middle_dsl,
            "inner": leaf_dsl,
        },
    )
    x = torch.arange(4, dtype=torch.float32)
    out = ns["tiled_reference"]({}, {"x": x})
    assert torch.equal(out, x * 10 + 1)


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
        child_name_to_dsl={"split3": child_dsl},
    )
    x = torch.arange(4, dtype=torch.float32)
    out = ns["tiled_reference"]({}, {"x": x})
    # a + 2*b + 3*c = x + 2*(2x) + 3*(4x) = x + 4x + 12x = 17x
    assert torch.equal(out, x * 17)
