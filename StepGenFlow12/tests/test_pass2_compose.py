import torch
from src.orchestrator import _pass2_compose_namespace


def test_compose_rebinds_child_to_verified_dsl():
    # Parent calls "double" as if it were a blackbox.
    parent_dsl = '''
def tiled_reference(dims, tensors):
    return double(tensors["x"], out_shape=tuple(tensors["x"].shape))
'''
    # Child's verified DSL implements "double" properly.
    child_dsl = '''
def double(x, *, out_shape, out_perm=None):
    return (x * 2).reshape(out_shape)
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
    return outer(tensors["x"], out_shape=tuple(tensors["x"].shape))
'''
    middle_dsl = '''
def outer(x, *, out_shape, out_perm=None):
    return inner(x, out_shape=out_shape) + 1
'''
    leaf_dsl = '''
def inner(x, *, out_shape, out_perm=None):
    return (x * 10).reshape(out_shape)
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
