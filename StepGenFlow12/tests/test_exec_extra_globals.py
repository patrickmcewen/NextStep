import torch
from src.tools import _exec_dsl_ref


SIMPLE_REF = '''
def tiled_reference(dims, tensors):
    return external_double(tensors["x"])
'''


def test_exec_dsl_ref_uses_extra_globals():
    def double(t):
        return t * 2
    x = torch.arange(8, dtype=torch.float32)
    out = _exec_dsl_ref(
        SIMPLE_REF,
        dims={},
        tensors={"x": x},
        extra_globals={"external_double": double},
    )
    assert torch.equal(out, x * 2)


def test_exec_dsl_ref_extra_globals_optional():
    """Ensure the change is backward-compatible: omitting extra_globals still works."""
    REF = '''
def tiled_reference(dims, tensors):
    return tensors["x"] * 3
'''
    x = torch.arange(8, dtype=torch.float32)
    out = _exec_dsl_ref(REF, dims={}, tensors={"x": x})
    assert torch.equal(out, x * 3)


NONROOT_REF = '''
def my_node(a, b, *, out_shapes):
    s = a + b
    return s.reshape(*out_shapes[0])
'''


def test_exec_dsl_ref_non_root_entry_point():
    """Non-root entry: function named after the planner node, called with
    contract-recorded tiled inputs and parent-declared out_shapes."""
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    b = torch.ones(2, 4)
    out = _exec_dsl_ref(
        NONROOT_REF,
        dims={},
        tensors={},
        entry_point="my_node",
        call_args=(a, b),
        call_kwargs={"out_shapes": ((4, 2),)},
    )
    assert torch.equal(out, (a + b).reshape(4, 2))


def test_exec_dsl_ref_missing_entry_point_raises():
    """Assertion message names the requested entry point."""
    REF = '''
def some_other_name(x):
    return x
'''
    try:
        _exec_dsl_ref(REF, dims={}, tensors={}, entry_point="my_node",
                      call_args=(torch.zeros(1),), call_kwargs={})
        raised = False
        msg = ""
    except AssertionError as exc:
        raised = True
        msg = str(exc)
    assert raised
    assert "my_node" in msg
