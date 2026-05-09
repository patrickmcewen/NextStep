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
