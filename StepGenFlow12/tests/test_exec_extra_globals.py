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
    raw = s.reshape(*out_shapes[0])
    # Non-root nodes must return a StepTensor so the parent's DSL ops can
    # chain on the return; wrap the raw result with a tile-stream dtype
    # derived from the parent-declared ``out_shapes`` (last two dims = tile).
    tile_shape = (int(out_shapes[0][-2]), int(out_shapes[0][-1]))
    return StepTensor(raw, stream_dtype=Tile(Float32(), tile_shape))
'''


def test_exec_dsl_ref_non_root_entry_point():
    """Non-root entry: function named after the planner node, called with
    contract-recorded tiled inputs and parent-declared out_shapes. The
    return must be a StepTensor (the executor unwraps it for the caller)."""
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


NONROOT_REF_RAW_RETURN = '''
def my_node(a, b, *, out_shapes):
    # Bug-class regression: a non-root node that returns a raw torch.Tensor
    # would silently pass the correctness gate (values match gold) but blow
    # up at pass2 the first time a parent DSL op meets a Tensor where a
    # StepTensor is required (e.g. accum_retile_row, .underlying_tensor).
    return (a + b).reshape(*out_shapes[0])
'''


def test_exec_dsl_ref_rejects_raw_tensor_from_non_root():
    """Non-root return must be a StepTensor; a raw torch.Tensor return
    fails the pass-1 correctness gate with a message that points to the
    StepTensor-on-chip return contract.

    Regression for the pattern observed in outer_1/.../update_cache:
    ``random_offchip_store(k_cache, ...); return k_cache, v_cache`` (the
    raw input cache) — values matched gold so pass1 was happy, but pass2
    composition crashed accessing ``.underlying_tensor`` on the raw
    return.
    """
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    b = torch.ones(2, 4)
    try:
        _exec_dsl_ref(
            NONROOT_REF_RAW_RETURN,
            dims={},
            tensors={},
            entry_point="my_node",
            call_args=(a, b),
            call_kwargs={"out_shapes": ((4, 2),)},
        )
        raised = False
        msg = ""
    except AssertionError as exc:
        raised = True
        msg = str(exc)
    assert raised, "executor must reject a non-root raw torch.Tensor return"
    assert "non-root" in msg, msg
    assert "StepTensor" in msg, msg


NONROOT_REF_RAW_RETURN_TUPLE = '''
def my_node(a, b, *, out_shapes):
    # Tuple-return variant of the raw-tensor bug.
    raw0 = (a + b).reshape(*out_shapes[0])
    raw1 = (a - b).reshape(*out_shapes[1])
    return raw0, raw1
'''


def test_exec_dsl_ref_rejects_raw_tensor_tuple_element_from_non_root():
    """A non-root tuple return with any raw element must be rejected, and
    the error message must identify which element failed."""
    a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    b = torch.ones(2, 4)
    try:
        _exec_dsl_ref(
            NONROOT_REF_RAW_RETURN_TUPLE,
            dims={},
            tensors={},
            entry_point="my_node",
            call_args=(a, b),
            call_kwargs={"out_shapes": ((4, 2), (4, 2))},
        )
        raised = False
        msg = ""
    except AssertionError as exc:
        raised = True
        msg = str(exc)
    assert raised
    assert "element [0]" in msg, msg
    assert "non-root" in msg, msg


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
