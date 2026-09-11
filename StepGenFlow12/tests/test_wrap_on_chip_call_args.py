"""Unit-test the IntArg pass-through path in ``_wrap_on_chip_call_args``.

The pre-fix bug crashed every KernelBench transformer kernel at pass-1
recursion: the parent's call site recorded ``tiled_shape = ()`` for the
0-D scalar ``n_head_tensor`` it forwarded to ``attention_layer``, and
the wrapper rejected the rank-0 shape. After introducing ``IntArg``,
the parent passes a Python ``int`` and the wrapper forwards it
unchanged — no rank check, no tensor wrap.
"""

import torch

from src.node_signature import IntArg, ListOfIntArg, TensorArg
from src.step_dsl import StepRawTensor, StepTensor
from src.tools import _wrap_on_chip_call_args


def test_int_arg_passes_through_as_python_int():
    """The fix for the GPT-2 / BART pass-1 wrap crash."""
    x = torch.randn(1, 2, 4, 8)
    wrapped = _wrap_on_chip_call_args(
        call_args=(x, 4),
        arg_specs=(TensorArg(shape=(2, 4, 8)), IntArg()),
        arg_is_raw=(False, False),
        tiled_shapes=((1, 2, 4, 8), ()),
    )
    assert isinstance(wrapped[0], StepTensor)
    assert wrapped[1] == 4
    assert isinstance(wrapped[1], int)


def test_int_arg_rejects_non_int():
    """Type-check at the boundary so a planner regression doesn't slip
    a torch.Tensor or a list through the IntArg slot."""
    try:
        _wrap_on_chip_call_args(
            call_args=(torch.tensor(4),),
            arg_specs=(IntArg(),),
            arg_is_raw=(False,),
            tiled_shapes=((),),
        )
        raised = False
    except AssertionError as e:
        raised = "IntArg must be a Python int" in str(e)
    assert raised


def test_on_chip_tensor_arg_still_rejects_rank0():
    """Defense in depth: even if a TensorArg slot somehow ends up with a 0-D
    tiled shape (e.g. a contract built before the fix), the wrapper still
    fails loud rather than silently producing a bogus stream_dtype."""
    try:
        _wrap_on_chip_call_args(
            call_args=(torch.tensor(4.0),),
            arg_specs=(TensorArg(shape=()),),
            arg_is_raw=(False,),
            tiled_shapes=((),),
        )
        raised = False
    except AssertionError as e:
        raised = "tiled shape" in str(e) and "rank >= 2" in str(e)
    assert raised


def test_raw_tensor_and_list_of_int_unchanged_alongside_int_arg():
    """Verify the new branch doesn't collide with the existing list /
    raw-tensor passes (regression test for branch ordering)."""
    w = torch.randn(8, 16)
    wrapped = _wrap_on_chip_call_args(
        call_args=(w, 7, [3, 5, 2]),
        arg_specs=(
            TensorArg(shape=(8, 16)),
            IntArg(),
            ListOfIntArg(length=3),
        ),
        arg_is_raw=(True, False, False),
        tiled_shapes=((8, 16), (), ()),
    )
    assert isinstance(wrapped[0], StepRawTensor)
    assert wrapped[1] == 7
    assert wrapped[2] == [3, 5, 2]
