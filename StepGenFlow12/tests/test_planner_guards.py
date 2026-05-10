"""Guards verifying planner-level structural rules."""

from dataclasses import dataclass

import pytest

from src.planner import (
    GuardFailure,
    check_children_called_with_tensors_only,
)


@dataclass
class _StubChild:
    name: str
    reference_code: str


_PARENT_REF = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

def get_inputs(dims):
    import torch
    return [torch.randn(4, 8), torch.randn(4, 8)]
"""

_LEAF_REF_TENSORS_ONLY = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, a, b):
        return a + b

def get_inputs(dims):
    import torch
    return [torch.randn(4, 8), torch.randn(4, 8)]
"""

_LEAF_REF_WITH_INT_ARG = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, a, b, e_idx: int = 0):
        return a + b + e_idx

def get_inputs(dims):
    import torch
    return [torch.randn(4, 8), torch.randn(4, 8)]
"""

_REFACTOR_TENSORS_ONLY = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adder = AdderModel()

    def forward(self, x, y):
        return self.adder(x, y)
"""

_REFACTOR_PASSES_INT = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adder = AdderModel()

    def forward(self, x, y):
        out = torch.zeros_like(x)
        for e_idx in range(2):
            out = out + self.adder(x, y, e_idx)
        return out
"""


def test_guard_passes_when_all_args_are_tensors():
    children = [_StubChild(name="adder", reference_code=_LEAF_REF_TENSORS_ONLY)]
    check_children_called_with_tensors_only(
        _REFACTOR_TENSORS_ONLY, children, dims={}, original_reference_code=_PARENT_REF)


def test_guard_rejects_python_int_at_call_site():
    children = [_StubChild(name="adder", reference_code=_LEAF_REF_WITH_INT_ARG)]
    with pytest.raises(GuardFailure) as exc_info:
        check_children_called_with_tensors_only(
            _REFACTOR_PASSES_INT, children, dims={}, original_reference_code=_PARENT_REF)
    msg = str(exc_info.value)
    assert "adder" in msg
    assert "position 2" in msg
    assert "int" in msg


def test_guard_skipped_for_function_based_parent():
    function_based_ref = """
def compute_gold(dims, tensors):
    import torch
    return tensors["x"] + tensors["y"]
"""
    children = [_StubChild(name="adder", reference_code=_LEAF_REF_WITH_INT_ARG)]
    check_children_called_with_tensors_only(
        _REFACTOR_PASSES_INT, children, dims={},
        original_reference_code=function_based_ref)
