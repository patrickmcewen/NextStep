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


def test_guard_accepts_python_int_at_call_site():
    """Python int args at a child call site are now first-class (IntArg).

    Before the IntArg fix, transformer kernels couldn't pass ``n_head`` to an
    ``attention_layer`` child without using a 0-D scalar tensor, which
    crashed pass-1 wrap. Now an int flows through unchanged."""
    children = [_StubChild(name="adder", reference_code=_LEAF_REF_WITH_INT_ARG)]
    check_children_called_with_tensors_only(
        _REFACTOR_PASSES_INT, children, dims={}, original_reference_code=_PARENT_REF)


_REFACTOR_PASSES_0D_TENSOR = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adder = AdderModel()

    def forward(self, x, y):
        scalar = torch.tensor(3, dtype=torch.int64)
        return self.adder(x, y, scalar)
"""


def test_guard_rejects_0d_tensor_at_call_site():
    """0-D ``torch.Tensor``s have no tile-stream form — guide the LLM to
    pass a Python int (IntArg) instead. This is the canonical mistake the
    planner LLM makes when emitting transformer kernels."""
    children = [_StubChild(name="adder", reference_code=_LEAF_REF_WITH_INT_ARG)]
    with pytest.raises(GuardFailure) as exc_info:
        check_children_called_with_tensors_only(
            _REFACTOR_PASSES_0D_TENSOR, children, dims={},
            original_reference_code=_PARENT_REF)
    msg = str(exc_info.value)
    assert "adder" in msg
    assert "0-D" in msg
    assert "Python int" in msg


_REFACTOR_PASSES_BOOL = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.adder = AdderModel()

    def forward(self, x, y):
        return self.adder(x, y, True)
"""


def test_guard_rejects_bool_at_call_site():
    """``bool`` is an ``int`` subclass — explicitly reject so a planner bug
    doesn't silently coerce ``True`` to ``IntArg(1)``."""
    children = [_StubChild(name="adder", reference_code=_LEAF_REF_WITH_INT_ARG)]
    with pytest.raises(GuardFailure) as exc_info:
        check_children_called_with_tensors_only(
            _REFACTOR_PASSES_BOOL, children, dims={},
            original_reference_code=_PARENT_REF)
    msg = str(exc_info.value)
    assert "adder" in msg
    assert "bool" in msg


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


# ---------------------------------------------------------------------------
# Parent-level list inputs are allowed; flowing them into a child is not.
# ---------------------------------------------------------------------------

_PARENT_REF_WITH_LIST = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, w_list, num_token_list):
        return x

def get_inputs(dims):
    import torch
    return [
        torch.randn(4, 8),
        [torch.randn(8, 8) for _ in range(3)],
        [2, 5, 3, 7],
    ]
"""

_LEAF_REF_TAKES_LIST = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, w_list):
        out = x
        for w in w_list:
            out = out + x @ w
        return out

def get_inputs(dims):
    import torch
    return [torch.randn(4, 8), [torch.randn(8, 8) for _ in range(3)]]
"""

_REFACTOR_PASSES_LIST_TO_CHILD = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_block = ExpertBlockModel()

    def forward(self, x, w_list, num_token_list):
        # Anti-pattern: parent forwards the whole list to a child instead
        # of iterating it and passing per-element tensors.
        return self.expert_block(x, w_list)
"""

_REFACTOR_STACKS_LIST_BEFORE_CHILD = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_block = ExpertBlockModel()

    def forward(self, x, w_list, num_token_list):
        # Correct pattern: stack the list to a tensor at the call site.
        w_stacked = torch.stack(w_list, dim=0)
        return self.expert_block(x, w_stacked)
"""

_LEAF_REF_TAKES_STACKED_TENSOR = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, w_stacked):
        out = x
        for e in range(w_stacked.shape[0]):
            out = out + x @ w_stacked[e]
        return out

def get_inputs(dims):
    import torch
    return [torch.randn(4, 8), torch.randn(3, 8, 8)]
"""


def test_guard_accepts_list_of_tensors_passed_to_child():
    """list[Tensor] (homogeneous element shape) is a supported child arg kind
    — make_stub records it as ListOfTensorArg and Pass-1 prompt rendering
    formats it as ``list[Tensor(...)] x N`` so the LLM can iterate it."""
    children = [_StubChild(name="expert_block", reference_code=_LEAF_REF_TAKES_LIST)]
    check_children_called_with_tensors_only(
        _REFACTOR_PASSES_LIST_TO_CHILD, children, dims={},
        original_reference_code=_PARENT_REF_WITH_LIST)


def test_guard_passes_when_parent_stacks_list_before_child_call():
    """A parent that stacks a list[Tensor] into a 3D tensor before calling
    its child still passes — the alternative legitimate pattern."""
    children = [_StubChild(name="expert_block",
                           reference_code=_LEAF_REF_TAKES_STACKED_TENSOR)]
    check_children_called_with_tensors_only(
        _REFACTOR_STACKS_LIST_BEFORE_CHILD, children, dims={},
        original_reference_code=_PARENT_REF_WITH_LIST)


_LEAF_REF_TAKES_LIST_OF_INT = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, num_token_list):
        out = torch.zeros_like(x)
        for i in range(x.shape[0]):
            out[i, :num_token_list[i]] = x[i, :num_token_list[i]]
        return out

def get_inputs(dims):
    import torch
    return [torch.randn(4, 8), [2, 5, 3, 7]]
"""

_REFACTOR_PASSES_LIST_OF_INT_TO_CHILD = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache_writer = CacheWriterModel()

    def forward(self, x, w_list, num_token_list):
        return self.cache_writer(x, num_token_list)
"""


def test_guard_accepts_list_of_int_passed_to_child():
    """list[int] (e.g. per-batch sequence lengths) is supported at child
    call sites; the child's DSL output converts it to a tensor and feeds
    ``metadata_gen`` / ``cache_*_addr_gen`` etc."""
    children = [_StubChild(name="cache_writer",
                           reference_code=_LEAF_REF_TAKES_LIST_OF_INT)]
    check_children_called_with_tensors_only(
        _REFACTOR_PASSES_LIST_OF_INT_TO_CHILD, children, dims={},
        original_reference_code=_PARENT_REF_WITH_LIST)


_LEAF_REF_TAKES_BAD_LIST = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, mixed_list):
        return x

def get_inputs(dims):
    import torch
    return [torch.randn(4, 8), [torch.randn(8), 3]]
"""

_REFACTOR_PASSES_MIXED_LIST = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bad = BadModel()

    def forward(self, x, w_list, num_token_list):
        # Build a mixed-type list inline and pass it to the child.
        return self.bad(x, [w_list[0], num_token_list[0]])
"""


def test_guard_rejects_mixed_list_at_child_call_site():
    """Mixed-type lists (e.g. [Tensor, int]) cannot be classified as either
    list[Tensor] or list[int] — must be rejected with a clear message."""
    children = [_StubChild(name="bad", reference_code=_LEAF_REF_TAKES_BAD_LIST)]
    with pytest.raises(GuardFailure) as exc_info:
        check_children_called_with_tensors_only(
            _REFACTOR_PASSES_MIXED_LIST, children, dims={},
            original_reference_code=_PARENT_REF_WITH_LIST)
    msg = str(exc_info.value)
    assert "bad" in msg
    assert "mixed/unsupported" in msg
