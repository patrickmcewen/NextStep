import pytest
from src.planner import (check_anti_passthrough, check_anti_monolith,
                         check_compose, GuardFailure, ParsedChild)


_PASSTHROUGH_CHILD = ParsedChild(name="bad", reference_code="""\
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x
def get_inputs(dims):
    return (torch.randn(4),)
""")

_REAL_CHILD = ParsedChild(name="good", reference_code="""\
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2 + 1
def get_inputs(dims):
    return (torch.randn(4),)
""")


def test_anti_passthrough_rejects_pure_return():
    with pytest.raises(GuardFailure, match="passthrough"):
        check_anti_passthrough([_PASSTHROUGH_CHILD])


def test_anti_passthrough_accepts_real_op():
    check_anti_passthrough([_REAL_CHILD])


def test_anti_passthrough_reports_first_offender_name():
    with pytest.raises(GuardFailure, match="bad"):
        check_anti_passthrough([_REAL_CHILD, _PASSTHROUGH_CHILD])


_ORIGINAL_PARENT = """\
import torch
class Model(torch.nn.Module):
    def forward(self, Q, K, V):
        s = Q @ K.T
        m = s.max(dim=-1, keepdim=True).values
        e = torch.exp(s - m)
        ctx = e @ V
        norm = e.sum(dim=-1, keepdim=True)
        return ctx / norm
"""

_REFACTORED_SMALLER = """\
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = ScoresModel()
        self.softmax = SafeSoftmaxModel()
    def forward(self, Q, K, V):
        s = self.scores(Q, K)
        p = self.softmax(s)
        return p @ V
"""

_REFACTORED_NOT_SMALLER = """\
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = ScoresModel()
    def forward(self, Q, K, V):
        s = self.scores(Q, K)
        m = s.max(dim=-1, keepdim=True).values
        e = torch.exp(s - m)
        ctx = e @ V
        norm = e.sum(dim=-1, keepdim=True)
        return ctx / norm
"""


def test_anti_monolith_rejects_when_refactor_not_smaller():
    with pytest.raises(GuardFailure, match="not smaller"):
        check_anti_monolith(_ORIGINAL_PARENT, _REFACTORED_NOT_SMALLER)


def test_anti_monolith_accepts_smaller_refactor():
    check_anti_monolith(_ORIGINAL_PARENT, _REFACTORED_SMALLER)


_ORIGINAL_REFERENCE = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * 2 + 1

def get_inputs(dims):
    torch.manual_seed(0)
    return (torch.randn(dims["M"]),)

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""

_CORRECT_DECOMPOSITION_CHILDREN = [
    ParsedChild(name="doubler", reference_code="""\
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    return (torch.randn(dims["M"]),)
"""),
    ParsedChild(name="adder", reference_code="""\
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        return x + 1
def get_inputs(dims):
    return (torch.randn(dims["M"]),)
"""),
]

_CORRECT_REFACTORED_PARENT = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.doubler = DoublerModel()
        self.adder = AdderModel()
    def forward(self, x):
        return self.adder(self.doubler(x))
"""

_INCORRECT_REFACTORED_PARENT = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.doubler = DoublerModel()
        self.adder = AdderModel()
    def forward(self, x):
        return self.doubler(self.doubler(x))
"""


def test_compose_check_accepts_correct_decomposition():
    check_compose(_ORIGINAL_REFERENCE,
                  _CORRECT_REFACTORED_PARENT,
                  _CORRECT_DECOMPOSITION_CHILDREN,
                  dims={"M": 8})


def test_compose_check_rejects_wrong_decomposition():
    with pytest.raises(GuardFailure, match="compose check failed"):
        check_compose(_ORIGINAL_REFERENCE,
                      _INCORRECT_REFACTORED_PARENT,
                      _CORRECT_DECOMPOSITION_CHILDREN,
                      dims={"M": 8})
