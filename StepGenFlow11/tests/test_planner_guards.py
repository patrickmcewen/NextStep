import pytest
from src.planner import (check_anti_passthrough, check_anti_monolith,
                         check_children_runnable, check_compose,
                         check_no_dead_children,
                         GuardFailure, ParsedChild, has_class_model)


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


_FUNCTION_BASED_REFERENCE = """\
import torch
import torch.nn as nn

def get_inputs(dims):
    torch.manual_seed(0)
    return [torch.randn(dims["M"]), torch.randn(dims["M"])]

def compute_gold(dims):
    a, b = get_inputs(dims)
    out = a * 2
    out = out + b
    return out
"""

_FUNCTION_BASED_REFACTORED_PARENT = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.doubler = DoublerModel()
    def forward(self, a, b):
        return self.doubler(a) + b
"""

_FUNCTION_BASED_CHILDREN = [
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
    def forward(self, a, b):
        return a + b
def get_inputs(dims):
    return (torch.randn(dims["M"]), torch.randn(dims["M"]))
"""),
]


def test_has_class_model_detects_class_model():
    assert has_class_model(_ORIGINAL_REFERENCE) is True
    assert has_class_model(_FUNCTION_BASED_REFERENCE) is False


def test_anti_monolith_handles_function_based_original():
    """A reference with no class Model (only compute_gold) is the anti-monolith
    baseline; refactored Model.forward must be smaller than compute_gold's body."""
    check_anti_monolith(_FUNCTION_BASED_REFERENCE, _FUNCTION_BASED_REFACTORED_PARENT)


_PARENT_WITH_LIVE_CHILDREN = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = ScoresModel()
        self.softmax = SafeSoftmaxModel()
    def forward(self, Q, K, V):
        s = self.scores(Q, K)
        p = self.softmax(s)
        return p @ V
"""

_PARENT_WITH_DEAD_CHILD = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = RopeApplyAllModel()
        self.rot = RotateHalfModel()
    def forward(self, Q, K, cos, sin):
        return self.rope(Q, K, cos, sin)
"""

_LIVE_CHILDREN = [
    ParsedChild(name="scores", reference_code=""),
    ParsedChild(name="softmax", reference_code=""),
]

_DEAD_CHILDREN = [
    ParsedChild(name="rope", reference_code=""),
    ParsedChild(name="rot", reference_code=""),
]


def test_no_dead_children_accepts_when_all_invoked():
    check_no_dead_children(_PARENT_WITH_LIVE_CHILDREN, _LIVE_CHILDREN)


def test_no_dead_children_rejects_when_child_only_in_init():
    with pytest.raises(GuardFailure, match="dead"):
        check_no_dead_children(_PARENT_WITH_DEAD_CHILD, _DEAD_CHILDREN)


def test_no_dead_children_accepts_carve_out_pattern():
    """Single child invoked from forward — the carve-out pattern."""
    parent = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotate_half = RotateHalfModel()
    def forward(self, x, cos, sin):
        return x * cos + self.rotate_half(x) * sin
"""
    check_no_dead_children(parent,
                           [ParsedChild(name="rotate_half", reference_code="")])


def test_compose_check_converts_runtime_error_to_GuardFailure():
    """Regression: when the refactored parent crashes at run time (e.g. shape
    mismatch from a wrongly-decomposed child contract), check_compose must
    raise GuardFailure so the planner can feed the message back to the LLM
    instead of crashing the whole run."""
    parent_with_shape_bug = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.doubler = DoublerModel()
    def forward(self, x):
        bad = torch.randn(3, 5)
        return bad + self.doubler(x)
"""
    children = [ParsedChild(name="doubler", reference_code="""\
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    return (torch.randn(dims["M"]),)
""")]
    with pytest.raises(GuardFailure, match="refactored parent's forward.*crashed"):
        check_compose(_ORIGINAL_REFERENCE, parent_with_shape_bug,
                      children, dims={"M": 8})


def test_compose_check_handles_function_based_original():
    """compose check on a function-based reference uses compute_gold(dims) as the
    gold instead of Model()(*inputs)."""
    check_compose(_FUNCTION_BASED_REFERENCE,
                  _FUNCTION_BASED_REFACTORED_PARENT,
                  _FUNCTION_BASED_CHILDREN,
                  dims={"M": 8})


def test_children_runnable_accepts_well_formed_child():
    check_children_runnable([_REAL_CHILD], dims={"M": 4})


def test_children_runnable_rejects_child_with_NameError_in_get_inputs():
    """LLMs sometimes emit children that reference helpers (e.g. ``_model_config``)
    without defining them. The bug only surfaces at the next recursion level
    when the planner uses the child as the new "original" — which is too late;
    by then it crashes the outer iteration as a non-GuardFailure exception."""
    bad = ParsedChild(name="undef_helper", reference_code="""\
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    cfg = _model_config(dims["model_name"])  # noqa: F821 - intentionally undefined
    return (torch.randn(4),)
""")
    with pytest.raises(GuardFailure, match="undef_helper.*get_inputs.*NameError"):
        check_children_runnable([bad], dims={"model_name": "mixtral"})


def test_children_runnable_rejects_child_with_assertion_in_get_inputs():
    """Mirrors the actual outer_3 crash: child's _model_config dispatch had a
    typo so the dims["model_name"] fell through to assertion."""
    bad = ParsedChild(name="typo_dispatch", reference_code="""\
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    if dims["model_name"] == "qwen":
        return (torch.randn(4),)
    raise AssertionError(f"Unknown model_name: {dims['model_name']!r}")
""")
    with pytest.raises(GuardFailure, match="typo_dispatch.*Unknown model_name"):
        check_children_runnable([bad], dims={"model_name": "mixtral"})


def test_children_runnable_accepts_tuple_returning_forward():
    """Multi-output forwards (e.g. fused QKV projection returning Q, K, V) are
    legal at intermediate (non-root) nodes. Per-node DSLs are only verified
    against gold — they are never translated to STeP individually, so a tuple
    return is fine. The root inherits its single-tensor contract from the
    original kernel."""
    multi = ParsedChild(name="qkv_split", reference_code="""\
import torch
class Model(torch.nn.Module):
    def forward(self, x, q_proj, k_proj, v_proj):
        return x @ q_proj, x @ k_proj, x @ v_proj
def get_inputs(dims):
    return (torch.randn(4, 8), torch.randn(8, 8),
            torch.randn(8, 8), torch.randn(8, 8))
""")
    check_children_runnable([multi], dims={})


def test_children_runnable_rejects_child_whose_Model_forward_crashes():
    """If Model()(*get_inputs(dims)) crashes (e.g. wrong input contract from
    the parent's perspective), surface it as a GuardFailure too."""
    bad = ParsedChild(name="shape_bug", reference_code="""\
import torch
class Model(torch.nn.Module):
    def forward(self, x, y):
        return x @ y
def get_inputs(dims):
    return (torch.randn(4, 5), torch.randn(7, 3))  # incompatible shapes
""")
    with pytest.raises(GuardFailure, match="shape_bug.*Model"):
        check_children_runnable([bad], dims={"M": 4})


def test_children_runnable_rejects_child_missing_get_inputs():
    """If the LLM emits a child block without get_inputs(dims), surface it
    as a retryable GuardFailure (not a fatal AssertionError that crashes
    the whole outer iteration)."""
    bad = ParsedChild(name="no_inputs", reference_code="""\
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x * 2
""")
    with pytest.raises(GuardFailure, match="no_inputs.*get_inputs"):
        check_children_runnable([bad], dims={"M": 4})


def test_children_runnable_rejects_child_missing_Model_class():
    """If the LLM emits a child block without class Model, surface it as a
    retryable GuardFailure."""
    bad = ParsedChild(name="no_model", reference_code="""\
import torch
def get_inputs(dims):
    return (torch.randn(dims["M"]),)
""")
    with pytest.raises(GuardFailure, match="no_model.*Model"):
        check_children_runnable([bad], dims={"M": 4})
