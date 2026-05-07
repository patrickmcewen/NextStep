import pytest
import torch  # noqa: F401  used in exec'd code
from src.planner import (parse_planner_response, ParsedSplit,
                         NotADecision, MalformedSplit,
                         synthesize_reference_module)


_SAMPLE_LEAF_RESPONSE = """
Some reasoning here.

DECISION: leaf
"""

_SAMPLE_SPLIT_RESPONSE = """
Reasoning preamble.

DECISION: split

# child: scores
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K):
        return Q @ K.T

def get_inputs(dims):
    torch.manual_seed(101)
    M, N, D = dims["M"], dims["N"], dims["D"]
    return torch.randn(M, D), torch.randn(N, D)

# child: safe_softmax
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, scores):
        return scores

def get_inputs(dims):
    torch.manual_seed(102)
    M, N = dims["M"], dims["N"]
    return (torch.randn(M, N),)

# refactored parent
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = ScoresModel()
        self.safe_softmax = SafeSoftmaxModel()
    def forward(self, Q, K, V):
        s = self.scores(Q, K)
        return self.safe_softmax(s)
"""


def test_parse_leaf_returns_leaf_decision():
    result = parse_planner_response(_SAMPLE_LEAF_RESPONSE)
    assert result == "leaf"


def test_parse_split_returns_parsed_split():
    result = parse_planner_response(_SAMPLE_SPLIT_RESPONSE)
    assert isinstance(result, ParsedSplit)
    assert [c.name for c in result.children] == ["scores", "safe_softmax"]
    assert "ScoresModel" in result.refactored_parent_code
    assert "self.scores = ScoresModel()" in result.refactored_parent_code


def test_parse_split_each_child_has_self_contained_reference_code():
    result = parse_planner_response(_SAMPLE_SPLIT_RESPONSE)
    scores = result.children[0]
    assert "class Model(nn.Module):" in scores.reference_code
    assert "def get_inputs(dims):" in scores.reference_code
    assert "torch.manual_seed(101)" in scores.reference_code


def test_parse_no_decision_raises_NotADecision():
    with pytest.raises(NotADecision):
        parse_planner_response("just some text without DECISION:")


def test_parse_split_with_one_child_raises_MalformedSplit():
    bad = """
DECISION: split

# child: only_one
class Model(nn.Module):
    def forward(self, x):
        return x
def get_inputs(dims):
    return (torch.randn(4),)

# refactored parent
class Model(nn.Module):
    pass
"""
    with pytest.raises(MalformedSplit, match="at least 2 children"):
        parse_planner_response(bad)


def test_parse_split_missing_refactored_parent_raises():
    bad = """
DECISION: split

# child: a
class Model(nn.Module):
    def forward(self, x):
        return x
def get_inputs(dims):
    return (torch.randn(4),)

# child: b
class Model(nn.Module):
    def forward(self, x):
        return x
def get_inputs(dims):
    return (torch.randn(4),)
"""
    with pytest.raises(MalformedSplit, match="refactored parent"):
        parse_planner_response(bad)


def test_parse_split_duplicate_child_names_raises():
    bad = """
DECISION: split

# child: dup
class Model(nn.Module):
    def forward(self, x):
        return x
def get_inputs(dims):
    return (torch.randn(4),)

# child: dup
class Model(nn.Module):
    def forward(self, x):
        return x
def get_inputs(dims):
    return (torch.randn(4),)

# refactored parent
class Model(nn.Module):
    pass
"""
    with pytest.raises(MalformedSplit, match="duplicate"):
        parse_planner_response(bad)


def test_parse_split_child_missing_class_model_raises_MalformedSplit():
    bad = """
DECISION: split

# child: rowwise_softmax
def get_inputs(dims):
    return (torch.randn(4),)

# child: other
class Model(nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    return (torch.randn(4),)

# refactored parent
class Model(nn.Module):
    pass
"""
    with pytest.raises(MalformedSplit, match="missing 'class Model'"):
        parse_planner_response(bad)


def test_parse_split_strips_markdown_code_fences_in_bodies():
    """Some LLMs wrap each block in ```python ... ``` fences. The parser must
    strip them before returning the reference_code, otherwise downstream
    ast.parse() chokes on the literal backticks."""
    fenced = """
DECISION: split

# child: a
```python
class Model(nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    return (torch.randn(4),)
```

# child: b
```python
class Model(nn.Module):
    def forward(self, x):
        return x + 1
def get_inputs(dims):
    return (torch.randn(4),)
```

# refactored parent
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = AModel()
        self.b = BModel()
    def forward(self, x):
        return self.b(self.a(x))
```
"""
    result = parse_planner_response(fenced)
    assert isinstance(result, ParsedSplit)
    for child in result.children:
        assert "```" not in child.reference_code
    assert "```" not in result.refactored_parent_code


def test_parse_split_child_missing_get_inputs_raises_MalformedSplit():
    bad = """
DECISION: split

# child: a
class Model(nn.Module):
    def forward(self, x):
        return x

# child: b
class Model(nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    return (torch.randn(4),)

# refactored parent
class Model(nn.Module):
    pass
"""
    with pytest.raises(MalformedSplit, match="missing 'def get_inputs'"):
        parse_planner_response(bad)


def test_synthesize_appends_compute_gold_and_init_inputs():
    body = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * 2

def get_inputs(dims):
    torch.manual_seed(7)
    return (torch.randn(dims["M"]),)
"""
    full = synthesize_reference_module(body)
    assert "def get_init_inputs(dims):" in full
    assert "return []" in full
    assert "def compute_gold(dims):" in full
    namespace = {}
    exec(full, namespace)
    out = namespace["compute_gold"]({"M": 4})
    inputs = namespace["get_inputs"]({"M": 4})
    expected = namespace["Model"]()(*inputs)
    assert torch.equal(out, expected)


def test_synthesize_idempotent_when_compute_gold_already_present():
    body_with_gold = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x

def get_inputs(dims):
    return (torch.randn(4),)

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""
    full = synthesize_reference_module(body_with_gold)
    assert full.count("def compute_gold(dims):") == 1
    assert full.count("def get_init_inputs(dims):") == 1


from src.planner import build_node_tensors


def test_build_node_tensors_zips_forward_args_with_get_inputs():
    code = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, Q, K, V):
        return Q @ K.T @ V
def get_inputs(dims):
    torch.manual_seed(7)
    M, N, D = dims["M"], dims["N"], dims["D"]
    return torch.randn(M, D), torch.randn(N, D), torch.randn(N, D)
def get_init_inputs(dims):
    return []
def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""
    tensors = build_node_tensors(code, dims={"M": 2, "N": 3, "D": 4})
    assert sorted(tensors) == ["K", "Q", "V"]
    assert tensors["Q"].shape == (2, 4)
    assert tensors["K"].shape == (3, 4)
    assert tensors["V"].shape == (3, 4)


def test_build_node_tensors_handles_single_arg_returning_single_tensor():
    code = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    return torch.randn(dims["M"]),  # tuple with trailing comma
def get_init_inputs(dims):
    return []
def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""
    tensors = build_node_tensors(code, dims={"M": 5})
    assert list(tensors) == ["x"]
    assert tensors["x"].shape == (5,)
