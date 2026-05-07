import pytest
from src.planner import (parse_planner_response, ParsedSplit,
                         NotADecision, MalformedSplit)


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
