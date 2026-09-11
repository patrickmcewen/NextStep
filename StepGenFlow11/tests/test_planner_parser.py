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


def test_parse_split_with_one_child_is_accepted():
    """The carve-out pattern (1 child, parent does the rest) is a valid
    decomposition. The parser must not require ≥2 children — that artificial
    quota leads LLMs to invent dead siblings just to satisfy it."""
    response = """
DECISION: split

# child: rotate_half
class Model(nn.Module):
    def forward(self, x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
def get_inputs(dims):
    return (torch.randn(dims["M"]),)

# refactored parent
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotate_half = RotateHalfModel()
    def forward(self, x, cos, sin):
        return x * cos + self.rotate_half(x) * sin
"""
    result = parse_planner_response(response)
    assert isinstance(result, ParsedSplit)
    assert [c.name for c in result.children] == ["rotate_half"]


def test_parse_split_with_zero_children_raises_MalformedSplit():
    bad = """
DECISION: split

# refactored parent
class Model(nn.Module):
    pass
"""
    with pytest.raises(MalformedSplit, match="at least 1 child"):
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


def test_parse_split_accepts_child_class_named_after_camelcase_child():
    """LLMs sometimes name a child's class ``<CamelCase>Model`` instead of the
    literal ``class Model``. The parser normalizes it so downstream guards and
    synth see a uniform ``class Model`` per child."""
    response = """
DECISION: split

# child: pre_attention
```python
class PreAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * 2

def get_inputs(dims):
    return (torch.randn(4),)
```

# child: post_norm
```python
class PostNormModel(nn.Module):
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
        self.pre_attention = PreAttentionModel()
        self.post_norm = PostNormModel()
    def forward(self, x):
        return self.post_norm(self.pre_attention(x))
```
"""
    result = parse_planner_response(response)
    assert isinstance(result, ParsedSplit)
    assert [c.name for c in result.children] == ["pre_attention", "post_norm"]
    for child in result.children:
        assert "class Model(nn.Module):" in child.reference_code
        assert "class PreAttentionModel" not in child.reference_code
        assert "class PostNormModel" not in child.reference_code


def test_parse_split_normalizes_acronym_class_name_to_Model():
    """LLMs preserve acronym capitalization (rms_norm → RMSNormModel, not
    RmsNormModel). When snake→Camel exact-match misses, the parser falls
    back to renaming the unique top-level ``class <Identifier>Model(nn.Module)``
    to ``class Model``."""
    response = """
DECISION: split

# child: rms_norm
```python
import torch
import torch.nn as nn

class RMSNormModel(nn.Module):
    def forward(self, x):
        return x * 2

def get_inputs(dims):
    return (torch.randn(4),)
```

# refactored parent
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms_norm = RMSNormModel()
    def forward(self, x):
        return self.rms_norm(x)
```
"""
    result = parse_planner_response(response)
    assert isinstance(result, ParsedSplit)
    assert "class Model(nn.Module):" in result.children[0].reference_code
    assert "class RMSNormModel" not in result.children[0].reference_code


def test_parse_split_accepts_indented_markers_and_dedents_bodies():
    """Models sometimes emit the entire DECISION: split block indented (e.g.
    as a sub-bullet). Markers ``# child:`` / ``# refactored parent`` and the
    code bodies must still parse — and bodies must be dedented so downstream
    ast.parse / exec succeed."""
    response = """
DECISION: split

  # child: doubler
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def forward(self, x):
            return x * 2

    def get_inputs(dims):
        return (torch.randn(4),)

  # refactored parent
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.doubler = DoublerModel()
        def forward(self, x):
            return self.doubler(x)
"""
    result = parse_planner_response(response)
    assert isinstance(result, ParsedSplit)
    assert [c.name for c in result.children] == ["doubler"]
    # Bodies must be dedented so they're valid module-level Python.
    import ast
    ast.parse(result.children[0].reference_code)
    ast.parse(result.refactored_parent_code)


def test_parse_split_discards_trailing_prose_after_closing_fence():
    """LLMs sometimes append an "explanation" paragraph after the closing
    fence of the refactored parent block. That prose must be discarded —
    leaving it in lets unicode characters (e.g. non-breaking hyphen) crash
    ast.parse() downstream."""
    response = """
DECISION: split

# child: doubler
```python
class Model(nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    return (torch.randn(4),)
```

# refactored parent
```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.doubler = DoublerModel()
    def forward(self, x):
        return self.doubler(x)
```

The parent now delegates to DoublerModel — note the non‑breaking hyphen here.
"""
    result = parse_planner_response(response)
    assert isinstance(result, ParsedSplit)
    # The refactored parent body must contain ONLY the fenced code, no prose.
    assert "non" not in result.refactored_parent_code
    assert "delegates" not in result.refactored_parent_code
    assert "class Model(nn.Module):" in result.refactored_parent_code


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


def test_synthesize_adds_import_torch_when_only_import_torch_nn_present():
    """Regression: 'import torch' substring check used to match
    'import torch.nn as nn' and skip adding 'import torch', leaving the
    refactored module unable to reference ``torch.X``."""
    body = """\
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.cat([x, x], dim=-1)

def get_inputs(dims):
    return (torch.randn(4),)
"""
    full = synthesize_reference_module(body)
    namespace = {}
    exec(full, namespace)
    assert "torch" in namespace
    out = namespace["Model"]()(namespace["torch"].zeros(2, 3))
    assert out.shape == (2, 6)


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


def test_build_node_tensors_handles_list_return_from_get_inputs():
    """Some StepDB references return ``[Q, K, cos, sin]`` (a list, not a tuple).
    The walker must accept either."""
    code = """\
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, Q, K, cos, sin):
        return Q + K + cos + sin
def get_inputs(dims):
    M = dims["M"]
    return [torch.randn(M), torch.randn(M), torch.randn(M), torch.randn(M)]
def get_init_inputs(dims):
    return []
def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""
    tensors = build_node_tensors(code, dims={"M": 3})
    assert sorted(tensors) == ["K", "Q", "cos", "sin"]
    for t in tensors.values():
        assert t.shape == (3,)


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
