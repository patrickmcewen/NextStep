import pytest
from unittest.mock import MagicMock
from src.planner import plan, PlannerExhausted


_LEAF_REF = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x * 2

def get_inputs(dims):
    return (torch.randn(dims["M"]),)

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""

_PARENT_REF = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        a = x * 2
        b = a + 1
        return b

def get_inputs(dims):
    torch.manual_seed(0)
    return (torch.randn(dims["M"]),)

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""

_SPLIT_RESPONSE = """
DECISION: split

# child: doubler
class Model(nn.Module):
    def forward(self, x):
        return x * 2
def get_inputs(dims):
    torch.manual_seed(101)
    return (torch.randn(dims["M"]),)

# child: adder
class Model(nn.Module):
    def forward(self, x):
        return x + 1
def get_inputs(dims):
    torch.manual_seed(102)
    return (torch.randn(dims["M"]),)

# refactored parent
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.doubler = DoublerModel()
        self.adder = AdderModel()
    def forward(self, x):
        return self.adder(self.doubler(x))
"""

_LEAF_RESPONSE = "DECISION: leaf"


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, agent, conversation):
        self.calls.append(conversation)
        return _FakeResult(self.responses.pop(0))


class _FakeResult:
    def __init__(self, text):
        self.final_output = text


@pytest.mark.asyncio
async def test_plan_leaf_returns_leaf_node():
    fake = _FakeLLM([_LEAF_RESPONSE])
    tree = await plan(reference_code=_LEAF_REF, dims={"M": 4},
                      agent=MagicMock(), path="root",
                      runner_fn=fake, retry_budget=3)
    assert tree.is_leaf
    assert tree.path == "root"


@pytest.mark.asyncio
async def test_plan_split_then_two_leaves_returns_subtree():
    fake = _FakeLLM([_SPLIT_RESPONSE, _LEAF_RESPONSE, _LEAF_RESPONSE])
    tree = await plan(reference_code=_PARENT_REF, dims={"M": 4},
                      agent=MagicMock(), path="root",
                      runner_fn=fake, retry_budget=3)
    assert not tree.is_leaf
    assert tree.refactored_code is not None
    assert [c.name for c in tree.children] == ["doubler", "adder"]
    assert all(c.is_leaf for c in tree.children)


@pytest.mark.asyncio
async def test_plan_retries_on_NotADecision_then_succeeds():
    fake = _FakeLLM(["I am thinking out loud", _LEAF_RESPONSE])
    tree = await plan(reference_code=_LEAF_REF, dims={"M": 4},
                      agent=MagicMock(), path="root",
                      runner_fn=fake, retry_budget=3)
    assert tree.is_leaf
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_plan_exhausts_retry_budget_raises():
    fake = _FakeLLM(["nope", "nope", "nope"])
    with pytest.raises(PlannerExhausted, match="root"):
        await plan(reference_code=_LEAF_REF, dims={"M": 4},
                   agent=MagicMock(), path="root",
                   runner_fn=fake, retry_budget=3)
