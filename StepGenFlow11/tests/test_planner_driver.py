import pytest
from unittest.mock import MagicMock
from src.planner import plan


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
async def test_plan_exhausts_retry_budget_falls_back_to_leaf():
    """When the LLM can't produce a valid decision after retry_budget attempts,
    the planner declares the node a leaf instead of failing the whole subtree.
    Every retryable failure mode is safely recoverable as a leaf — the leaf
    path bypasses children entirely and just runs the node's own reference
    through the refactor pass."""
    fake = _FakeLLM(["nope", "nope", "nope"])
    tree = await plan(reference_code=_LEAF_REF, dims={"M": 4},
                      agent=MagicMock(), path="root",
                      runner_fn=fake, retry_budget=3)
    assert tree.is_leaf
    assert tree.path == "root"
    assert tree.reference_code == _LEAF_REF
    assert len(fake.calls) == 3


@pytest.mark.asyncio
async def test_plan_max_depth_forces_children_to_leaves_without_llm():
    """With max_depth=1 the root may split, but every child is forced to a
    leaf without consulting the LLM. So the LLM is called exactly once
    (the root's split decision), even if it would otherwise have asked
    each child to split too."""
    fake = _FakeLLM([_SPLIT_RESPONSE])  # only one response — children must not call
    tree = await plan(reference_code=_PARENT_REF, dims={"M": 4},
                      agent=MagicMock(), path="root",
                      runner_fn=fake, retry_budget=3,
                      max_depth=1)
    assert not tree.is_leaf
    assert len(tree.children) == 2
    assert all(c.is_leaf for c in tree.children)
    assert len(fake.calls) == 1  # only root consulted the LLM


@pytest.mark.asyncio
async def test_plan_max_depth_zero_forces_root_to_leaf_without_llm():
    """With max_depth=0 even the root is forced to a leaf without an LLM
    call. Useful as a kill-switch to disable the planner entirely."""
    fake = _FakeLLM([])
    tree = await plan(reference_code=_LEAF_REF, dims={"M": 4},
                      agent=MagicMock(), path="root",
                      runner_fn=fake, retry_budget=3,
                      max_depth=0)
    assert tree.is_leaf
    assert tree.path == "root"
    assert len(fake.calls) == 0


@pytest.mark.asyncio
async def test_plan_writes_per_turn_artifacts_when_turn_root_set(tmp_path):
    """When ``turn_root`` is provided, every LLM turn writes user_prompt.txt,
    response.txt, and status.txt under ``turn_root/<safe_path>/turn_<N>/``."""
    fake = _FakeLLM(["I am thinking out loud", _LEAF_RESPONSE])
    await plan(reference_code=_LEAF_REF, dims={"M": 4},
               agent=MagicMock(), path="root",
               runner_fn=fake, retry_budget=3,
               turn_root=tmp_path)

    turn_0 = tmp_path / "root" / "turn_0"
    turn_1 = tmp_path / "root" / "turn_1"
    assert (turn_0 / "user_prompt.txt").exists()
    assert (turn_0 / "response.txt").read_text() == "I am thinking out loud"
    assert "NO_DECISION" in (turn_0 / "status.txt").read_text()
    assert (turn_1 / "response.txt").read_text() == _LEAF_RESPONSE
    assert (turn_1 / "status.txt").read_text() == "LEAF"


class _FakeReasoningItem:
    def __init__(self, summary_texts):
        class _S:
            def __init__(self, t):
                self.text = t
        self.raw_item = type("R", (), {"summary": [_S(t) for t in summary_texts]})()


class _FakeResultWithReasoning:
    def __init__(self, text, reasoning_chunks):
        self.final_output = text
        from agents import ReasoningItem
        items = []
        for chunks in reasoning_chunks:
            item = _FakeReasoningItem(chunks)
            item.__class__ = type("RI", (_FakeReasoningItem, ReasoningItem), {})
            items.append(item)
        self.new_items = items


class _FakeLLMWithReasoning:
    def __init__(self, results):
        self.results = list(results)

    async def __call__(self, agent, conversation):
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_plan_writes_reasoning_when_present(tmp_path):
    """When the LLM result carries ReasoningItems, the planner writes a
    reasoning.txt alongside response.txt for each turn."""
    fake = _FakeLLMWithReasoning([
        _FakeResultWithReasoning(_LEAF_RESPONSE,
                                  [["I considered splitting but decided leaf."]]),
    ])
    await plan(reference_code=_LEAF_REF, dims={"M": 4},
               agent=MagicMock(), path="root",
               runner_fn=fake, retry_budget=3,
               turn_root=tmp_path)
    reasoning_path = tmp_path / "root" / "turn_0" / "reasoning.txt"
    assert reasoning_path.exists()
    assert "considered splitting" in reasoning_path.read_text()


@pytest.mark.asyncio
async def test_plan_skips_reasoning_when_absent(tmp_path):
    """Non-reasoning models (or fake test results) produce no reasoning items;
    no reasoning.txt should be written in that case."""
    fake = _FakeLLM([_LEAF_RESPONSE])
    await plan(reference_code=_LEAF_REF, dims={"M": 4},
               agent=MagicMock(), path="root",
               runner_fn=fake, retry_budget=3,
               turn_root=tmp_path)
    assert not (tmp_path / "root" / "turn_0" / "reasoning.txt").exists()


@pytest.mark.asyncio
async def test_plan_writes_system_prompt_when_agent_has_instructions(tmp_path):
    """If the agent exposes ``instructions`` (a string), the planner dumps it to
    ``system_prompt.txt`` alongside the per-turn artifacts."""
    fake = _FakeLLM([_LEAF_RESPONSE])
    agent = MagicMock()
    agent.instructions = "You are the planner. Output DECISION: leaf or DECISION: split."
    await plan(reference_code=_LEAF_REF, dims={"M": 4},
               agent=agent, path="root",
               runner_fn=fake, retry_budget=3,
               turn_root=tmp_path)
    sys_prompt = (tmp_path / "root" / "turn_0" / "system_prompt.txt").read_text()
    assert "You are the planner" in sys_prompt


@pytest.mark.asyncio
async def test_plan_writes_split_status_and_recurses_into_child_turn_dirs(tmp_path):
    """A split at root + leaf at each child should yield turn dirs for all 3 nodes."""
    fake = _FakeLLM([_SPLIT_RESPONSE, _LEAF_RESPONSE, _LEAF_RESPONSE])
    await plan(reference_code=_PARENT_REF, dims={"M": 4},
               agent=MagicMock(), path="root",
               runner_fn=fake, retry_budget=3,
               turn_root=tmp_path)

    assert "SPLIT_OK" in (tmp_path / "root" / "turn_0" / "status.txt").read_text()
    assert (tmp_path / "root_doubler" / "turn_0" / "status.txt").read_text() == "LEAF"
    assert (tmp_path / "root_adder" / "turn_0" / "status.txt").read_text() == "LEAF"
