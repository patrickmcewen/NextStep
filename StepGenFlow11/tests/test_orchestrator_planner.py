import asyncio
import pytest

from src.planner import PlanNode, Tree
from src import orchestrator as orch_mod


_LEAF_REF = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return x * 2

def get_inputs(dims):
    torch.manual_seed(101)
    return (torch.randn(dims["M"]),)

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    return Model()(*get_inputs(dims))
"""


def _two_leaf_tree() -> Tree:
    a = PlanNode(name="a", path="root/a", reference_code=_LEAF_REF,
                 refactored_code=None, is_leaf=True, children=())
    b = PlanNode(name="b", path="root/b", reference_code=_LEAF_REF,
                 refactored_code=None, is_leaf=True, children=())
    root = PlanNode(name="root", path="root", reference_code=_LEAF_REF,
                    refactored_code="...", is_leaf=False, children=(a, b))
    return Tree(root=root)


@pytest.mark.asyncio
async def test_refactor_tree_calls_pass_loop_for_each_node_leaves_first(tmp_path, monkeypatch):
    tree = _two_leaf_tree()
    call_order: list[str] = []
    async def fake_pass_loop(*args, **kwargs):
        call_order.append(kwargs["kernel_name"])
        return {"success": True, "code": f"# verified DSL for {kwargs['kernel_name']}",
                "total_tokens": 0}
    monkeypatch.setattr(orch_mod, "_run_pass_loop", fake_pass_loop)

    result = await orch_mod.refactor_tree(
        tree=tree, dims={"M": 4},
        root_kernel="kernel_x",
        ckpt_root=tmp_path,
        agent_factory=lambda few_shot: None,
        max_turns=1, log=lambda m: None,
    )

    assert result["success"] is True
    leaves = [n for n in call_order if "_a__" in n or "_b__" in n]
    roots = [n for n in call_order if "_root__" in n]
    assert all(call_order.index(l) < call_order.index(r)
               for l in leaves for r in roots)


@pytest.mark.asyncio
async def test_refactor_tree_cancels_siblings_on_failure(tmp_path, monkeypatch):
    tree = _two_leaf_tree()
    started: list[str] = []
    async def fake_pass_loop(*args, **kwargs):
        kn = kwargs["kernel_name"]
        started.append(kn)
        if "_a__" in kn:
            return {"success": False, "code": None, "total_tokens": 0,
                    "last_messages": ["err A"]}
        await asyncio.sleep(0.05)
        return {"success": True, "code": "...", "total_tokens": 0}
    monkeypatch.setattr(orch_mod, "_run_pass_loop", fake_pass_loop)

    result = await orch_mod.refactor_tree(
        tree=tree, dims={"M": 4}, root_kernel="kernel_x",
        ckpt_root=tmp_path,
        agent_factory=lambda few_shot: None,
        max_turns=1, log=lambda m: None,
    )

    assert result["success"] is False
    assert "_a__" in result["failing_node"]
