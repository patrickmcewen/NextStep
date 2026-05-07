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


@pytest.mark.asyncio
async def test_planner_phase_re_plans_failing_subtree_then_succeeds(tmp_path, monkeypatch):
    """First refactor pass fails on subtree X; replan emits a different X; second succeeds."""
    from src.planner import PlanNode, Tree

    def _leaf(path):
        return PlanNode(name=path.rsplit("/")[-1], path=path,
                        reference_code=_LEAF_REF, refactored_code=None,
                        is_leaf=True, children=())

    tree_v1 = Tree(root=PlanNode(
        name="root", path="root", reference_code=_LEAF_REF,
        refactored_code="...", is_leaf=False,
        children=(_leaf("root/a"), _leaf("root/b"))))

    tree_v2 = Tree(root=PlanNode(
        name="root", path="root", reference_code=_LEAF_REF,
        refactored_code="...", is_leaf=False,
        children=(_leaf("root/c"), _leaf("root/d"))))

    plan_calls = []
    async def fake_plan_initial(**kwargs):
        plan_calls.append(("initial", kwargs))
        return tree_v1.root
    async def fake_replan(**kwargs):
        plan_calls.append(("replan", kwargs))
        return tree_v2.root

    refactor_attempts = []
    async def fake_refactor_tree(*, tree, **kwargs):
        refactor_attempts.append(tree.root.children[0].name)
        if tree.root.children[0].name == "a":
            return {"success": False, "failing_node": "root/a",
                    "last_messages": ["fail msg"]}
        return {"success": True, "root_dsl": "verified root dsl"}

    monkeypatch.setattr(orch_mod, "_initial_plan", fake_plan_initial)
    monkeypatch.setattr(orch_mod, "_replan", fake_replan)
    monkeypatch.setattr(orch_mod, "refactor_tree", fake_refactor_tree)

    result = await orch_mod._run_planner_phase(
        root_reference="...", dims={"M": 4}, root_kernel="kernel_x",
        ckpt_root=tmp_path, agent_factory=lambda fs: None,
        max_turns=1, log=lambda m: None, max_replans=3,
    )
    assert result["success"] is True
    assert refactor_attempts == ["a", "c"]
    assert plan_calls[0][0] == "initial"
    assert plan_calls[1][0] == "replan"


@pytest.mark.asyncio
async def test_run_kernel_invokes_planner_when_enabled(tmp_path, monkeypatch):
    """Smoke: run_kernel calls _run_planner_phase, then runs outer iterations
    with the verified root DSL as resume_dsl_code."""
    captured = {}
    async def fake_planner_phase(**kwargs):
        captured["called"] = True
        captured["root_kernel"] = kwargs["root_kernel"]
        return {"success": True, "root_dsl": "# verified root dsl"}

    async def fake_outer_iteration(*args, **kwargs):
        captured.setdefault("outer_resume_codes", []).append(kwargs.get("resume_dsl_code"))
        return {"success": True, "outer_iteration": 0, "outer_iterations": 1,
                "total_tokens": 0, "total_tool_calls": 0, "cycle_count": 1,
                "final_diagnosis": "ok"}

    monkeypatch.setattr(orch_mod, "_run_planner_phase", fake_planner_phase)
    monkeypatch.setattr(orch_mod, "_run_outer_iteration", fake_outer_iteration)

    from src.prompts import _load_stepdb_config
    config = _load_stepdb_config()
    kernel = next(iter(config))
    preset = next(iter(config[kernel]["presets"]))

    result = await orch_mod.run_kernel(
        kernel_name=kernel, preset=preset,
        llm_config={"url": "http://x", "api_key": "k", "model": "m"},
        max_outer=1, max_turns=1,
        checkpoint_dir=str(tmp_path),
        plan_enabled=True, max_replans=0,
    )
    assert captured["called"] is True
    assert captured["outer_resume_codes"] == ["# verified root dsl"]
    assert result["success"] is True
