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
async def test_refactor_tree_uses_verified_cache_to_skip_LLM(tmp_path, monkeypatch):
    """When ``verified_cache`` contains a node's path, ``refactor_tree`` must
    not invoke the LLM for that node, must use the cached DSL when assembling
    parents, and must still refactor any uncached nodes."""
    tree = _two_leaf_tree()
    pass_loop_calls: list[str] = []

    async def fake_pass_loop(*args, **kwargs):
        pass_loop_calls.append(kwargs["kernel_name"])
        return {"success": True, "code": f"# fresh DSL for {kwargs['kernel_name']}",
                "total_tokens": 0}
    monkeypatch.setattr(orch_mod, "_run_pass_loop", fake_pass_loop)

    cache = {
        "root/a": "# CACHED a",
        "root/b": "# CACHED b",
        # 'root' deliberately uncached — must still hit the LLM.
    }
    result = await orch_mod.refactor_tree(
        tree=tree, dims={"M": 4}, root_kernel="kernel_x",
        ckpt_root=tmp_path,
        agent_factory=lambda few_shot: None,
        max_turns=1, log=lambda m: None,
        verified_cache=cache,
    )

    assert result["success"] is True
    assert len(pass_loop_calls) == 1, f"expected 1 LLM call (root only), got {pass_loop_calls!r}"
    assert "_root__" in pass_loop_calls[0]


def test_load_tree_from_dir_round_trips_with_persist_tree(tmp_path):
    """A tree persisted by ``_persist_tree`` and reloaded by
    ``_load_tree_from_dir`` must compare equal node-by-node."""
    tree = _two_leaf_tree()
    iter_dir = tmp_path / "plan" / "iteration_0"
    orch_mod._persist_tree(tree, iter_dir)
    loaded = orch_mod._load_tree_from_dir(tmp_path)

    orig_nodes = list(tree.iter_topological())
    loaded_nodes = list(loaded.iter_topological())
    assert [n.path for n in orig_nodes] == [n.path for n in loaded_nodes]
    for o, l in zip(orig_nodes, loaded_nodes):
        assert o.is_leaf == l.is_leaf
        assert o.reference_code == l.reference_code
        assert o.refactored_code == l.refactored_code


def test_load_verified_dsls_picks_up_PASS_turns(tmp_path):
    """``_load_verified_dsls`` must pick up nodes that have any
    ``refactor_final/turn_*/status.txt == "PASS"``, in either single-attempt or
    multi-attempt layout, and ignore non-PASS turns."""
    refactor = tmp_path / "refactor"

    # Single-attempt layout: refactor/<node_path>/refactor_final/turn_N/
    a_turn = refactor / "root" / "a" / "refactor_final" / "turn_0"
    a_turn.mkdir(parents=True)
    (a_turn / "status.txt").write_text("PASS")
    (a_turn / "extracted_code.py").write_text("# verified A")

    # Multi-attempt layout: refactor/<node_path>/attempt_K/refactor_final/turn_N/
    b_turn = refactor / "root" / "b" / "attempt_2" / "refactor_final" / "turn_3"
    b_turn.mkdir(parents=True)
    (b_turn / "status.txt").write_text("PASS")
    (b_turn / "extracted_code.py").write_text("# verified B")

    # Failing turn — must NOT appear in cache.
    c_turn = refactor / "root" / "c" / "refactor_final" / "turn_0"
    c_turn.mkdir(parents=True)
    (c_turn / "status.txt").write_text("FAIL: mismatch")
    (c_turn / "extracted_code.py").write_text("# bad C")

    cache = orch_mod._load_verified_dsls(tmp_path)
    assert cache == {"root/a": "# verified A", "root/b": "# verified B"}


@pytest.mark.asyncio
async def test_refactor_tree_forwards_translate_fn_as_post_validator(tmp_path, monkeypatch):
    """When ``translate_fn`` is provided, every per-node ``_run_pass_loop``
    must receive a non-None ``post_validator`` so translatability is gated at
    each node (root and non-root) — not just the final translate step."""
    tree = _two_leaf_tree()
    seen_post_validators: list = []

    async def fake_pass_loop(*args, **kwargs):
        seen_post_validators.append(kwargs.get("post_validator"))
        return {"success": True, "code": f"# verified DSL for {kwargs['kernel_name']}",
                "total_tokens": 0}
    monkeypatch.setattr(orch_mod, "_run_pass_loop", fake_pass_loop)

    def fake_translate(_code: str) -> str:
        return "# step graph"

    result = await orch_mod.refactor_tree(
        tree=tree, dims={"M": 4}, root_kernel="kernel_x",
        ckpt_root=tmp_path,
        agent_factory=lambda few_shot: None,
        max_turns=1, log=lambda m: None,
        translate_fn=fake_translate,
    )

    assert result["success"] is True
    assert len(seen_post_validators) == 3, (
        f"expected one call per node (2 leaves + root), got {len(seen_post_validators)}"
    )
    assert all(pv is not None for pv in seen_post_validators), (
        f"every node must get a post_validator when translate_fn is set; got {seen_post_validators!r}"
    )


@pytest.mark.asyncio
async def test_refactor_tree_post_validator_omitted_when_translate_fn_none(tmp_path, monkeypatch):
    """When ``translate_fn`` is None (e.g. ``--translator=llm``), no post-
    validator is attached — preserves the legacy behavior."""
    tree = _two_leaf_tree()
    seen_post_validators: list = []

    async def fake_pass_loop(*args, **kwargs):
        seen_post_validators.append(kwargs.get("post_validator"))
        return {"success": True, "code": "...", "total_tokens": 0}
    monkeypatch.setattr(orch_mod, "_run_pass_loop", fake_pass_loop)

    result = await orch_mod.refactor_tree(
        tree=tree, dims={"M": 4}, root_kernel="kernel_x",
        ckpt_root=tmp_path,
        agent_factory=lambda few_shot: None,
        max_turns=1, log=lambda m: None,
    )
    assert result["success"] is True
    assert all(pv is None for pv in seen_post_validators), seen_post_validators


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
async def test_run_kernel_passes_planner_config_to_each_outer_iteration(tmp_path, monkeypatch):
    """Smoke: run_kernel builds the planner_agent + root_reference once and
    passes them, along with plan_enabled/max_replans/node_attempts, into each
    outer iteration. Each outer is responsible for running its own planner phase."""
    captured: list[dict] = []

    async def fake_outer_iteration(*args, **kwargs):
        captured.append(kwargs)
        return {"success": True, "outer_iteration": 0, "outer_iterations": 1,
                "total_tokens": 0, "total_tool_calls": 0, "cycle_count": 1,
                "final_diagnosis": "ok"}

    monkeypatch.setattr(orch_mod, "_run_outer_iteration", fake_outer_iteration)

    from src.prompts import _load_stepdb_config
    config = _load_stepdb_config()
    kernel = next(iter(config))
    preset = next(iter(config[kernel]["presets"]))

    result = await orch_mod.run_kernel(
        kernel_name=kernel, preset=preset,
        llm_config={"url": "http://x", "api_key": "k", "model": "m"},
        max_outer=2, max_turns=1,
        checkpoint_dir=str(tmp_path),
        plan_enabled=True, max_replans=0, node_attempts=3,
    )
    assert result["success"] is True
    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["plan_enabled"] is True
        assert kwargs["planner_agent"] is not None
        assert isinstance(kwargs["root_reference"], str) and len(kwargs["root_reference"]) > 0
        assert kwargs["max_replans"] == 0
        assert kwargs["node_attempts"] == 3
        # The DSL is produced inside the outer (by the planner phase),
        # so resume_dsl_code is NOT pre-filled at run_kernel level any more.
        assert kwargs.get("resume_dsl_code") is None
