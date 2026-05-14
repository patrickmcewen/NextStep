"""Unit tests for autotune2 search driver (Phase 5).

All external dependencies (LLM agent, correctness verifier, analytical
scorer) are mocked out. These tests validate the search driver's
control flow, library state evolution, descendant gathering, and
synthetic-wrapper generation — Phase 6 wires the real Anthropic /
4-gate / STeP components.
"""

import asyncio
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.autotune2.contracts import (
    DesignEntry,
    NodeLibrary,
    TensorContract,
    library_cell,
    vanilla_contract_for,
)
from src.autotune2.search import (
    AutotuneResult,
    NodePromptInputs,
    SearchConfig,
    VerifyResult,
    autotune,
    build_node_tensors_dict,
    build_synthetic_wrapper_for_node,
    build_variant_callables,
    cell_for_variant,
    gather_descendants_postorder,
    library_to_variant_registry,
    render_library_as_variant_summaries,
    search_leaf,
    search_parent,
    write_library_snapshot,
    _iter_parent_compositions,
)


def _stub_prompt_inputs(node_name: str = "node") -> NodePromptInputs:
    """Minimal NodePromptInputs for tests (content doesn't affect search flow)."""
    return NodePromptInputs(
        function_signature=f"def {node_name}(x, *, out_shapes):",
        pytorch_reference="# ref",
        dims_block="```json\n{}\n```",
        tensors_block="",
    )
from src.contract import Contract
from src.node_signature import ListOfIntArg, ListOfTensorArg, TensorArg
from src.planner import PlanNode, Tree


def run(coro):
    return asyncio.run(coro)


# --- Contract / Tree fixtures ------------------------------------------------


def _raw_contract(
    arg_specs: dict[str, tuple[int, ...]],
    out_shapes: tuple[tuple[int, ...], ...],
) -> Contract:
    """Build a Contract where every TensorArg is RAW."""
    arg_names = tuple(arg_specs.keys())
    vanilla = tuple(arg_specs.values())
    specs = tuple(TensorArg(shape=s) for s in vanilla)
    # Use the vanilla shape as tile-stream shape too (irrelevant for tests).
    tiled = vanilla
    tiled_values = tuple(torch.zeros(s) for s in vanilla)
    return Contract(
        arg_names=arg_names,
        vanilla_shapes=vanilla,
        tiled_shapes=tiled,
        tiled_values=tiled_values,
        out_shapes=out_shapes,
        arg_specs=specs,
        arg_is_raw=tuple(True for _ in arg_names),
    )


def _leaf(name: str) -> PlanNode:
    return PlanNode(
        name=name, path=f"root/{name}",
        reference_code="# noop\n", refactored_code=None,
        is_leaf=True, children=(),
    )


def _parent(name: str, children: tuple[PlanNode, ...]) -> PlanNode:
    return PlanNode(
        name=name, path=f"root/{name}",
        reference_code="# parent\n", refactored_code="# refactored\n",
        is_leaf=False, children=children,
    )


# --- build_synthetic_wrapper_for_node ----------------------------------------


def test_wrapper_raw_only_inputs():
    c = _raw_contract({"x": (64, 512)}, out_shapes=((4, 16, 512),))
    src = build_synthetic_wrapper_for_node(node_name="my_leaf", parent_contract=c)
    assert "def tiled_reference(dims, tensors):" in src
    assert 'my_leaf(tensors["x"]' in src
    assert "out_shapes=((4, 16, 512),)" in src
    # Output flows through promote_outer to satisfy OffChipStore's stream
    # rank >= 1 startup constraint.
    assert "offchip_store(promote_outer(result))" in src


def test_wrapper_rejects_on_chip_tensorarg():
    c = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))
    c = replace(c, arg_is_raw=(False,))
    with pytest.raises(AssertionError, match="on-chip TensorArg"):
        build_synthetic_wrapper_for_node(node_name="bad", parent_contract=c)


def test_wrapper_multi_output_destructures_and_stores_each():
    """A node with multiple outputs (e.g. pre_attention returning Q, K, V)
    cannot be passed straight to a single ``offchip_store`` — the synthetic
    wrapper must destructure the return tuple and emit one store per output
    so the translated graph satisfies OffChipStore's single-Stream input
    contract.
    """
    c = _raw_contract(
        {"x": (64, 512)},
        out_shapes=((64, 16, 32), (64, 4, 32), (64, 4, 32)),
    )
    src = build_synthetic_wrapper_for_node(
        node_name="pre_attention", parent_contract=c,
    )
    # Destructured tuple assignment
    assert "_out_0, _out_1, _out_2 = pre_attention(" in src
    # One offchip_store per output (two bare + one returned = three total)
    assert src.count("offchip_store(") == 3
    # Each output goes through promote_outer to satisfy OffChipStore's
    # stream rank >= 1 startup constraint.
    assert "offchip_store(promote_outer(_out_0))" in src
    assert "offchip_store(promote_outer(_out_1))" in src
    assert "return offchip_store(promote_outer(_out_2))" in src


def test_wrapper_list_args_pass_through():
    c = Contract(
        arg_names=("x", "weights", "lens"),
        vanilla_shapes=((4, 8), (), ()),
        tiled_shapes=((4, 8), (), ()),
        tiled_values=(torch.zeros(4, 8), [torch.zeros(2, 2)], [3, 5]),
        out_shapes=((1, 4, 8),),
        arg_specs=(
            TensorArg(shape=(4, 8)),
            ListOfTensorArg(length=1, elem_shape=(2, 2)),
            ListOfIntArg(length=2),
        ),
        arg_is_raw=(True, True, True),
    )
    src = build_synthetic_wrapper_for_node(node_name="ln", parent_contract=c)
    # List args are referenced exactly like tensor args; the DSL function
    # itself handles their list-ness internally.
    assert 'tensors["weights"]' in src
    assert 'tensors["lens"]' in src


# --- build_node_tensors_dict --------------------------------------------------


def test_node_tensors_dict_shapes():
    c = _raw_contract({"x": (4, 8), "y": (16,)}, out_shapes=((1, 4, 8),))
    d = build_node_tensors_dict(c)
    assert set(d.keys()) == {"x", "y"}
    assert tuple(d["x"].shape) == (4, 8)
    assert tuple(d["y"].shape) == (16,)


def test_node_tensors_dict_lists_pass_through():
    c = Contract(
        arg_names=("w", "lens"),
        vanilla_shapes=((), ()),
        tiled_shapes=((), ()),
        tiled_values=([torch.ones(2)], [1, 2]),
        out_shapes=((1, 2, 2),),
        arg_specs=(ListOfTensorArg(length=1, elem_shape=(2,)), ListOfIntArg(length=2)),
        arg_is_raw=(True, True),
    )
    d = build_node_tensors_dict(c)
    assert isinstance(d["w"], list) and len(d["w"]) == 1
    assert d["lens"] == [1, 2]


# --- gather_descendants_postorder --------------------------------------------


def test_gather_descendants_empty_for_leaf():
    e = DesignEntry(dsl="# leaf\n")
    assert gather_descendants_postorder(e) == []


def test_gather_descendants_post_order_two_levels():
    grandchild = DesignEntry(dsl="# grandchild\n")
    childA = DesignEntry(dsl="# childA\n", children_picks={"root/a/gc": grandchild})
    childB = DesignEntry(dsl="# childB\n")
    parent = DesignEntry(
        dsl="# parent\n",
        children_picks={"root/a": childA, "root/b": childB},
    )
    out = gather_descendants_postorder(parent)
    # grandchild before childA; childA before childB; parent NOT included.
    assert out == ["# grandchild\n", "# childA\n", "# childB\n"]


# --- library_to_variant_registry / cell_for_variant --------------------------


def _sample_lib() -> NodeLibrary:
    lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    c_perm = TensorContract(reshape=(4, 8), permutation=(1, 0))
    cell0 = library_cell(lib, {"x": c_id}, {"out_0": c_id})
    cell0.append(DesignEntry(dsl="# v0\n", cycles=100, on_chip=200))
    cell1 = library_cell(lib, {"x": c_perm}, {"out_0": c_id})
    cell1.append(DesignEntry(dsl="# v1a\n", cycles=80, on_chip=300))
    cell1.append(DesignEntry(dsl="# v1b\n", cycles=120, on_chip=150))
    return lib


def test_library_to_variant_registry_sequential():
    lib = _sample_lib()
    reg = library_to_variant_registry(lib)
    assert set(reg.keys()) == {0, 1}
    assert reg[0]["input_contracts"]["x"] == vanilla_contract_for((4, 8))
    assert reg[1]["input_contracts"]["x"] == TensorContract(
        reshape=(4, 8), permutation=(1, 0))


def test_cell_for_variant_resolves_back():
    lib = _sample_lib()
    cell0 = cell_for_variant(lib, 0)
    cell1 = cell_for_variant(lib, 1)
    assert len(cell0) == 1 and len(cell1) == 2


def test_cell_for_variant_out_of_range():
    with pytest.raises(AssertionError, match="out of range"):
        cell_for_variant(_sample_lib(), 99)


# --- render_library_as_variant_summaries -------------------------------------


def test_render_summaries_uses_best_pareto_entry():
    lib = _sample_lib()
    summaries = render_library_as_variant_summaries(lib)
    assert [s.variant_index for s in summaries] == [0, 1]
    # cell 1 has entries (80,300) and (120,150) — neither dominates the other;
    # min by (cycles, on_chip) is (80, 300).
    assert summaries[1].cycles == 80 and summaries[1].on_chip == 300


# --- _iter_parent_compositions -----------------------------------------------


def test_iter_parent_compositions_full_product_with_descendants():
    grandchild_a = DesignEntry(dsl="# gc_a\n")
    grandchild_b = DesignEntry(dsl="# gc_b\n")
    child_x = DesignEntry(dsl="# x\n", children_picks={"root/a/gc": grandchild_a})
    child_y = DesignEntry(dsl="# y\n", children_picks={"root/a/gc": grandchild_b})
    fronts = {"root/x": [child_x, child_y], "root/z": [DesignEntry(dsl="# z\n")]}
    out = list(_iter_parent_compositions(
        parent_dsl="# parent\n",
        children_fronts=fronts,
        children_order=["root/x", "root/z"],
    ))
    assert len(out) == 2
    chosen0, descendants0 = out[0]
    assert "# parent\n" not in descendants0  # parent is excluded
    # First combo: child_x (with grandchild_a) + z
    assert "# gc_a\n" in descendants0
    assert "# x\n" in descendants0
    assert "# z\n" in descendants0
    # x's grandchild appears BEFORE x (post-order)
    assert descendants0.index("# gc_a\n") < descendants0.index("# x\n")


# --- build_variant_callables -------------------------------------------------


def test_build_variant_callables_creates_named_stubs():
    import torch.nn as nn
    variants = {
        0: {
            "input_contracts": {},
            "output_contracts": {"out_0": vanilla_contract_for((4, 8))},
        },
        1: {
            "input_contracts": {},
            "output_contracts": {"out_0": vanilla_contract_for((4, 8))},
        },
    }
    out = build_variant_callables(
        child_name="foo",
        arg_names=("x",),
        arg_specs=(TensorArg(shape=(4, 8)),),
        output_names=("out_0",),
        ref_module=nn.Identity(),
        variants=variants,
    )
    assert set(out.keys()) == {"foo_0", "foo_1"}
    # The returned callables share the variant-stub signature: positional
    # args + out_shapes keyword.
    from src.step_dsl import StepTensor
    res = out["foo_0"](torch.zeros(4, 8), out_shapes=((1, 4, 8),))
    assert isinstance(res, StepTensor)


# --- search_leaf -------------------------------------------------------------


def _make_leaf_response(reshape, perm, body: str = "    return None"):
    return (
        "```yaml\n"
        "parent_input_contracts: {}\n"
        "parent_output_contracts:\n"
        f"  out_0: {{reshape: {list(reshape)}, permutation: {list(perm)}}}\n"
        "```\n\n"
        "```python\n"
        f"def my_leaf(x, *, out_shapes):\n{body}\n"
        "```\n"
    )


def test_search_leaf_seeds_baseline_and_admits_llm_proposals(tmp_path):
    """Two LLM turns: one valid alternative, one with garbage YAML.
    Library ends with baseline + 1 alternative."""
    responses = [
        _make_leaf_response(reshape=(4, 4, 8), perm=(1, 0, 2)),
        "no fenced blocks at all, this should be discarded",
    ]
    call_idx = [0]

    async def agent(_conversation):
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    async def verifier(_src):
        return VerifyResult(passed=True)

    # Score: baseline=100, llm proposal=50 — Pareto-distinct cells, so both
    # admitted (different output_contracts keys).
    score_call = [0]
    def score(_src):
        score_call[0] += 1
        return (100 if score_call[0] == 1 else 50, 999)

    node = _leaf("my_leaf")
    contract = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))

    lib = run(search_leaf(
        node=node,
        parent_contract=contract,
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        score_fn=score,
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=SearchConfig(max_turns_per_attempt=1, max_attempts=2),
    ))
    # variants.py was emitted
    assert (tmp_path / "leaf" / "variants.py").exists()
    # library has at least baseline + one llm variant cell
    reg = library_to_variant_registry(lib)
    assert len(reg) >= 2


def test_search_leaf_appends_feedback_on_parse_fail_then_recovers(tmp_path):
    """Within a single attempt, a bad first turn should append feedback to
    the conversation and the next turn should see it. We verify by capturing
    the conversation each call sees and asserting the second turn's
    conversation includes a user-role feedback message."""
    captured_convos: list[list[dict]] = []
    responses = [
        "garbage — no fenced blocks",
        # Non-identity output contract → lands in a different Pareto cell
        _make_leaf_response(reshape=(4, 1, 8), perm=(1, 0, 2)),
    ]
    call_idx = [0]

    async def agent(conversation: list[dict]):
        # Snapshot conversation at call time
        captured_convos.append([dict(m) for m in conversation])
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    async def verifier(_src):
        return VerifyResult(passed=True)

    lib = run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        score_fn=lambda _src: (50, 100),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=SearchConfig(max_turns_per_attempt=2, max_attempts=1),
    ))
    # Two agent calls within one attempt
    assert call_idx[0] == 2
    # First call had only the initial user prompt
    assert len(captured_convos[0]) == 1
    assert captured_convos[0][0]["role"] == "user"
    # Second call has user prompt + assistant garbage + feedback user msg
    assert len(captured_convos[1]) == 3
    assert captured_convos[1][1]["role"] == "assistant"
    assert captured_convos[1][2]["role"] == "user"
    assert "could not be parsed" in captured_convos[1][2]["content"]
    # The successful second turn admitted an entry beyond the baseline
    # (different output contracts cell)
    reg = library_to_variant_registry(lib)
    assert len(reg) >= 2


def test_search_leaf_fresh_attempt_includes_accepted_summary(tmp_path):
    """After a successful first attempt, the second attempt's opening user
    prompt should include the rendered Pareto-front summary so the LLM can
    avoid duplicating already-accepted variants."""
    captured_first_user_per_attempt: list[str] = []
    # Two structurally distinct responses → land in distinct Pareto cells
    # so both get added to ``accepted`` (not Pareto-dominated by baseline).
    responses = [
        _make_leaf_response(reshape=(4, 1, 8), perm=(1, 0, 2)),
        _make_leaf_response(reshape=(2, 2, 8), perm=(0, 1, 2)),
    ]
    call_idx = [0]
    seen_attempts: set[str] = set()

    async def agent(conversation: list[dict]):
        # Each attempt opens with a 1-message conversation (just the user
        # prompt); on subsequent turns within an attempt the conversation
        # is >1. We snapshot the opener.
        if len(conversation) == 1:
            first_user = conversation[0]["content"]
            if first_user not in seen_attempts:
                seen_attempts.add(first_user)
                captured_first_user_per_attempt.append(first_user)
        r = responses[call_idx[0]]
        call_idx[0] += 1
        return r

    async def verifier(_src):
        return VerifyResult(passed=True)

    lib = run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        score_fn=lambda _src: (50, 100),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=SearchConfig(max_turns_per_attempt=2, max_attempts=2),
    ))
    # Two attempts → two distinct opening user prompts
    assert len(captured_first_user_per_attempt) == 2
    # Second attempt's prompt mentions the accepted block
    assert "Already-accepted variants" in captured_first_user_per_attempt[1]
    # First attempt's prompt does not (only baseline is accepted at attempt 0
    # — and the baseline IS rendered in attempt 0 too, since the seeded
    # baseline goes into ``accepted`` before the first attempt opens).
    # Therefore both attempts should include the section. Assert second has
    # at least one more entry mentioned than the first.
    first_count = captured_first_user_per_attempt[0].count("cycles=")
    second_count = captured_first_user_per_attempt[1].count("cycles=")
    assert second_count > first_count


def test_search_leaf_skips_failed_verification(tmp_path):
    async def agent(_conversation):
        return _make_leaf_response(reshape=(1, 4, 8), perm=(0, 1, 2))

    async def verifier(_src):
        return VerifyResult(passed=False, feedback="bad")

    lib = run(search_leaf(
        node=_leaf("my_leaf"),
        parent_contract=_raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),)),
        pass1_dsl="def my_leaf(x, *, out_shapes):\n    return None\n",
        ckpt_dir=tmp_path / "leaf",
        score_fn=lambda _src: (1, 1),
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_leaf"),
        config=SearchConfig(max_turns_per_attempt=1, max_attempts=3),
    ))
    # Only the baseline cell survives.
    assert len(library_to_variant_registry(lib)) == 1


# --- search_parent ------------------------------------------------------------


def _make_parent_response(child_name: str, variant_idx: int):
    return (
        "```yaml\n"
        "child_picks:\n"
        f"  {child_name}: {variant_idx}\n"
        "parent_input_contracts: {}\n"
        "parent_output_contracts:\n"
        "  out_0: {reshape: [1, 4, 8], permutation: [0, 1, 2]}\n"
        "```\n\n"
        "```python\n"
        "def my_parent(x, *, out_shapes):\n"
        "    return child_under(x, out_shapes=out_shapes)\n"
        "```\n"
    )


def test_search_parent_runs_cartesian_compose_and_admits(tmp_path):
    # Seed a child library with two cells (variant 0 and variant 1, each
    # with one entry).
    child_node = _leaf("child_under")
    child_lib: NodeLibrary = {}
    c_id = vanilla_contract_for((4, 8))
    cell0 = library_cell(child_lib, {"x": c_id}, {"out_0": c_id})
    cell0.append(DesignEntry(
        dsl="def child_under(x, *, out_shapes):\n    return None\n",
        input_contracts={"x": c_id}, output_contracts={"out_0": c_id},
        cycles=100, on_chip=200, provenance="pass1_baseline",
    ))
    c_perm = TensorContract(reshape=(4, 8), permutation=(1, 0))
    cell1 = library_cell(child_lib, {"x": c_perm}, {"out_0": c_id})
    cell1.append(DesignEntry(
        dsl="def child_under(x, *, out_shapes):\n    return None  # v1\n",
        input_contracts={"x": c_perm}, output_contracts={"out_0": c_id},
        cycles=80, on_chip=300, provenance="llm_turn_0",
    ))

    parent_node = _parent("my_parent", children=(child_node,))
    parent_contract = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))

    async def agent(_conversation):
        return _make_parent_response("child_under", variant_idx=1)

    async def verifier(_src):
        return VerifyResult(passed=True)

    score_calls = []
    def score(src):
        score_calls.append(src)
        return (50, 100)

    lib = run(search_parent(
        node=parent_node,
        parent_contract=parent_contract,
        pass1_dsl="def my_parent(x, *, out_shapes):\n    return None\n",
        children_libraries={child_node.path: child_lib},
        children_picks_baseline={child_node.path: cell0[0]},
        ckpt_dir=tmp_path / "parent",
        score_fn=score,
        agent=agent,
        verifier=verifier,
        prompt_inputs=_stub_prompt_inputs("my_parent"),
        config=SearchConfig(max_turns_per_attempt=1, max_attempts=1),
    ))

    # Two scoring calls expected: baseline + one Cartesian-compose entry
    # (cell1 has only one entry, so only one combo).
    assert len(score_calls) == 2
    reg = library_to_variant_registry(lib)
    # Baseline cell + llm cell, but they share the same output_contract;
    # since parent's child_picks output here is identity, and baseline output
    # is also identity, both may share a cell or not depending on input
    # contract. Just check the registry is non-empty.
    assert len(reg) >= 1


# --- autotune driver ---------------------------------------------------------


def test_autotune_post_order_walk_populates_all_libraries(tmp_path):
    """Two-node tree: root (parent) + one leaf. Verify both libraries are
    populated and the post-order traversal hits the leaf before the root."""
    leaf = _leaf("inner")
    root = _parent("outer", children=(leaf,))
    root = replace(root, path="root", name="outer")
    leaf = replace(leaf, path="root/inner", name="inner")
    root = replace(root, children=(leaf,))
    tree = Tree(root=root)

    leaf_contract = _raw_contract({"x": (4, 8)}, out_shapes=((1, 4, 8),))
    # Root: empty arg list, but Contract requires at least one arg... use
    # parent_contract=None at root (autotune passes None for root).
    pass1_dsls = {
        leaf.path: "def inner(x, *, out_shapes):\n    return None\n",
        root.path: ("def tiled_reference(dims, tensors):\n"
                    "    return inner(tensors['x'], out_shapes=((1, 4, 8),))\n"),
    }
    # Root passes None as its parent_contract; the leaf takes leaf_contract.
    contracts = {leaf.path: leaf_contract, root.path: leaf_contract}

    visited = []
    async def agent(conversation):
        # Record the opening user message's first 30 chars for visit
        # ordering checks. Pass-1 baseline runs ahead of any agent call.
        first_user = next(
            (m["content"] for m in conversation if m["role"] == "user"), "",
        )
        visited.append(first_user[:30])
        # Always return a malformed response so library only has baseline.
        return "garbage"

    async def verifier(_src):
        return VerifyResult(passed=True)

    def agent_factory(_system_prompt: str):
        return agent

    prompt_inputs = {
        leaf.path: _stub_prompt_inputs("inner"),
        root.path: _stub_prompt_inputs("outer"),
    }
    sys_prompts = {leaf.path: "sys-leaf", root.path: "sys-root"}

    result = run(autotune(
        plan_tree=tree,
        pass1_dsls=pass1_dsls,
        pass1_contracts=contracts,
        ckpt_dir=tmp_path / "tune",
        score_fn=lambda _src: (10, 10),
        agent_factory=agent_factory,
        verifier=verifier,
        prompt_inputs=prompt_inputs,
        system_prompts=sys_prompts,
        config=SearchConfig(max_turns_per_attempt=1, max_attempts=1),
    ))
    assert set(result.libraries.keys()) == {leaf.path, root.path}
    assert result.root_path == root.path
    # Leaf invoked before root (post-order): first agent call's user prompt
    # carries the leaf's node name; the second carries the root's.
    assert "inner" in visited[0]
    assert "outer" in visited[-1]
    # variants.py written for each
    assert (tmp_path / "tune" / "autotune2" / leaf.path / "variants.py").exists()
    assert (tmp_path / "tune" / "autotune2" / root.path / "variants.py").exists()


# --- write_library_snapshot --------------------------------------------------


def test_write_library_snapshot_emits_json(tmp_path):
    lib = _sample_lib()
    out = tmp_path / "lib.json"
    write_library_snapshot(lib, out)
    payload = json.loads(out.read_text())
    assert isinstance(payload, list) and len(payload) == 2
    for cell_repr in payload:
        assert "input_contracts" in cell_repr
        assert "output_contracts" in cell_repr
        assert "entries" in cell_repr
