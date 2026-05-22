"""Unit tests for autotune2 persistence (snapshot + transitive stamp)."""

from __future__ import annotations

import json
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
from src.autotune2.persistence import (
    SNAPSHOT_FILENAME,
    compute_node_stamp,
    compute_plan_stamps,
    contract_structural_repr,
    deserialize_library,
    save_library_snapshot,
    serialize_library,
    try_load_library_snapshot,
)
from src.autotune2.sim_manager import AnalyticalOnly
from src.contract import Contract
from src.node_signature import TensorArg
from src.planner import PlanNode, Tree


# --- fixtures ----------------------------------------------------------------


class _TickingClock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        now = self._now
        self._now += 1.0
        return now


def _entry(
    *,
    dsl: str = "def f(): pass",
    cycles: int = 100,
    on_chip: int = 200,
    provenance: str = "test",
    input_contracts: dict[str, TensorContract] | None = None,
    output_contracts: dict[str, TensorContract] | None = None,
    children_picks: dict[str, DesignEntry] | None = None,
) -> DesignEntry:
    return DesignEntry(
        dsl=dsl,
        cycles=cycles,
        on_chip=on_chip,
        provenance=provenance,
        input_contracts=input_contracts or {},
        output_contracts=output_contracts or {},
        children_picks=children_picks or {},
    )


def _identity_in(arg: str, shape: tuple[int, ...]) -> dict[str, TensorContract]:
    return {arg: vanilla_contract_for(shape)}


def _make_plan(root: PlanNode) -> Tree:
    return Tree(root=root)


def _leaf(name: str, path: str | None = None) -> PlanNode:
    return PlanNode(
        name=name, path=path or name, reference_code="# ref",
        refactored_code=None, is_leaf=True, children=(),
    )


def _parent(name: str, children: tuple[PlanNode, ...], path: str | None = None) -> PlanNode:
    return PlanNode(
        name=name, path=path or name, reference_code="# ref",
        refactored_code="# refactor", is_leaf=False, children=children,
    )


def _contract(out_shape: tuple[int, ...] = (1, 4, 8)) -> Contract:
    return Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 8),),
        tiled_shapes=((4, 8),),
        tiled_values=(torch.zeros(4, 8),),
        out_shapes=(out_shape,),
        arg_specs=(TensorArg(shape=(4, 8)),),
        arg_is_raw=(True,),
    )


# --- round-trip --------------------------------------------------------------


def test_serialize_deserialize_round_trip_preserves_entry_fields():
    lib: NodeLibrary = {}
    cell = library_cell(
        lib,
        _identity_in("x", (4, 8)),
        {"out_0": vanilla_contract_for((1, 4, 8))},
    )
    e = _entry(
        dsl="def f(): return 1",
        cycles=42, on_chip=1024, provenance="pass1_baseline",
        input_contracts=_identity_in("x", (4, 8)),
        output_contracts={"out_0": vanilla_contract_for((1, 4, 8))},
    )
    cell.append(e)

    data = serialize_library(lib, stamp="s1", children_libraries={})
    restored = deserialize_library(data, children_libraries={})

    in_key = list(restored.keys())[0]
    out_key = list(restored[in_key].keys())[0]
    [restored_e] = restored[in_key][out_key]
    assert restored_e.dsl == e.dsl
    assert restored_e.cycles == e.cycles
    assert restored_e.on_chip == e.on_chip
    assert restored_e.provenance == e.provenance
    assert restored_e.input_contracts == e.input_contracts
    assert restored_e.output_contracts == e.output_contracts
    assert restored_e.children_picks == {}


def test_serialize_and_deserialize_drop_empty_cells():
    """Empty cells are a degenerate state (no Pareto entries) and trip
    render_library_as_variant_summaries on the very next prompt build.
    serialize_library/deserialize_library must filter them so a snapshot
    written by an earlier buggy version remains loadable."""
    lib: NodeLibrary = {}
    in_a = _identity_in("x", (4, 8))
    out_a = {"out_0": vanilla_contract_for((1, 4, 8))}
    cell_full = library_cell(lib, in_a, out_a)
    cell_full.append(_entry(
        dsl="def f(): return 1",
        provenance="pass1_baseline",
        input_contracts=in_a, output_contracts=out_a,
    ))
    # Empty sibling cell with a different output contract — would survive a
    # naive round-trip and trip the "no empty cells" invariant downstream.
    out_b = {"out_0": vanilla_contract_for((4, 1, 8))}
    library_cell(lib, in_a, out_b)

    data = serialize_library(lib, stamp="s1", children_libraries={})
    assert len(data["cells"]) == 1
    assert data["cells"][0]["entries"]

    # And: a hand-written payload with an empty cell deserializes cleanly.
    data["cells"].append({
        "in_key": data["cells"][0]["in_key"],
        "out_key": [["out_0", {"reshape": [4, 1, 8], "permutation": [0, 1, 2]}]],
        "entries": [],
    })
    restored = deserialize_library(data, children_libraries={})
    assert sum(len(by_out) for by_out in restored.values()) == 1


def test_save_then_load_yields_equivalent_library(tmp_path: Path):
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    cell.append(_entry(provenance="pass1_baseline"))
    cell.append(_entry(cycles=80, on_chip=300, provenance="llm_0"))

    path = tmp_path / SNAPSHOT_FILENAME
    save_library_snapshot(lib, path=path, stamp="abc", children_libraries={})
    loaded = try_load_library_snapshot(
        path, expected_stamp="abc", children_libraries={},
    )
    assert loaded is not None
    [in_key] = list(loaded.keys())
    [out_key] = list(loaded[in_key].keys())
    cell = loaded[in_key][out_key]
    assert len(cell) == 2
    assert {e.provenance for e in cell} == {"pass1_baseline", "llm_0"}


def test_save_uses_atomic_rename(tmp_path: Path):
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    cell.append(_entry())
    path = tmp_path / SNAPSHOT_FILENAME
    save_library_snapshot(lib, path=path, stamp="s", children_libraries={})
    # No leftover .tmp file after a successful save.
    assert not (tmp_path / (SNAPSHOT_FILENAME + ".tmp")).exists()
    assert path.exists()


# --- stamp mismatch ----------------------------------------------------------


def test_try_load_returns_none_when_file_missing(tmp_path: Path):
    result = try_load_library_snapshot(
        tmp_path / "missing.json",
        expected_stamp="anything",
        children_libraries={},
    )
    assert result is None


def test_try_load_returns_none_on_stamp_mismatch(tmp_path: Path):
    lib: NodeLibrary = {}
    library_cell(lib, {}, {}).append(_entry())
    path = tmp_path / SNAPSHOT_FILENAME
    save_library_snapshot(lib, path=path, stamp="old", children_libraries={})
    result = try_load_library_snapshot(
        path, expected_stamp="new", children_libraries={},
    )
    assert result is None


# --- children_picks coordinate rehydration -----------------------------------


def test_children_picks_serialized_as_coords_and_resolved_on_load(tmp_path: Path):
    # Build a child library with two entries at different cell positions.
    child_lib: NodeLibrary = {}
    child_cell = library_cell(child_lib, {}, {})
    child_baseline = _entry(provenance="pass1_baseline", cycles=200, on_chip=200)
    child_llm = _entry(provenance="llm_0", cycles=100, on_chip=300, dsl="def child(): pass")
    child_cell.append(child_baseline)
    child_cell.append(child_llm)

    # Parent entry references the LLM child (idx=1).
    parent_lib: NodeLibrary = {}
    parent_cell = library_cell(parent_lib, {}, {})
    parent_cell.append(_entry(
        provenance="llm_parent",
        children_picks={"child": child_llm},
    ))

    children_libraries = {"child": child_lib}
    path = tmp_path / SNAPSHOT_FILENAME
    save_library_snapshot(
        parent_lib, path=path, stamp="s",
        children_libraries=children_libraries,
    )

    # Inspect the on-disk JSON to confirm coords, not object refs, are saved.
    raw = json.loads(path.read_text())
    [cell_data] = raw["cells"]
    [entry_data] = cell_data["entries"]
    assert entry_data["children_picks"]["child"]["idx"] == 1

    # Reload using the same in-memory child library and confirm reference
    # resolves back to the same object.
    loaded = try_load_library_snapshot(
        path, expected_stamp="s", children_libraries=children_libraries,
    )
    [in_key] = list(loaded.keys())
    [out_key] = list(loaded[in_key].keys())
    [restored_parent] = loaded[in_key][out_key]
    assert restored_parent.children_picks["child"] is child_llm


def test_serialize_raises_when_child_entry_not_in_child_library():
    orphan_child_entry = _entry(provenance="evicted")
    parent_lib: NodeLibrary = {}
    library_cell(parent_lib, {}, {}).append(_entry(
        children_picks={"child": orphan_child_entry},
    ))
    empty_child_lib: NodeLibrary = {}
    with pytest.raises(AssertionError, match="entry not found"):
        serialize_library(
            parent_lib, stamp="s",
            children_libraries={"child": empty_child_lib},
        )


# --- transitive stamp invalidation ------------------------------------------


def test_compute_plan_stamps_changes_when_leaf_dsl_changes():
    leaf = _leaf("a")
    root = _parent("root", (leaf,))
    tree = _make_plan(root)

    pass1 = {"root": "parent_dsl_v1", "a": "leaf_dsl_v1"}
    contracts = {"a": _contract()}

    stamps_v1 = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1, pass1_contracts=contracts, extra={},
    )

    # Change only the leaf — both leaf AND root stamps must change.
    pass1_v2 = {**pass1, "a": "leaf_dsl_v2"}
    stamps_v2 = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1_v2, pass1_contracts=contracts, extra={},
    )

    assert stamps_v1["a"] != stamps_v2["a"]
    assert stamps_v1["root"] != stamps_v2["root"], (
        "Root stamp must invalidate transitively when a descendant DSL "
        "changes — saved root cycles/on_chip were measured against the "
        "old descendant"
    )


def test_compute_plan_stamps_unchanged_root_dsl_still_invalidates_on_extra_change():
    leaf = _leaf("a")
    root = _parent("root", (leaf,))
    tree = _make_plan(root)
    pass1 = {"root": "v1", "a": "v1"}
    contracts = {"a": _contract()}

    s1 = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1, pass1_contracts=contracts,
        extra={"hw_config": {"freq": 1000}},
    )
    s2 = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1, pass1_contracts=contracts,
        extra={"hw_config": {"freq": 2000}},
    )
    assert s1["a"] != s2["a"]
    assert s1["root"] != s2["root"]


def test_compute_node_stamp_sibling_change_does_not_affect_other_sibling():
    a, b = _leaf("a"), _leaf("b")
    root = _parent("root", (a, b))
    tree = _make_plan(root)
    contracts = {"a": _contract(), "b": _contract()}

    pass1 = {"root": "v1", "a": "av1", "b": "bv1"}
    s1 = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1, pass1_contracts=contracts, extra={},
    )
    pass1_b_changed = {**pass1, "b": "bv2"}
    s2 = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1_b_changed, pass1_contracts=contracts, extra={},
    )

    assert s1["a"] == s2["a"], "Sibling change must not invalidate the other leaf"
    assert s1["b"] != s2["b"]
    assert s1["root"] != s2["root"]


def test_contract_structural_repr_is_stable_across_tensor_value_changes():
    # Different tensor identities, same shapes/specs → identical repr.
    c1 = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 8),),
        tiled_shapes=((4, 8),),
        tiled_values=(torch.zeros(4, 8),),  # one tensor
        out_shapes=((1, 4, 8),),
        arg_specs=(TensorArg(shape=(4, 8)),),
        arg_is_raw=(True,),
    )
    c2 = Contract(
        arg_names=("x",),
        vanilla_shapes=((4, 8),),
        tiled_shapes=((4, 8),),
        tiled_values=(torch.ones(4, 8),),  # different tensor
        out_shapes=((1, 4, 8),),
        arg_specs=(TensorArg(shape=(4, 8)),),
        arg_is_raw=(True,),
    )
    assert contract_structural_repr(c1) == contract_structural_repr(c2)


def test_contract_structural_repr_handles_none():
    assert contract_structural_repr(None) is None


# --- end-to-end resume via autotune() driver --------------------------------


def test_autotune_skips_already_completed_nodes_on_resume(tmp_path: Path):
    """End-to-end resume test: run autotune() once, then run it again with
    the same stamps and verify it short-circuits (no agent calls, no new
    turn artifacts written)."""
    import asyncio
    from dataclasses import replace
    from src.autotune2.search import (
        SearchConfig, VerifyResult, autotune,
    )

    leaf_a = _leaf("leaf_a", path="root/leaf_a")
    leaf_b = _leaf("leaf_b", path="root/leaf_b")
    root = _parent("outer", (leaf_a, leaf_b), path="root")
    root = replace(root, children=(leaf_a, leaf_b))
    tree = Tree(root=root)

    leaf_ct = _contract()
    pass1_dsls = {
        leaf_a.path: "def leaf_a(x, *, out_shapes):\n    return None\n",
        leaf_b.path: "def leaf_b(x, *, out_shapes):\n    return None\n",
        root.path: (
            "def tiled_reference(dims, tensors):\n"
            "    return leaf_a(tensors['x'], out_shapes=((1, 4, 8),))\n"
        ),
    }
    contracts = {leaf_a.path: leaf_ct, leaf_b.path: leaf_ct}

    agent_calls = {"n": 0}

    async def agent(_conversation):
        agent_calls["n"] += 1
        return "garbage"  # forces VERIFY_FAIL → only baseline ends up in lib

    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)

    prompts = {
        p: _stub_for_path(p) for p in (leaf_a.path, leaf_b.path, root.path)
    }
    sys_prompts = {p: f"sys-{p}" for p in prompts}

    stamps = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1_dsls, pass1_contracts=contracts,
        extra={},
    )

    def _run():
        return asyncio.run(autotune(
            plan_tree=tree,
            pass1_dsls=pass1_dsls,
            pass1_contracts=contracts,
            ckpt_dir=tmp_path / "tune",
            make_sim_manager=lambda _tensors: AnalyticalOnly(lambda _src: (10, 10)),
            root_tensors={},
            agent_factory=lambda _sp: agent,
            make_verifier=lambda _node, _pc, _t: verifier,
            prompt_inputs=prompts,
            system_prompts=sys_prompts,
            config=SearchConfig(
                time_limit_seconds=1.0,
                attempt_budgets_bytes=[None]*1,
                clock=_TickingClock(),
            ),
            node_stamps=stamps,
        ))

    first = _run()
    first_calls = agent_calls["n"]
    assert first_calls > 0, "first run should have made agent calls"

    # Confirm snapshots exist for every node.
    for p in (leaf_a.path, leaf_b.path, root.path):
        assert (tmp_path / "tune" / "autotune2" / p / SNAPSHOT_FILENAME).exists(), (
            f"missing snapshot for {p}"
        )

    # Second run with identical stamps: agent is not called at all and the
    # returned libraries match the first run's by structure.
    second = _run()
    assert agent_calls["n"] == first_calls, (
        "resume run must skip every node — agent count should be unchanged"
    )
    assert set(second.libraries) == set(first.libraries)


def test_autotune_reruns_node_when_stamp_changes(tmp_path: Path):
    """Changing one node's stamp between runs must invalidate that node (and
    every ancestor under transitive stamping) — they re-run, others skip."""
    import asyncio
    from dataclasses import replace
    from src.autotune2.search import (
        SearchConfig, VerifyResult, autotune,
    )

    leaf_a = _leaf("leaf_a", path="root/leaf_a")
    leaf_b = _leaf("leaf_b", path="root/leaf_b")
    root = _parent("outer", (leaf_a, leaf_b), path="root")
    root = replace(root, children=(leaf_a, leaf_b))
    tree = Tree(root=root)

    leaf_ct = _contract()
    pass1_dsls_v1 = {
        leaf_a.path: "def leaf_a(x, *, out_shapes):\n    return None  # v1\n",
        leaf_b.path: "def leaf_b(x, *, out_shapes):\n    return None\n",
        root.path: (
            "def tiled_reference(dims, tensors):\n"
            "    return leaf_a(tensors['x'], out_shapes=((1, 4, 8),))\n"
        ),
    }
    contracts = {leaf_a.path: leaf_ct, leaf_b.path: leaf_ct}

    # ``make_verifier`` is called exactly once per node that actually runs
    # its search (short-circuited resume nodes never call it), so it's the
    # cleanest "did this node re-run?" probe.
    invoked_nodes: list[str] = []

    def make_verifier(node, _pc, _t):
        invoked_nodes.append(node.path)
        async def verifier(_src, *_a, **_kw):
            return VerifyResult(passed=True)
        return verifier

    async def agent(_conversation):
        return "garbage"

    prompts = {
        p: _stub_for_path(p) for p in (leaf_a.path, leaf_b.path, root.path)
    }
    sys_prompts = {p: "sp" for p in prompts}

    stamps_v1 = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1_dsls_v1, pass1_contracts=contracts,
        extra={},
    )

    def _run(pass1, stamps):
        return asyncio.run(autotune(
            plan_tree=tree, pass1_dsls=pass1, pass1_contracts=contracts,
            ckpt_dir=tmp_path / "tune",
            make_sim_manager=lambda _t: AnalyticalOnly(lambda _s: (10, 10)),
            root_tensors={}, agent_factory=lambda _sp: agent,
            make_verifier=make_verifier,
            prompt_inputs=prompts, system_prompts=sys_prompts,
            config=SearchConfig(
                time_limit_seconds=1.0,
                attempt_budgets_bytes=[None]*1,
                clock=_TickingClock(),
            ),
            node_stamps=stamps,
        ))

    _run(pass1_dsls_v1, stamps_v1)
    invoked_nodes.clear()

    # Change leaf_a only — leaf_a and root must re-run; leaf_b must skip.
    pass1_dsls_v2 = {**pass1_dsls_v1,
                    leaf_a.path: "def leaf_a(x, *, out_shapes):\n    return 0  # v2\n"}
    stamps_v2 = compute_plan_stamps(
        plan_tree=tree, pass1_dsls=pass1_dsls_v2, pass1_contracts=contracts,
        extra={},
    )
    _run(pass1_dsls_v2, stamps_v2)
    assert leaf_a.path in invoked_nodes
    assert root.path in invoked_nodes
    assert leaf_b.path not in invoked_nodes, (
        "leaf_b's stamp is unchanged — it should have been skipped"
    )


def _stub_for_path(path: str):
    """Minimal NodePromptInputs for the integration tests."""
    from src.autotune2.search import NodePromptInputs
    return NodePromptInputs(
        function_signature=f"def {path.replace('/', '_')}(x, *, out_shapes):",
        pytorch_reference="# ref", dims_block="```json\n{}\n```",
        tensors_block="",
    )
