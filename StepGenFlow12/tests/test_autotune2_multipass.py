"""Unit tests for the multi-pass autotune2 extensions (chunks 4 + 5):

  - ``autotune(initial_libraries=..., max_baselines_per_node=..., baseline_selection=..., pass_subdir=...)``
    plumbing.
  - ``run_autotune2._resolve_pass_specs`` config parsing.

These tests use the same stub agent / verifier / score-fn idiom as
``test_autotune2_search.py``; no real LLM / verifier / scorer is
invoked. The goal here is to validate control flow, on-disk layout,
and config resolution — algorithmic behavior of ``select_baselines``
and ``search_leaf`` is covered in their own test files.
"""

import argparse
import asyncio
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.autotune2.contracts import (
    DesignEntry,
    NodeLibrary,
    library_cell,
    vanilla_contract_for,
)
from src.autotune2.search import (
    NodePromptInputs,
    SearchConfig,
    VerifyResult,
    autotune,
)
from src.autotune2.sim_manager import AnalyticalOnly
from src.contract import Contract
from src.node_signature import TensorArg
from src.planner import PlanNode, Tree


# --- shared fixtures (mirrors test_autotune2_search.py) ---------------------


def _stub_prompt_inputs(node_name: str = "node") -> NodePromptInputs:
    return NodePromptInputs(
        function_signature=f"def {node_name}(x, *, out_shapes):",
        pytorch_reference="# ref",
        dims_block="```json\n{}\n```",
        tensors_block="",
    )


def _raw_contract(
    arg_specs: dict[str, tuple[int, ...]],
    out_shapes: tuple[tuple[int, ...], ...],
) -> Contract:
    arg_names = tuple(arg_specs.keys())
    vanilla = tuple(arg_specs.values())
    specs = tuple(TensorArg(shape=s) for s in vanilla)
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


def _leaf(name: str, *, path: str | None = None) -> PlanNode:
    return PlanNode(
        name=name, path=path or f"root/{name}",
        reference_code="# noop\n", refactored_code=None,
        is_leaf=True, children=(),
    )


def _single_leaf_tree() -> tuple[Tree, PlanNode]:
    """Root-as-leaf tree: one node, treated as both root and leaf."""
    n = PlanNode(
        name="solo", path="root",
        reference_code="# noop\n", refactored_code=None,
        is_leaf=True, children=(),
    )
    return Tree(root=n), n


def _async_agent_returning(text: str):
    async def agent(_conversation):
        return text
    return agent


def _async_pass_verifier():
    async def verifier(_src, *_a, **_kw):
        return VerifyResult(passed=True)
    return verifier


def _baseline_invariant_inputs(node, tree):
    """Build the kwargs common to every ``autotune()`` call in these tests."""
    pass1_dsls = {node.path: ("def tiled_reference(dims, tensors):\n"
                              "    return None\n")}
    return dict(
        plan_tree=tree,
        pass1_dsls=pass1_dsls,
        pass1_contracts={},  # root-as-leaf has no parent_contract
        root_tensors={},
        make_sim_manager=lambda _tensors: AnalyticalOnly(lambda _src: (10, 20)),
        agent_factory=lambda _sys: _async_agent_returning("garbage"),
        make_verifier=lambda _node, _pc, _t: _async_pass_verifier(),
        prompt_inputs={node.path: _stub_prompt_inputs("solo")},
        system_prompts={node.path: "sys"},
        config=SearchConfig(
            max_turns_per_attempt=1, attempt_budgets_bytes=[None],
        ),
    )


# --- autotune(initial_libraries=...) plumbing -------------------------------


def test_autotune_default_initial_libraries_preserves_single_pass_layout(tmp_path):
    """Default ``initial_libraries=None`` + ``pass_subdir=None`` writes
    artifacts at ``<ckpt>/autotune2/<node>/`` (no per-pass subdir).
    Verifies the chunk-4 additions don't drift the single-pass layout.
    """
    tree, node = _single_leaf_tree()
    kwargs = _baseline_invariant_inputs(node, tree)
    asyncio.run(autotune(ckpt_dir=tmp_path / "tune", **kwargs))
    assert (tmp_path / "tune" / "autotune2" / node.path / "variants.py").exists()
    assert not (tmp_path / "tune" / "autotune2" / node.path / "pass_0_pass0").exists()


def test_autotune_pass_subdir_lays_artifacts_under_per_pass_dir(tmp_path):
    """``pass_subdir="pass_0_tiling"`` places per-node artifacts under
    ``<ckpt>/autotune2/<node>/pass_0_tiling/``, matching the multi-pass
    node-major layout."""
    tree, node = _single_leaf_tree()
    kwargs = _baseline_invariant_inputs(node, tree)
    asyncio.run(autotune(
        ckpt_dir=tmp_path / "tune", pass_subdir="pass_0_tiling", **kwargs,
    ))
    assert (
        tmp_path / "tune" / "autotune2" / node.path
        / "pass_0_tiling" / "variants.py"
    ).exists()


def test_autotune_threads_initial_baselines_into_search_leaf(tmp_path, monkeypatch):
    """When ``initial_libraries`` is supplied, the per-node search call
    receives ``initial_baselines`` derived via ``select_baselines``,
    with the prior library's pass-1 entry excluded by identity (so
    this pass's re-seeded pass-1 doesn't double-seed)."""
    tree, node = _single_leaf_tree()

    # Build a prior library with one pass-1 entry plus two LLM variants.
    prior_lib: NodeLibrary = {}
    pass1_entry = DesignEntry(
        dsl="def tiled_reference(dims, tensors):\n    return None\n# prior_pass1\n",
        input_contracts={}, output_contracts={},
        cycles=100, on_chip=100, provenance="pass1_baseline",
    )
    llm_a = DesignEntry(
        dsl="def tiled_reference(dims, tensors):\n    return None\n# A\n",
        input_contracts={}, output_contracts={},
        cycles=80, on_chip=120, provenance="llm_baseline_0_attempt_0_binf_turn_0",
    )
    llm_b = DesignEntry(
        dsl="def tiled_reference(dims, tensors):\n    return None\n# B\n",
        input_contracts={}, output_contracts={},
        cycles=120, on_chip=60, provenance="llm_baseline_0_attempt_0_binf_turn_1",
    )
    cell = library_cell(prior_lib, {}, {})
    cell.extend([pass1_entry, llm_a, llm_b])

    # Capture search_leaf's initial_baselines kwarg.
    captured: dict = {}
    from src.autotune2 import search as search_mod
    real_search_leaf = search_mod.search_leaf

    async def spy_search_leaf(**kwargs):
        captured["initial_baselines"] = kwargs.get("initial_baselines")
        return await real_search_leaf(**kwargs)
    monkeypatch.setattr(search_mod, "search_leaf", spy_search_leaf)

    kwargs = _baseline_invariant_inputs(node, tree)
    asyncio.run(autotune(
        ckpt_dir=tmp_path / "tune",
        initial_libraries={node.path: prior_lib},
        max_baselines_per_node=2,
        baseline_selection="pareto_diverse",
        **kwargs,
    ))

    initial_baselines = captured["initial_baselines"]
    assert initial_baselines is not None, "initial_baselines not threaded"
    # 2 prior LLM variants, prior pass-1 excluded — k=2 keeps both A and B
    # since the prior pass-1 is gone and they're both on the Pareto front.
    assert {b.provenance for b in initial_baselines} == {llm_a.provenance, llm_b.provenance}
    # Prior pass-1 must NOT be in the threaded list — that would double-seed.
    for b in initial_baselines:
        assert b.provenance != "pass1_baseline"


def test_autotune_accumulates_prior_pass_library_into_snapshot(tmp_path):
    """The autotune2 library is meant to grow monotonically across passes:
    each pass's snapshot for a node carries forward every variant ever
    admitted, with the prior pass's pass-1 baseline excluded (the current
    pass re-seeds its own) and byte-identical DSL deduped.

    Here we drive autotune with a synthetic prior library holding one
    pass-1 entry and two LLM variants, run a single pass whose agent
    returns garbage (no new admissions), and verify the returned library
    contains the two prior LLM variants — i.e. the merge brought them
    forward even though this pass admitted nothing new.
    """
    tree, node = _single_leaf_tree()

    pass1_entry = DesignEntry(
        dsl="def tiled_reference(dims, tensors):\n    return None\n# prior_pass1\n",
        input_contracts={}, output_contracts={},
        cycles=100, on_chip=100, provenance="pass1_baseline",
    )
    llm_a = DesignEntry(
        dsl="def tiled_reference(dims, tensors):\n    return None\n# A\n",
        input_contracts={}, output_contracts={},
        cycles=80, on_chip=120, provenance="llm_baseline_0_attempt_0_binf_turn_0",
    )
    llm_b = DesignEntry(
        dsl="def tiled_reference(dims, tensors):\n    return None\n# B\n",
        input_contracts={}, output_contracts={},
        cycles=120, on_chip=60, provenance="llm_baseline_0_attempt_0_binf_turn_1",
    )
    prior_lib: NodeLibrary = {}
    cell = library_cell(prior_lib, {}, {})
    cell.extend([pass1_entry, llm_a, llm_b])

    kwargs = _baseline_invariant_inputs(node, tree)
    result = asyncio.run(autotune(
        ckpt_dir=tmp_path / "tune",
        initial_libraries={node.path: prior_lib},
        max_baselines_per_node=2,
        baseline_selection="pareto_diverse",
        pass_subdir="pass_1_test",
        **kwargs,
    ))

    merged = result.libraries[node.path]
    all_provenances = {
        e.provenance
        for by_out in merged.values()
        for cell in by_out.values() for e in cell
    }
    # Current pass re-seeds its own pass-1 baseline.
    assert "pass1_baseline" in all_provenances
    # Prior LLM variants are brought forward by the merge.
    assert llm_a.provenance in all_provenances
    assert llm_b.provenance in all_provenances
    # And only one pass-1 baseline survives the merge (the current pass's).
    pass1_entries = [
        e
        for by_out in merged.values()
        for cell in by_out.values() for e in cell
        if e.provenance == "pass1_baseline"
    ]
    assert len(pass1_entries) == 1, (
        f"expected exactly one pass1_baseline post-merge, got "
        f"{len(pass1_entries)}: {pass1_entries!r}"
    )


def test_autotune_no_initial_libraries_threads_none(tmp_path, monkeypatch):
    """Without ``initial_libraries``, search_leaf receives
    ``initial_baselines=None`` — no per-pass branching."""
    tree, node = _single_leaf_tree()

    captured: dict = {}
    from src.autotune2 import search as search_mod
    real_search_leaf = search_mod.search_leaf

    async def spy_search_leaf(**kwargs):
        captured["initial_baselines"] = kwargs.get("initial_baselines")
        return await real_search_leaf(**kwargs)
    monkeypatch.setattr(search_mod, "search_leaf", spy_search_leaf)

    kwargs = _baseline_invariant_inputs(node, tree)
    asyncio.run(autotune(ckpt_dir=tmp_path / "tune", **kwargs))

    assert captured["initial_baselines"] is None


# --- run_autotune2._resolve_pass_specs --------------------------------------


def _load_run_autotune2_module():
    """Import the runner module by path so its private helpers are testable."""
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_autotune2", repo_root / "run_autotune2.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cli_args(**overrides) -> argparse.Namespace:
    """Stub CLI namespace with only the fields _resolve_pass_specs reads."""
    base = dict(fewshot="tile_shrink", max_turns_per_attempt=16)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_pass_specs_absent_block_synthesizes_single_pass():
    """No ``passes`` key ⇒ one synthesized spec using CLI / library
    defaults. ``is_multi_pass=False`` so the runner keeps the legacy
    flat ckpt layout."""
    mod = _load_run_autotune2_module()
    cfg = {
        "hw_config": {},
        "max_on_chip_memory": 1024,
        "attempt_budgets": [None, 0.5],
    }
    specs, is_multi_pass = mod._resolve_pass_specs(
        cfg, cli_args=_cli_args(), source="test",
    )
    assert is_multi_pass is False
    assert len(specs) == 1
    s = specs[0]
    assert s["name"] == "pass0"
    assert s["fewshot"] == "tile_shrink"
    assert s["max_baselines_per_node"] == 4
    assert s["baseline_selection"] == "pareto_diverse"
    assert s["max_turns_per_attempt"] == 16
    assert s["attempt_budgets_bytes"] == [None, 512]


def test_resolve_pass_specs_multi_pass_inherits_top_level_defaults():
    """Top-level ``fewshot``/``max_baselines_per_node``/etc. fill every
    pass's missing knobs. Per-pass keys override."""
    mod = _load_run_autotune2_module()
    cfg = {
        "hw_config": {},
        "max_on_chip_memory": 1000,
        "attempt_budgets": [None],
        "fewshot": "tile_shrink",
        "max_baselines_per_node": 8,
        "baseline_selection": "min_cycles",
        "passes": [
            {"name": "tiling"},
            {"name": "parallel", "fewshot": "parallel",
             "max_baselines_per_node": 2},
        ],
    }
    specs, is_multi_pass = mod._resolve_pass_specs(
        cfg, cli_args=_cli_args(), source="test",
    )
    assert is_multi_pass is True
    assert len(specs) == 2

    tiling, parallel = specs
    # tiling inherits everything from top-level defaults
    assert tiling["name"] == "tiling"
    assert tiling["fewshot"] == "tile_shrink"
    assert tiling["max_baselines_per_node"] == 8
    assert tiling["baseline_selection"] == "min_cycles"

    # parallel overrides fewshot + max_baselines, inherits selection
    assert parallel["name"] == "parallel"
    assert parallel["fewshot"] == "parallel"
    assert parallel["max_baselines_per_node"] == 2
    assert parallel["baseline_selection"] == "min_cycles"


def test_resolve_pass_specs_per_pass_attempt_budgets_override():
    """Per-pass ``attempt_budgets`` shadows the top-level list and is
    resolved against ``max_on_chip_memory`` independently."""
    mod = _load_run_autotune2_module()
    cfg = {
        "hw_config": {},
        "max_on_chip_memory": 1000,
        "attempt_budgets": [None, 1.0],
        "passes": [
            {"name": "tiling"},  # inherits [None, 1000]
            {"name": "parallel", "attempt_budgets": [0.1, 0.5]},
        ],
    }
    specs, _ = mod._resolve_pass_specs(
        cfg, cli_args=_cli_args(), source="test",
    )
    assert specs[0]["attempt_budgets_bytes"] == [None, 1000]
    assert specs[1]["attempt_budgets_bytes"] == [100, 500]


def test_resolve_pass_specs_unknown_key_asserts():
    """Typos / unrecognized per-pass keys must fail loudly."""
    mod = _load_run_autotune2_module()
    cfg = {
        "hw_config": {},
        "max_on_chip_memory": 1000,
        "attempt_budgets": [None],
        "passes": [{"name": "tiling", "feeshot": "tile_shrink"}],  # typo
    }
    with pytest.raises(AssertionError, match="unknown key"):
        mod._resolve_pass_specs(cfg, cli_args=_cli_args(), source="test")


def test_resolve_pass_specs_missing_name_asserts():
    mod = _load_run_autotune2_module()
    cfg = {
        "hw_config": {},
        "max_on_chip_memory": 1000,
        "attempt_budgets": [None],
        "passes": [{"fewshot": "tile_shrink"}],  # no name
    }
    with pytest.raises(AssertionError, match="missing required 'name'"):
        mod._resolve_pass_specs(cfg, cli_args=_cli_args(), source="test")


def test_resolve_pass_specs_cli_fallback_for_unspecified_top_level():
    """When neither top-level nor per-pass set ``fewshot``, CLI value
    wins as a fallback (legacy single-pass invocation semantics)."""
    mod = _load_run_autotune2_module()
    cfg = {
        "hw_config": {},
        "max_on_chip_memory": 1000,
        "attempt_budgets": [None],
        "passes": [{"name": "only"}],
    }
    specs, _ = mod._resolve_pass_specs(
        cfg, cli_args=_cli_args(fewshot="parallel", max_turns_per_attempt=8),
        source="test",
    )
    assert specs[0]["fewshot"] == "parallel"
    assert specs[0]["max_turns_per_attempt"] == 8
