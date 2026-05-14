"""Unit tests for autotune2 Phase 6 runtime: top-K, rust promotion, summary."""

import asyncio
import json
from pathlib import Path

import pytest

from src.autotune2.contracts import (
    DesignEntry,
    NodeLibrary,
    TensorContract,
    library_cell,
    vanilla_contract_for,
)
from src.autotune2.runtime import (
    RustPromotionResult,
    pick_top_k_pareto_entries,
    promote_top_k,
    write_autotune2_summary,
)
from src.autotune2.search import AutotuneResult


# --- pick_top_k_pareto_entries -----------------------------------------------


def _root_lib_with_entries(entries: list[tuple[int, int, str]]) -> NodeLibrary:
    """Build a root library with the given (cycles, on_chip, prov) entries
    all in a single cell."""
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    for cycles, on_chip, prov in entries:
        cell.append(DesignEntry(
            dsl=f"# dsl prov={prov}\n",
            cycles=cycles, on_chip=on_chip, provenance=prov,
        ))
    return lib


def test_pick_top_k_drops_dominated_and_limits_size():
    lib = _root_lib_with_entries([
        (100, 100, "A"),   # dominated by B
        (50, 50, "B"),     # Pareto
        (80, 30, "C"),     # Pareto
        (200, 10, "D"),    # Pareto
        (150, 150, "E"),   # dominated by B
    ])
    picks = pick_top_k_pareto_entries(lib, k=3)
    provs = [e.provenance for e in picks]
    # Non-dominated: B, C, D (3 total)
    assert set(provs) == {"B", "C", "D"}
    # Sorted by (cycles, on_chip)
    assert provs == ["B", "C", "D"]


def test_pick_top_k_truncates_when_more_than_k_non_dominated():
    lib = _root_lib_with_entries([
        (10, 100, "alpha"),
        (20, 80, "beta"),
        (30, 60, "gamma"),
        (40, 40, "delta"),
    ])
    picks = pick_top_k_pareto_entries(lib, k=2)
    provs = [e.provenance for e in picks]
    # All 4 are non-dominated; we keep the lowest-cycle 2
    assert provs == ["alpha", "beta"]


def test_pick_top_k_returns_fewer_when_pareto_smaller_than_k():
    lib = _root_lib_with_entries([(50, 50, "X")])
    picks = pick_top_k_pareto_entries(lib, k=5)
    assert len(picks) == 1


def test_pick_top_k_empty_library_asserts():
    with pytest.raises(AssertionError, match="has no entries"):
        pick_top_k_pareto_entries({}, k=1)


def test_pick_top_k_rejects_k_zero():
    with pytest.raises(AssertionError, match=">= 1"):
        pick_top_k_pareto_entries({}, k=0)


# --- promote_top_k -----------------------------------------------------------


def test_promote_top_k_runs_rust_per_pick_and_sorts_by_rust_cycles():
    lib = _root_lib_with_entries([
        (50, 100, "A"),
        (80, 30, "B"),
    ])
    # The two are non-dominated. Mock rust returns inverted ordering: B beats A.
    rust_calls = []
    def rust(src):
        rust_calls.append(src)
        # First call (A) returns 999, second (B) returns 100.
        return (999 if "A" in src else 100, 12.5)

    results = promote_top_k(root_library=lib, k=2, rust_evaluate_fn=rust)
    assert len(results) == 2
    assert results[0].entry.provenance == "B"
    assert results[0].rust_cycles == 100
    assert results[1].entry.provenance == "A"
    assert results[1].rust_cycles == 999
    assert rust_calls, "rust evaluator must be called per pick"


def test_promote_top_k_includes_descendants_in_composed_source():
    """Root entry with a children_picks reference should produce a composed
    source that includes the descendant DSL before the root DSL."""
    leaf_entry = DesignEntry(
        dsl="def leaf_fn():\n    return None\n",
        cycles=10, on_chip=10, provenance="leaf_baseline",
    )
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    cell.append(DesignEntry(
        dsl=("def tiled_reference(dims, tensors):\n"
             "    return offchip_store(leaf_fn())\n"),
        cycles=20, on_chip=20, provenance="root_baseline",
        children_picks={"root/leaf": leaf_entry},
    ))

    captured = []
    def rust(src):
        captured.append(src)
        return (1, 1.0)

    promote_top_k(root_library=lib, k=1, rust_evaluate_fn=rust)
    assert len(captured) == 1
    src = captured[0]
    # leaf DSL appears before root's tiled_reference
    assert src.index("def leaf_fn") < src.index("def tiled_reference")


# --- write_autotune2_summary -------------------------------------------------


def _trivial_autotune_result() -> AutotuneResult:
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    cell.append(DesignEntry(
        dsl="# root\n", cycles=50, on_chip=100, provenance="pass1_baseline",
    ))
    cell.append(DesignEntry(
        dsl="# root_v1\n", cycles=40, on_chip=120, provenance="llm_turn_0",
    ))
    return AutotuneResult(libraries={"root": lib}, root_path="root")


def test_write_summary_includes_pareto_and_winner(tmp_path):
    res = _trivial_autotune_result()
    promotions = [
        RustPromotionResult(
            entry=next(iter(next(iter(res.libraries["root"].values())).values()))[0],
            rust_cycles=55, rust_dur_ms=20.0, composed_source="<<src>>",
        ),
    ]
    out = tmp_path / "autotune2_summary.json"
    write_autotune2_summary(
        autotune_result=res, rust_promotions=promotions, out_path=out,
    )
    payload = json.loads(out.read_text())
    assert payload["root_path"] == "root"
    assert payload["library_sizes"]["root"] == 1
    # root_pareto sorted by cycles
    assert payload["root_pareto"][0]["cycles"] == 40
    assert payload["root_pareto"][1]["cycles"] == 50
    assert payload["best_rust_entry"]["rust_cycles"] == 55
    # sources NOT included by default
    assert "best_composed_source" not in payload


def test_write_summary_include_sources_writes_source(tmp_path):
    res = _trivial_autotune_result()
    promotions = [
        RustPromotionResult(
            entry=next(iter(next(iter(res.libraries["root"].values())).values()))[0],
            rust_cycles=1, rust_dur_ms=0.0, composed_source="MARKER_SOURCE",
        ),
    ]
    out = tmp_path / "autotune2_summary.json"
    write_autotune2_summary(
        autotune_result=res, rust_promotions=promotions,
        out_path=out, include_sources=True,
    )
    payload = json.loads(out.read_text())
    assert payload["best_composed_source"] == "MARKER_SOURCE"


def test_write_summary_no_promotions_still_valid(tmp_path):
    res = _trivial_autotune_result()
    out = tmp_path / "autotune2_summary.json"
    write_autotune2_summary(
        autotune_result=res, rust_promotions=[], out_path=out,
    )
    payload = json.loads(out.read_text())
    assert payload["best_rust_entry"] is None
    assert payload["rust_winners"] == []
