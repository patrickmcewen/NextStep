"""Unit tests for autotune2 PR3 final_pick.

``final_pick`` is the end-of-run replacement for ``promote_top_k(k=N)``:
exactly one variant is picked from the root Pareto and rust-evaluated so
the resulting number is comparable across every sim_mode the user might
run (HANDOFF design decision #6).
"""

import json

import pytest

from src.autotune2.contracts import (
    DesignEntry,
    NodeLibrary,
    library_cell,
)
from src.autotune2.runtime import (
    RustPromotionResult,
    final_pick,
    promote_top_k,
    write_autotune2_summary,
)
from src.autotune2.search import AutotuneResult


def _root_lib(entries: list[DesignEntry]) -> NodeLibrary:
    """Single-cell root library containing the given DesignEntries."""
    lib: NodeLibrary = {}
    cell = library_cell(lib, {}, {})
    for e in entries:
        cell.append(e)
    return lib


# --- min_cycles, analytical-only library -------------------------------------


def test_final_pick_returns_single_promotion_for_min_cycles():
    lib = _root_lib([
        DesignEntry(dsl="# A\n", cycles=100, on_chip=100, provenance="A"),
        DesignEntry(dsl="# B\n", cycles=50, on_chip=50, provenance="B"),
        DesignEntry(dsl="# C\n", cycles=80, on_chip=30, provenance="C"),
    ])
    calls = []

    def rust(src):
        calls.append(src)
        return (55, 12.5)

    results = final_pick(
        root_library=lib, rust_evaluate_fn=rust, strategy="min_cycles",
    )
    assert len(results) == 1
    assert results[0].entry.provenance == "B"
    assert results[0].rust_cycles == 55
    assert results[0].rust_dur_ms == pytest.approx(12.5)
    # Rust called exactly once — for the chosen analytical entry.
    assert len(calls) == 1


def test_final_pick_sorts_pareto_by_cycles_then_on_chip():
    """Two non-dominated entries (each lower in one axis) — the
    ``min_cycles`` strategy picks the lower-cycle one."""
    lib = _root_lib([
        DesignEntry(dsl="# a\n", cycles=10, on_chip=200, provenance="a"),
        DesignEntry(dsl="# b\n", cycles=20, on_chip=100, provenance="b"),
    ])
    results = final_pick(
        root_library=lib, rust_evaluate_fn=lambda _s: (1, 0.0),
        strategy="min_cycles",
    )
    assert results[0].entry.provenance == "a"


def test_final_pick_ignores_dominated_entries():
    """Only Pareto-non-dominated entries are candidates."""
    lib = _root_lib([
        # Dominated by 'good': 'bad' is worse in BOTH axes.
        DesignEntry(dsl="# bad\n", cycles=5, on_chip=500, provenance="bad"),
        DesignEntry(dsl="# good\n", cycles=4, on_chip=10, provenance="good"),
    ])
    calls = []
    results = final_pick(
        root_library=lib,
        rust_evaluate_fn=lambda s: (calls.append(s) or (99, 1.0)),
        strategy="min_cycles",
    )
    assert results[0].entry.provenance == "good"


# --- mixed-source library ----------------------------------------------------


def test_final_pick_restricts_to_rust_entries_when_library_is_mixed():
    """When any non-dominated entry is rust-sourced, pick must be rust —
    analytical and rust cycles are not directly comparable (HANDOFF design
    decision #5), so falling back to a lower-cycle analytical entry would
    silently mix incommensurable numbers."""
    lib = _root_lib([
        # Lower cycles but analytical — must NOT be picked.
        DesignEntry(
            dsl="# fast_analytical\n", cycles=10, on_chip=500,
            provenance="fast_analytical", cycle_source="analytical",
        ),
        # Higher cycles but rust-measured — picked because the rust axis
        # is the one we trust for the final number.
        DesignEntry(
            dsl="# slow_rust\n", cycles=80, on_chip=20,
            provenance="slow_rust", cycle_source="rust",
        ),
    ])

    def rust(_src):
        raise AssertionError(
            "rust evaluator must not be called when the picked entry was "
            "already rust-measured in-loop"
        )

    results = final_pick(
        root_library=lib, rust_evaluate_fn=rust, strategy="min_cycles",
    )
    assert results[0].entry.provenance == "slow_rust"
    assert results[0].rust_cycles == 80
    # Cached in-loop measurement; rust_dur_ms reads 0.0 to signal "no
    # fresh measurement happened here".
    assert results[0].rust_dur_ms == 0.0


def test_final_pick_among_multiple_rust_entries_picks_min_cycles():
    """When several rust-sourced entries coexist in the Pareto front,
    the lowest-cycles one wins."""
    lib = _root_lib([
        DesignEntry(
            dsl="# r1\n", cycles=200, on_chip=10,
            provenance="r1", cycle_source="rust",
        ),
        DesignEntry(
            dsl="# r2\n", cycles=100, on_chip=50,
            provenance="r2", cycle_source="rust",
        ),
        DesignEntry(
            dsl="# a\n", cycles=5, on_chip=500,
            provenance="a", cycle_source="analytical",
        ),
    ])

    results = final_pick(
        root_library=lib, rust_evaluate_fn=lambda _s: (1, 0.0),
        strategy="min_cycles",
    )
    assert results[0].entry.provenance == "r2"
    assert results[0].rust_cycles == 100


# --- caching of in-loop rust cycles ------------------------------------------


def test_final_pick_reuses_entry_cycles_when_picked_is_rust_source():
    """If the picked entry's ``cycle_source`` is already 'rust', skip the
    fresh rust call — its cycles came from the same simulator in-loop and
    a rerun would burn budget for no gain."""
    lib = _root_lib([
        DesignEntry(
            dsl="# rusty\n", cycles=42, on_chip=10,
            provenance="rusty", cycle_source="rust",
        ),
    ])

    def rust(_s):
        raise AssertionError("rust must not run for rust-sourced picks")

    results = final_pick(
        root_library=lib, rust_evaluate_fn=rust, strategy="min_cycles",
    )
    assert results[0].rust_cycles == 42
    assert results[0].rust_dur_ms == 0.0


def test_final_pick_runs_rust_when_picked_is_analytical():
    """Analytical-sourced picks must trigger one fresh rust call so the
    reported cycle number is rust-measured ground truth."""
    lib = _root_lib([
        DesignEntry(
            dsl="# pure_analytical\n", cycles=99, on_chip=10,
            provenance="pure_analytical", cycle_source="analytical",
        ),
    ])
    calls = []

    def rust(src):
        calls.append(src)
        return (123, 7.0)

    results = final_pick(
        root_library=lib, rust_evaluate_fn=rust, strategy="min_cycles",
    )
    assert len(calls) == 1
    assert results[0].rust_cycles == 123
    assert results[0].rust_dur_ms == pytest.approx(7.0)


# --- composed source includes descendants ------------------------------------


def test_final_pick_composes_descendants_into_source():
    """The root entry's ``children_picks`` chain must be walked so the
    rust evaluator sees the full kernel, not just the root def."""
    leaf = DesignEntry(
        dsl="def leaf_fn():\n    return None\n",
        cycles=1, on_chip=1, provenance="leaf_baseline",
    )
    lib = _root_lib([
        DesignEntry(
            dsl=("def tiled_reference(dims, tensors):\n"
                 "    return offchip_store(leaf_fn())\n"),
            cycles=5, on_chip=5, provenance="root",
            children_picks={"root/leaf": leaf},
        ),
    ])
    captured = []

    def rust(src):
        captured.append(src)
        return (1, 1.0)

    final_pick(
        root_library=lib, rust_evaluate_fn=rust, strategy="min_cycles",
    )
    assert len(captured) == 1
    src = captured[0]
    assert src.index("def leaf_fn") < src.index("def tiled_reference")


# --- surface equivalence with promote_top_k(k=1) -----------------------------


def test_final_pick_surface_equivalent_to_promote_top_k_k1_for_analytical():
    """For an analytical-only library, ``final_pick`` and
    ``promote_top_k(k=1)`` should return the same picked entry and the
    same ``RustPromotionResult`` shape. (The two paths share the Pareto
    pick + rust-evaluation logic; only the cardinality of the result list
    differs in the general case.)"""
    lib = _root_lib([
        DesignEntry(dsl="# a\n", cycles=30, on_chip=80, provenance="a"),
        DesignEntry(dsl="# b\n", cycles=20, on_chip=100, provenance="b"),
        DesignEntry(dsl="# c\n", cycles=40, on_chip=40, provenance="c"),
    ])

    def rust(_src):
        return (1234, 5.0)

    fp = final_pick(
        root_library=lib, rust_evaluate_fn=rust, strategy="min_cycles",
    )
    ptk = promote_top_k(root_library=lib, k=1, rust_evaluate_fn=rust)
    assert len(fp) == 1
    assert len(ptk) == 1
    assert fp[0].entry.provenance == ptk[0].entry.provenance == "b"
    assert fp[0].rust_cycles == ptk[0].rust_cycles == 1234


# --- error handling ----------------------------------------------------------


def test_final_pick_redirects_agent_strategy_to_final_pick_agent():
    """``strategy='agent'`` is implemented by a separate async function
    (``final_pick_agent``) because the curation + final-pick LLM calls
    have to be awaited. The sync ``final_pick`` entry point rejects
    ``strategy='agent'`` with an actionable redirect."""
    lib = _root_lib([
        DesignEntry(dsl="# x\n", cycles=1, on_chip=1, provenance="x"),
    ])
    with pytest.raises(AssertionError, match="final_pick_agent"):
        final_pick(
            root_library=lib, rust_evaluate_fn=lambda _s: (0, 0.0),
            strategy="agent",
        )


def test_final_pick_rejects_unknown_strategy():
    lib = _root_lib([
        DesignEntry(dsl="# x\n", cycles=1, on_chip=1, provenance="x"),
    ])
    with pytest.raises(AssertionError, match="min_cycles"):
        final_pick(
            root_library=lib, rust_evaluate_fn=lambda _s: (0, 0.0),
            strategy="bogus",
        )


def test_final_pick_empty_library_asserts():
    with pytest.raises(AssertionError, match="no entries"):
        final_pick(
            root_library={}, rust_evaluate_fn=lambda _s: (0, 0.0),
            strategy="min_cycles",
        )


# --- summary integration -----------------------------------------------------


def test_write_summary_records_root_pick_strategy(tmp_path):
    lib = _root_lib([
        DesignEntry(dsl="# r\n", cycles=10, on_chip=10, provenance="r"),
    ])
    res = AutotuneResult(libraries={"root": lib}, root_path="root")
    promotions = final_pick(
        root_library=lib, rust_evaluate_fn=lambda _s: (15, 3.0),
        strategy="min_cycles",
    )
    out = tmp_path / "summary.json"
    write_autotune2_summary(
        autotune_result=res, rust_promotions=promotions, out_path=out,
        root_pick_strategy="final_pick:min_cycles",
    )
    payload = json.loads(out.read_text())
    assert payload["root_pick_strategy"] == "final_pick:min_cycles"
    assert payload["best_rust_entry"]["rust_cycles"] == 15
    assert payload["best_rust_entry"]["provenance"] == "r"


def test_write_summary_root_pick_strategy_defaults_to_none(tmp_path):
    """Old call sites that don't pass ``root_pick_strategy`` keep working;
    the field is present in the payload but null."""
    lib = _root_lib([
        DesignEntry(dsl="# r\n", cycles=10, on_chip=10, provenance="r"),
    ])
    res = AutotuneResult(libraries={"root": lib}, root_path="root")
    out = tmp_path / "summary.json"
    write_autotune2_summary(
        autotune_result=res, rust_promotions=[], out_path=out,
    )
    payload = json.loads(out.read_text())
    assert "root_pick_strategy" in payload
    assert payload["root_pick_strategy"] is None


# --- final_pick_agent (PR5) --------------------------------------------------
#
# Tests below cover the agent-driven end-of-run pick:
#   * happy path — curation + final-pick agents return well-formed picks,
#     one rust call follows for an analytical-sourced winner,
#   * single-Pareto short-circuit (no agent calls when there's nothing to
#     choose between),
#   * rust-sourced winner reuses cached cycles (no fresh rust call),
#   * cold-start (no calibration records) — final-pick agent still runs,
#     curation is skipped,
#   * fallback-to-min_cycles on parser failure, RPC failure, and bad
#     ``variant_index``.

import asyncio


def _run(coro):
    return asyncio.run(coro)


def _stub_agent(replies: list):
    """Return an async callable that yields ``replies`` in order.

    Each call asserts there is still a reply to pop so test
    expectations are explicit about call counts. Returned object has
    a ``calls`` list recording each prompt for inspection.
    """
    calls: list = []

    async def fn(conversation):
        calls.append(conversation)
        assert replies, (
            "_stub_agent: no replies left — the agent was called more "
            "times than the test expected"
        )
        return replies.pop(0)

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def test_final_pick_agent_calls_curation_then_pick_and_runs_rust():
    from src.autotune2.runtime import final_pick_agent
    lib = _root_lib([
        DesignEntry(
            dsl="# fastA\n", cycles=10, on_chip=200, provenance="A",
            cycle_source="analytical",
        ),
        DesignEntry(
            dsl="# slowA\n", cycles=80, on_chip=20, provenance="B",
            cycle_source="analytical",
        ),
    ])

    # One calibration record per candidate (curation gets one input,
    # returns it). The composed source is mocked away by the stub
    # fetcher: it returns a synthetic record per call.
    def fetch(_src):
        from src.autotune2.prompts import CurationCandidate
        return [CurationCandidate(
            record_id="recA", composed_source="# rec\n",
            analytical_cycles=10, rust_cycles=20,
            kernel="k", preset="p",
        )]

    # 2 Pareto candidates -> 2 curation calls + 1 final-pick call.
    curation = _stub_agent([
        '```json\n{"record_ids": ["recA"]}\n```',
        '```json\n{"record_ids": ["recA"]}\n```',
    ])
    pick = _stub_agent(
        ['```json\n{"variant_index": 1, "reason": "pick B"}\n```']
    )
    rust_calls: list = []

    def rust(src):
        rust_calls.append(src)
        return (77, 4.0)

    out = _run(final_pick_agent(
        root_library=lib, rust_evaluate_fn=rust,
        curation_agent_fn=curation, final_pick_agent_fn=pick,
        fetch_candidates_fn=fetch,
        root_path="root", kernel="k", preset="p",
    ))
    assert len(out) == 1
    assert out[0].entry.provenance == "B"
    assert out[0].rust_cycles == 77
    assert out[0].rust_dur_ms == pytest.approx(4.0)
    assert len(rust_calls) == 1
    assert len(curation.calls) == 2  # one per Pareto candidate
    assert len(pick.calls) == 1


def test_final_pick_agent_reuses_rust_cycles_when_picked_is_rust_source():
    from src.autotune2.runtime import final_pick_agent
    # Two non-dominated entries; after sort by (cycles, on_chip) the
    # rust-sourced "b" lands at variant_index 0 (lower cycles), so the
    # agent's "variant_index: 0" reply selects the rust entry.
    lib = _root_lib([
        DesignEntry(
            dsl="# a\n", cycles=200, on_chip=10, provenance="a",
            cycle_source="analytical",
        ),
        DesignEntry(
            dsl="# b\n", cycles=50, on_chip=100, provenance="b",
            cycle_source="rust",
        ),
    ])

    def fetch(_src):
        return []  # cold start — curation is skipped

    pick = _stub_agent(
        ['```json\n{"variant_index": 0, "reason": "trust the rust"}\n```']
    )

    def rust(_s):
        raise AssertionError(
            "rust must not run for a rust-sourced pick — cached cycles "
            "are reused"
        )

    out = _run(final_pick_agent(
        root_library=lib, rust_evaluate_fn=rust,
        curation_agent_fn=_stub_agent([]),  # never called
        final_pick_agent_fn=pick,
        fetch_candidates_fn=fetch,
        root_path="root", kernel="k", preset="p",
    ))
    assert out[0].entry.provenance == "b"
    assert out[0].rust_cycles == 50
    assert out[0].rust_dur_ms == 0.0


def test_final_pick_agent_short_circuits_single_pareto_entry():
    """One non-dominated entry → no agent calls; the entry is rust-
    evaluated directly. Saves an LLM round trip in the degenerate case."""
    from src.autotune2.runtime import final_pick_agent
    lib = _root_lib([
        DesignEntry(dsl="# only\n", cycles=42, on_chip=42, provenance="only"),
    ])

    def fetch(_s):
        raise AssertionError("fetch should not run when only one Pareto entry")

    curation = _stub_agent([])
    pick = _stub_agent([])
    rust_called = []

    def rust(src):
        rust_called.append(src)
        return (99, 2.5)

    out = _run(final_pick_agent(
        root_library=lib, rust_evaluate_fn=rust,
        curation_agent_fn=curation, final_pick_agent_fn=pick,
        fetch_candidates_fn=fetch,
        root_path="root", kernel="k", preset="p",
    ))
    assert out[0].entry.provenance == "only"
    assert out[0].rust_cycles == 99
    assert len(rust_called) == 1
    assert curation.calls == [] and pick.calls == []


def test_final_pick_agent_skips_curation_on_cold_start_but_still_picks():
    """Empty candidate set per Pareto entry → curation skipped, final-
    pick agent still runs with empty evidence blocks."""
    from src.autotune2.runtime import final_pick_agent
    lib = _root_lib([
        DesignEntry(dsl="# a\n", cycles=10, on_chip=50, provenance="a"),
        DesignEntry(dsl="# b\n", cycles=20, on_chip=30, provenance="b"),
    ])

    def fetch(_s):
        return []  # cold start

    curation = _stub_agent([])  # must not be called
    pick = _stub_agent(
        ['```json\n{"variant_index": 0, "reason": "pick a"}\n```']
    )
    out = _run(final_pick_agent(
        root_library=lib,
        rust_evaluate_fn=lambda _s: (123, 1.0),
        curation_agent_fn=curation, final_pick_agent_fn=pick,
        fetch_candidates_fn=fetch,
        root_path="root", kernel="k", preset="p",
    ))
    assert out[0].entry.provenance == "a"
    assert out[0].rust_cycles == 123
    assert curation.calls == []
    assert len(pick.calls) == 1


def test_final_pick_agent_falls_back_to_min_cycles_on_parser_failure(capsys):
    """Bad fenced JSON in the final-pick reply → AssertionError inside
    ``parse_final_pick_response`` is caught at the call-site boundary;
    the run falls back to the deterministic ``min_cycles`` pick so a
    reportable number still gets produced. The warning is logged."""
    from src.autotune2.runtime import final_pick_agent
    lib = _root_lib([
        DesignEntry(dsl="# a\n", cycles=10, on_chip=50, provenance="a"),
        DesignEntry(dsl="# b\n", cycles=20, on_chip=30, provenance="b"),
    ])

    pick = _stub_agent(["no fence here, just prose"])
    warnings: list[str] = []

    out = _run(final_pick_agent(
        root_library=lib,
        rust_evaluate_fn=lambda _s: (55, 1.0),
        curation_agent_fn=_stub_agent([]),
        final_pick_agent_fn=pick,
        fetch_candidates_fn=lambda _s: [],
        root_path="root", kernel="k", preset="p",
        log_warning=warnings.append,
    ))
    # min_cycles pick is "a" (lowest cycles).
    assert out[0].entry.provenance == "a"
    assert out[0].rust_cycles == 55
    assert any("final_pick_agent" in w for w in warnings), warnings


def test_final_pick_agent_falls_back_on_out_of_range_variant_index():
    """The parser asserts on out-of-range indices; the outer try/except
    catches and falls back, the run still produces a number."""
    from src.autotune2.runtime import final_pick_agent
    lib = _root_lib([
        DesignEntry(dsl="# a\n", cycles=10, on_chip=50, provenance="a"),
        DesignEntry(dsl="# b\n", cycles=20, on_chip=30, provenance="b"),
    ])
    pick = _stub_agent(
        ['```json\n{"variant_index": 99, "reason": "bad"}\n```']
    )
    warnings: list[str] = []
    out = _run(final_pick_agent(
        root_library=lib,
        rust_evaluate_fn=lambda _s: (60, 1.0),
        curation_agent_fn=_stub_agent([]),
        final_pick_agent_fn=pick,
        fetch_candidates_fn=lambda _s: [],
        root_path="root", kernel="k", preset="p",
        log_warning=warnings.append,
    ))
    assert out[0].entry.provenance == "a"  # min_cycles fallback
    assert warnings


def test_final_pick_agent_falls_back_on_rpc_failure():
    """A non-AssertionError raised by the agent (simulating an RPC
    failure) is also caught and degraded to min_cycles."""
    from src.autotune2.runtime import final_pick_agent
    lib = _root_lib([
        DesignEntry(dsl="# a\n", cycles=10, on_chip=50, provenance="a"),
        DesignEntry(dsl="# b\n", cycles=20, on_chip=30, provenance="b"),
    ])

    async def boom(_):
        raise RuntimeError("rpc dead")

    warnings: list[str] = []
    out = _run(final_pick_agent(
        root_library=lib,
        rust_evaluate_fn=lambda _s: (60, 1.0),
        curation_agent_fn=_stub_agent([]),
        final_pick_agent_fn=boom,
        fetch_candidates_fn=lambda _s: [],
        root_path="root", kernel="k", preset="p",
        log_warning=warnings.append,
    ))
    assert out[0].entry.provenance == "a"
    assert any("rpc dead" in w or "RuntimeError" in w for w in warnings)


def test_final_pick_agent_fallback_respects_mixed_source_pareto():
    """When the agent path fails on a mixed-source library, the
    ``min_cycles`` fallback restricts to rust-sourced entries — same
    rule as ``final_pick(strategy='min_cycles')``."""
    from src.autotune2.runtime import final_pick_agent
    lib = _root_lib([
        DesignEntry(
            dsl="# fast_a\n", cycles=5, on_chip=500,
            provenance="fast_analytical", cycle_source="analytical",
        ),
        DesignEntry(
            dsl="# slow_r\n", cycles=80, on_chip=20,
            provenance="slow_rust", cycle_source="rust",
        ),
    ])

    async def boom(_):
        raise RuntimeError("nope")

    def rust(_s):
        raise AssertionError(
            "rust must not run — the fallback picks the rust-sourced "
            "entry whose cycles are already known"
        )

    out = _run(final_pick_agent(
        root_library=lib, rust_evaluate_fn=rust,
        curation_agent_fn=_stub_agent([]),
        final_pick_agent_fn=boom,
        fetch_candidates_fn=lambda _s: [],
        root_path="root", kernel="k", preset="p",
        log_warning=lambda _w: None,
    ))
    assert out[0].entry.provenance == "slow_rust"
    assert out[0].rust_cycles == 80
    assert out[0].rust_dur_ms == 0.0


# --- final_pick_agent telemetry + capacity knobs (PR6) --------------------


def _load_telemetry_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def test_final_pick_agent_writes_telemetry_on_picked(tmp_path):
    """Happy-path agent pick writes one ``stage='final_pick'`` row with
    decision=='picked', the agent's reason, the curated record ids for
    the picked variant, and both timing fields populated.
    """
    from src.autotune2.agent_telemetry import AgentDecisionStore
    from src.autotune2.prompts import CurationCandidate
    from src.autotune2.runtime import final_pick_agent

    lib = _root_lib([
        DesignEntry(
            dsl="# fastA\n", cycles=10, on_chip=200, provenance="A",
            cycle_source="analytical",
        ),
        DesignEntry(
            dsl="# slowA\n", cycles=80, on_chip=20, provenance="B",
            cycle_source="analytical",
        ),
    ])

    def fetch(_src):
        # Different record_id per candidate so the picked-variant
        # curated_ids field is unambiguously the agent's evidence for
        # the *chosen* candidate. Record ids must be whitespace-free
        # (asserted by build_curation_user_prompt).
        tag = "fast" if "fast" in _src else "slow"
        return [CurationCandidate(
            record_id=f"rec_{tag}",
            composed_source="# rec\n",
            analytical_cycles=10, rust_cycles=20,
            kernel="k", preset="p",
        )]

    curation_replies = [
        '```json\n{"record_ids": ["rec_fast"]}\n```',
        '```json\n{"record_ids": ["rec_slow"]}\n```',
    ]
    curation = _stub_agent(list(curation_replies))
    pick = _stub_agent(
        ['```json\n{"variant_index": 1, "reason": "pick B for thrift"}\n```']
    )

    tel_path = tmp_path / "agent_decisions.jsonl"
    telemetry = AgentDecisionStore(path=tel_path)

    _run(final_pick_agent(
        root_library=lib, rust_evaluate_fn=lambda _s: (77, 4.0),
        curation_agent_fn=curation, final_pick_agent_fn=pick,
        fetch_candidates_fn=fetch,
        root_path="root", kernel="k", preset="p",
        telemetry_store=telemetry,
        hw_config_hash="hwhash", run_id="rid",
    ))

    rows = _load_telemetry_rows(tel_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "final_pick"
    assert row["decision"] == "picked"
    assert row["reason"] == "pick B for thrift"
    assert row["picked_variant_index"] == 1
    assert row["num_pareto_entries"] == 2
    assert row["hw_config_hash"] == "hwhash"
    assert row["run_id"] == "rid"
    # Curated ids on the picked-variant row reflect the second curation
    # call (variant_index=1), not the first one's evidence.
    assert row["curated_record_ids"] == ["rec_slow"]
    # Both stages timed.
    assert row["curation_dur_ms"] >= 0.0
    assert row["decision_dur_ms"] >= 0.0


def test_final_pick_agent_writes_telemetry_on_short_circuit(tmp_path):
    """Single-Pareto short-circuit still records a row tagged
    ``decision='pareto_short_circuit'`` so the audit log shows the
    end-of-run decision regardless of whether an agent ran.
    """
    from src.autotune2.agent_telemetry import AgentDecisionStore
    from src.autotune2.runtime import final_pick_agent

    lib = _root_lib([
        DesignEntry(
            dsl="# only\n", cycles=10, on_chip=100, provenance="only",
            cycle_source="rust",
        ),
    ])

    tel_path = tmp_path / "td.jsonl"
    telemetry = AgentDecisionStore(path=tel_path)

    def rust(_s):
        raise AssertionError("rust must not run for rust-sourced pick")

    _run(final_pick_agent(
        root_library=lib, rust_evaluate_fn=rust,
        curation_agent_fn=_stub_agent([]),  # no curation calls expected
        final_pick_agent_fn=_stub_agent([]),
        fetch_candidates_fn=lambda _s: [],
        root_path="root", kernel="k", preset="p",
        telemetry_store=telemetry,
        hw_config_hash="hw", run_id="rid",
    ))

    rows = _load_telemetry_rows(tel_path)
    assert len(rows) == 1
    assert rows[0]["decision"] == "pareto_short_circuit"
    assert rows[0]["picked_variant_index"] == 0
    assert rows[0]["num_pareto_entries"] == 1
    assert rows[0]["curation_dur_ms"] == -1.0
    assert rows[0]["decision_dur_ms"] == -1.0


def test_final_pick_agent_writes_telemetry_on_fallback(tmp_path):
    """Fallback path on a final-pick parser failure must still emit a
    telemetry row with decision='fallback' so the audit catches it.
    """
    from src.autotune2.agent_telemetry import AgentDecisionStore
    from src.autotune2.runtime import final_pick_agent

    lib = _root_lib([
        DesignEntry(
            dsl="# a\n", cycles=10, on_chip=100, provenance="a",
            cycle_source="rust",
        ),
        DesignEntry(
            dsl="# b\n", cycles=20, on_chip=80, provenance="b",
            cycle_source="rust",
        ),
    ])

    pick = _stub_agent(['not even json'])

    tel_path = tmp_path / "td.jsonl"
    telemetry = AgentDecisionStore(path=tel_path)

    _run(final_pick_agent(
        root_library=lib, rust_evaluate_fn=lambda _s: (0, 0.0),
        curation_agent_fn=_stub_agent([]),
        final_pick_agent_fn=pick,
        fetch_candidates_fn=lambda _s: [],
        root_path="root", kernel="k", preset="p",
        log_warning=lambda _w: None,
        telemetry_store=telemetry,
        hw_config_hash="hw", run_id="rid",
    ))

    rows = _load_telemetry_rows(tel_path)
    assert len(rows) == 1
    assert rows[0]["decision"] == "fallback"
    assert "AssertionError" in rows[0]["reason"] or "json" in rows[0]["reason"]


def test_final_pick_agent_rejects_invalid_capacity_knobs(tmp_path):
    """``curation_k`` must satisfy 1 <= curation_k <= curation_max_candidates,
    asserted up-front so the runner fails fast on bad CLI flags.
    """
    from src.autotune2.runtime import final_pick_agent

    lib = _root_lib([
        DesignEntry(
            dsl="# a\n", cycles=10, on_chip=100, provenance="a",
            cycle_source="rust",
        ),
        DesignEntry(
            dsl="# b\n", cycles=20, on_chip=80, provenance="b",
            cycle_source="rust",
        ),
    ])
    with pytest.raises(AssertionError, match="curation_k"):
        _run(final_pick_agent(
            root_library=lib, rust_evaluate_fn=lambda _s: (0, 0.0),
            curation_agent_fn=_stub_agent([]),
            final_pick_agent_fn=_stub_agent([]),
            fetch_candidates_fn=lambda _s: [],
            root_path="root", kernel="k", preset="p",
            curation_k=10, curation_max_candidates=4,
        ))


def test_final_pick_agent_respects_curation_max_candidates(tmp_path):
    """When the fetcher returns more candidates than the configured cap,
    only the first ``curation_max_candidates`` reach the curation agent.
    """
    from src.autotune2.prompts import CurationCandidate
    from src.autotune2.runtime import final_pick_agent

    lib = _root_lib([
        DesignEntry(
            dsl="# only\n", cycles=10, on_chip=100, provenance="only",
            cycle_source="analytical",
        ),
        DesignEntry(
            dsl="# also\n", cycles=20, on_chip=80, provenance="also",
            cycle_source="analytical",
        ),
    ])

    # Fetcher returns 6 records; cap is 2 → only 2 should reach curation.
    def fetch(_src):
        return [CurationCandidate(
            record_id=f"rec{i:02d}", composed_source=f"# s{i}\n",
            analytical_cycles=10 + i, rust_cycles=20 + i,
            kernel="k", preset="p",
        ) for i in range(6)]

    curation_payloads: list = []
    async def curation(conversation):
        curation_payloads.append(conversation[0]["content"])
        return '```json\n{"record_ids": ["rec00", "rec01"]}\n```'

    pick = _stub_agent(
        ['```json\n{"variant_index": 0, "reason": "ok"}\n```']
    )

    _run(final_pick_agent(
        root_library=lib, rust_evaluate_fn=lambda _s: (1, 0.1),
        curation_agent_fn=curation, final_pick_agent_fn=pick,
        fetch_candidates_fn=fetch,
        root_path="root", kernel="k", preset="p",
        curation_k=2, curation_max_candidates=2,
    ))
    # Two Pareto candidates → two curation calls. Each must only see
    # the first 2 records.
    assert len(curation_payloads) == 2
    for body in curation_payloads:
        assert "rec00" in body and "rec01" in body
        assert "rec02" not in body
