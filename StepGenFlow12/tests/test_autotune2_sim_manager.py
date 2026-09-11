"""Unit tests for the autotune2 simulation manager (PR2 RustAll + TimeBudget).

PR1 introduced the ``SimulationManager`` protocol + ``AnalyticalOnly`` impl;
those control-flow paths are exercised indirectly by ``test_autotune2_search``.
This file covers the PR2 additions:

  - ``TimeBudget``: arithmetic, ``recent_avg_seconds`` window, async-safe
    ``consume``, ``reset`` semantics across passes, ``inf`` for unlimited.
  - ``RustAll``: analytical-first-then-rust ordering, budget-exhausted
    fallback to analytical, calibration record append on success,
    content-addressed source persistence (one file per unique source),
    error-feedback propagation when the analytical scorer raises.
  - ``AnalyticalOnly.start_pass``: no-op shape conformance.

All external dependencies are stubbed — no rust subprocess, no
real timing model, no Anthropic SDK. Tests run under ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from src.autotune2.calibration import CalibrationRecord, CalibrationStore
from src.autotune2.sim_manager import (
    AgentManager,
    AnalyticalOnly,
    DeterministicSplit,
    RustAll,
    SimContext,
    SimulationResult,
    TimeBudget,
)


def _run(coro):
    return asyncio.run(coro)


def _ctx(
    node_path: str = "root",
    *,
    is_root: bool = True,
    turn_index: int = 0,
) -> SimContext:
    return SimContext(
        node_path=node_path,
        variant_kind="variant",
        is_root=is_root,
        attempt_index=0,
        turn_index=turn_index,
    )


# ---------------------------------------------------------------------------
# TimeBudget
# ---------------------------------------------------------------------------


def test_time_budget_unlimited_reports_inf_and_never_trips_cutoff():
    b = TimeBudget(total_seconds=None)
    assert b.remaining_seconds == float("inf")
    # Even after consuming significant time, an unlimited budget still
    # reports inf — the canonical ``<= 0`` cutoff must never fire for
    # the unlimited case.
    _run(b.consume(123.45))
    assert b.remaining_seconds == float("inf")


def test_time_budget_remaining_decreases_with_consume():
    b = TimeBudget(total_seconds=10.0)
    assert b.remaining_seconds == pytest.approx(10.0)
    _run(b.consume(3.5))
    assert b.consumed_seconds == pytest.approx(3.5)
    assert b.remaining_seconds == pytest.approx(6.5)


def test_time_budget_recent_avg_windowed():
    # _RECENT_WINDOW is 8; older entries roll off.
    b = TimeBudget(total_seconds=100.0)
    assert b.recent_avg_seconds == 0.0  # no entries
    for s in (1.0, 1.0, 1.0):
        _run(b.consume(s))
    assert b.recent_avg_seconds == pytest.approx(1.0)
    # Now push 8 new entries each equal to 5.0; the older 1.0s should be
    # evicted from the deque.
    for _ in range(8):
        _run(b.consume(5.0))
    assert b.recent_avg_seconds == pytest.approx(5.0)


def test_time_budget_reset_clears_consumed_and_recent():
    b = TimeBudget(total_seconds=10.0)
    _run(b.consume(2.0))
    _run(b.consume(3.0))
    assert b.consumed_seconds == pytest.approx(5.0)
    b.reset(20.0)
    assert b.total_seconds == pytest.approx(20.0)
    assert b.consumed_seconds == 0.0
    assert b.remaining_seconds == pytest.approx(20.0)
    assert b.recent_avg_seconds == 0.0  # cleared on reset


def test_time_budget_concurrent_consume_does_not_lose_writes():
    b = TimeBudget(total_seconds=100.0)

    async def worker(amount):
        await b.consume(amount)

    async def race():
        await asyncio.gather(*[worker(0.1) for _ in range(50)])

    _run(race())
    # 50 concurrent consume(0.1) — under the asyncio.Lock the sum must
    # be exact, not best-effort.
    assert b.consumed_seconds == pytest.approx(5.0)


def test_time_budget_rejects_negative_seconds():
    with pytest.raises(AssertionError, match="non-negative"):
        TimeBudget(total_seconds=-1.0)
    b = TimeBudget(total_seconds=10.0)
    with pytest.raises(AssertionError, match="non-negative"):
        _run(b.consume(-0.5))
    with pytest.raises(AssertionError, match="non-negative"):
        b.reset(-1.0)


# ---------------------------------------------------------------------------
# AnalyticalOnly.start_pass shape
# ---------------------------------------------------------------------------


def test_analytical_only_start_pass_is_noop():
    # The Protocol now requires start_pass; AnalyticalOnly's impl must
    # accept any value (including None and floats) without raising.
    mgr = AnalyticalOnly(score_fn=lambda _src: (10, 20))
    _run(mgr.start_pass(None))
    _run(mgr.start_pass(0.0))
    _run(mgr.start_pass(123.45))
    # ...and scoring still works after start_pass.
    result = _run(mgr.score(_ctx(), "def foo(): pass"))
    assert result.cycle_source == "analytical"
    assert result.cycles == 10
    assert result.on_chip == 20


# ---------------------------------------------------------------------------
# RustAll
# ---------------------------------------------------------------------------


def _make_rust_all(
    tmp_path: Path,
    *,
    score_fn=None,
    rust_evaluate_fn=None,
    total_seconds: float | None = None,
) -> tuple[RustAll, TimeBudget, CalibrationStore, Path]:
    if score_fn is None:
        score_fn = lambda _src: (100, 200)  # noqa: E731
    if rust_evaluate_fn is None:
        rust_evaluate_fn = lambda _src: (500, 12.34)  # noqa: E731
    budget = TimeBudget(total_seconds=total_seconds)
    store = CalibrationStore(path=tmp_path / "calibration.jsonl")
    sources_dir = tmp_path / "calibration_sources"
    mgr = RustAll(
        score_fn=score_fn,
        rust_evaluate_fn=rust_evaluate_fn,
        time_budget=budget,
        calibration_store=store,
        sources_dir=sources_dir,
        kernel="test_kernel",
        preset="test_preset",
        hw_config_hash="deadbeef" * 2,
        compute_bw=100_000,
        run_id="test_run",
    )
    return mgr, budget, store, sources_dir


def test_rust_all_success_records_rust_cycles_and_analytical_on_chip(tmp_path):
    mgr, budget, store, sources_dir = _make_rust_all(tmp_path)
    result = _run(mgr.score(_ctx(), "def src(): return 1"))
    # Cycles come from rust; on_chip from analytical. cycle_source = rust.
    assert result.cycle_source == "rust"
    assert result.cycles == 500
    assert result.on_chip == 200
    assert result.rust_dur_ms == pytest.approx(12.34)
    assert result.error_feedback is None
    # Budget consumed > 0 (real wall-clock from asyncio.to_thread on
    # the fake evaluator). Exact value isn't pin-down-able but must
    # be non-negative.
    assert budget.consumed_seconds >= 0.0
    # Calibration record appended.
    records = list(store.iter_records())
    assert len(records) == 1
    rec = records[0]
    assert rec.kernel == "test_kernel"
    assert rec.preset == "test_preset"
    assert rec.analytical_cycles == 100
    assert rec.analytical_on_chip == 200
    assert rec.rust_cycles == 500
    assert rec.rust_dur_ms == pytest.approx(12.34)
    assert rec.run_id == "test_run"
    assert rec.compute_bw == 100_000
    # composed_source_path points at a sha256-named file under sources_dir.
    p = Path(rec.composed_source_path)
    assert p.parent == sources_dir
    digest = hashlib.sha256(b"def src(): return 1").hexdigest()
    assert p.name == f"{digest}.py"
    assert p.read_text() == "def src(): return 1"


def test_rust_all_passes_node_path_and_turn_label_to_rust(tmp_path):
    captured = {}

    def rust(_src, **kwargs):
        captured.update(kwargs)
        return (500, 12.34)

    mgr, _budget, _store, _sources_dir = _make_rust_all(
        tmp_path, rust_evaluate_fn=rust, total_seconds=60.0,
    )
    turn_dir = (
        tmp_path / "autotune2" / "root" / "moe" / "pass_0_general"
        / "baseline_0_attempt_1_b5000000" / "session_0" / "turn_1"
    )
    ctx = SimContext(
        node_path="root/moe",
        variant_kind="variant",
        is_root=False,
        attempt_index=1,
        turn_index=1,
        turn_artifact_dir=turn_dir,
    )

    result = _run(mgr.score(ctx, "def src(): return 1"))

    assert result.cycle_source == "rust"
    assert captured["node_path"] == "root/moe"
    assert captured["run_label"] == (
        "pass_0_general/baseline_0_attempt_1_b5000000/session_0/turn_1"
    )


def test_rust_all_falls_back_to_analytical_when_budget_exhausted(tmp_path):
    # total_seconds=0 ⇒ remaining_seconds == 0 from the first call;
    # the manager should skip rust and return the analytical result.
    rust_called = {"n": 0}

    def fake_rust(_src):
        rust_called["n"] += 1
        return (999, 1.0)

    mgr, budget, store, _ = _make_rust_all(
        tmp_path, rust_evaluate_fn=fake_rust, total_seconds=0.0,
    )
    result = _run(mgr.score(_ctx(), "def s(): return 2"))
    assert result.cycle_source == "analytical"
    assert result.cycles == 100  # analytical estimate, not rust
    assert result.on_chip == 200
    assert rust_called["n"] == 0
    # No calibration record was written — only successful rust calls
    # produce records.
    assert list(store.iter_records()) == []


def test_rust_all_propagates_analytical_failure_as_feedback(tmp_path):
    def bad_score(_src):
        raise RuntimeError("scorer-blew-up")

    rust_called = {"n": 0}

    def fake_rust(_src):
        rust_called["n"] += 1
        return (1, 1.0)

    mgr, _budget, store, _ = _make_rust_all(
        tmp_path, score_fn=bad_score, rust_evaluate_fn=fake_rust,
        total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def s(): return 3"))
    assert result.cycles is None
    assert result.on_chip is None
    assert result.cycle_source is None
    assert result.error_feedback is not None
    assert "scorer-blew-up" in result.error_feedback
    # Rust must NOT be called when analytical fails — a variant the
    # analytical model can't even shape-check is unlikely to yield
    # useful rust evidence and would waste budget.
    assert rust_called["n"] == 0
    assert list(store.iter_records()) == []


def test_rust_all_converts_rust_simulator_failure_to_score_feedback(tmp_path):
    def bad_rust(_src):
        raise AssertionError(
            "build_rust_evaluate_fn: evaluate_kernel failed at stage "
            "'simulate': min_rank must be less than max_rank"
        )

    mgr, budget, store, _ = _make_rust_all(
        tmp_path, rust_evaluate_fn=bad_rust, total_seconds=60.0,
    )

    result = _run(mgr.score(_ctx(), "def s(): return 3"))

    assert result.cycles is None
    assert result.on_chip is None
    assert result.cycle_source is None
    assert result.error_feedback is not None
    assert "Rust simulator failed" in result.error_feedback
    assert "min_rank must be less than max_rank" in result.error_feedback
    assert budget.consumed_seconds >= 0.0
    assert list(store.iter_records()) == []


def test_rust_all_source_persistence_is_idempotent_for_duplicate_sources(tmp_path):
    mgr, _budget, store, sources_dir = _make_rust_all(
        tmp_path, total_seconds=60.0,
    )
    _run(mgr.score(_ctx(), "def s(): return 4"))
    _run(mgr.score(_ctx(), "def s(): return 4"))  # same source as above
    _run(mgr.score(_ctx(), "def s(): return 5"))  # different source
    # Two unique source files on disk (two distinct sha256 hashes).
    assert sorted(p.suffix for p in sources_dir.iterdir()) == [".py", ".py"]
    files = list(sources_dir.iterdir())
    assert len(files) == 2
    # Three calibration records (one per rust call).
    assert len(list(store.iter_records())) == 3


def test_rust_all_breakdown_threaded_through_when_scorer_exposes_it(tmp_path):
    score_fn = lambda _src: (100, 200)  # noqa: E731
    score_fn.breakdown = lambda _src: "per-op-bytes-report"

    mgr, _budget, _store, _ = _make_rust_all(
        tmp_path, score_fn=score_fn, total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def s(): pass"))
    assert result.breakdown == "per-op-bytes-report"


def test_rust_all_start_pass_resets_shared_budget(tmp_path):
    mgr, budget, _store, _ = _make_rust_all(tmp_path, total_seconds=5.0)
    _run(mgr.score(_ctx(), "def s(): return 6"))
    consumed_after_one_call = budget.consumed_seconds
    assert consumed_after_one_call >= 0.0
    # New pass starts: start_pass(...) re-arms the shared budget with
    # new seconds, clearing consumed.
    _run(mgr.start_pass(30.0))
    assert budget.consumed_seconds == 0.0
    assert budget.total_seconds == pytest.approx(30.0)
    assert budget.remaining_seconds == pytest.approx(30.0)


def test_rust_all_start_pass_unlimited_is_inf(tmp_path):
    mgr, budget, _store, _ = _make_rust_all(tmp_path, total_seconds=10.0)
    _run(mgr.start_pass(None))
    assert budget.total_seconds is None
    assert budget.remaining_seconds == float("inf")


def test_rust_all_concurrent_scores_share_one_budget(tmp_path):
    # Multiple node tasks scoring concurrently against one RustAll
    # share one TimeBudget (the production wiring builds one
    # TimeBudget per pass and reuses it via the factory closure).
    # Concurrent consume() must not lose writes.
    consumed_seen: list[float] = []

    def fake_rust(_src):
        return (42, 1.0)

    mgr, budget, store, _ = _make_rust_all(
        tmp_path, rust_evaluate_fn=fake_rust, total_seconds=1000.0,
    )

    async def race():
        await asyncio.gather(*[
            mgr.score(_ctx(), f"def s_{i}(): pass")
            for i in range(10)
        ])
        consumed_seen.append(budget.consumed_seconds)

    _run(race())
    # 10 rust calls → 10 calibration records, every consume() landed.
    assert len(list(store.iter_records())) == 10
    assert consumed_seen[0] >= 0.0


# ---------------------------------------------------------------------------
# Calibration record round-trip via the store
# ---------------------------------------------------------------------------


def test_calibration_records_round_trip_through_jsonl(tmp_path):
    mgr, _budget, store, _ = _make_rust_all(tmp_path, total_seconds=60.0)
    _run(mgr.score(_ctx("root/leaf_a", is_root=False), "def s(): return 7"))
    _run(mgr.score(_ctx("root/leaf_b", is_root=False), "def s(): return 8"))
    records = list(store.iter_records())
    assert {r.node_path for r in records} == {"root/leaf_a", "root/leaf_b"}
    assert all(r.is_root is False for r in records)
    # JSONL file is well-formed: every line parses as JSON with the
    # expected schema (asserts no torn lines under the per-record
    # atomic-append guarantee).
    text = (store.path).read_text().splitlines()
    assert len(text) == 2
    for line in text:
        parsed = json.loads(line)
        assert parsed["kernel"] == "test_kernel"
        assert "composed_source_path" in parsed


# ---------------------------------------------------------------------------
# DeterministicSplit (PR4)
# ---------------------------------------------------------------------------


def _baseline_ctx(node_path: str = "root", *, is_root: bool = True) -> SimContext:
    return SimContext(
        node_path=node_path, variant_kind="baseline",
        is_root=is_root, attempt_index=-1, turn_index=-1,
    )


def _make_deterministic_split(
    tmp_path: Path,
    *,
    score_fn=None,
    rust_evaluate_fn=None,
    total_seconds: float | None = None,
    rule: str = "rust_baselines",
):
    if score_fn is None:
        score_fn = lambda _s: (100, 200)  # noqa: E731
    if rust_evaluate_fn is None:
        rust_evaluate_fn = lambda _s: (500, 12.34)  # noqa: E731
    budget = TimeBudget(total_seconds=total_seconds)
    store = CalibrationStore(path=tmp_path / "calibration.jsonl")
    sources_dir = tmp_path / "calibration_sources"
    mgr = DeterministicSplit(
        score_fn=score_fn, rust_evaluate_fn=rust_evaluate_fn,
        time_budget=budget, calibration_store=store,
        sources_dir=sources_dir, kernel="k", preset="p",
        hw_config_hash="cafef00d" * 2, compute_bw=100_000,
        run_id="test_run", rule=rule,
    )
    return mgr, budget, store, sources_dir


def test_deterministic_split_rusts_baselines(tmp_path):
    mgr, _b, store, _ = _make_deterministic_split(tmp_path, total_seconds=60.0)
    result = _run(mgr.score(_baseline_ctx(), "def s(): pass"))
    assert result.cycle_source == "rust"
    assert result.cycles == 500
    # Calibration record written.
    assert len(list(store.iter_records())) == 1


def test_deterministic_split_keeps_variants_analytical(tmp_path):
    """Default rule rust_baselines: anything tagged 'variant' stays
    analytical even when the budget would allow rust."""
    mgr, _b, store, _ = _make_deterministic_split(tmp_path, total_seconds=60.0)
    result = _run(mgr.score(_ctx(), "def s(): pass"))
    assert result.cycle_source == "analytical"
    assert result.cycles == 100  # the analytical number
    # No rust call → no calibration record.
    assert list(store.iter_records()) == []


def test_deterministic_split_budget_exhaustion_degrades_baseline(tmp_path):
    """Even a baseline falls back to analytical when the per-pass time
    budget is at zero — same hard cutoff as RustAll."""
    mgr, _b, _store, _ = _make_deterministic_split(tmp_path, total_seconds=0.0)
    result = _run(mgr.score(_baseline_ctx(), "def s(): pass"))
    assert result.cycle_source == "analytical"


def test_deterministic_split_rejects_unknown_rule(tmp_path):
    with pytest.raises(AssertionError, match="rule"):
        _make_deterministic_split(tmp_path, rule="invented_rule")


def test_deterministic_split_propagates_analytical_failure_as_feedback(tmp_path):
    """Analytical scorer crash on a baseline still surfaces as feedback —
    same shape as RustAll / AnalyticalOnly."""
    def boom(_src):
        raise RuntimeError("scorer exploded")
    mgr, _b, _s, _ = _make_deterministic_split(
        tmp_path, score_fn=boom, total_seconds=60.0,
    )
    result = _run(mgr.score(_baseline_ctx(), "def s(): pass"))
    assert result.cycles is None
    assert result.cycle_source is None
    assert result.error_feedback is not None
    assert "Analytical scorer failed" in result.error_feedback


def test_deterministic_split_converts_rust_simulator_failure_to_score_feedback(
    tmp_path,
):
    def bad_rust(_src):
        raise RuntimeError("simulate crashed")

    mgr, _b, store, _ = _make_deterministic_split(
        tmp_path, rust_evaluate_fn=bad_rust, total_seconds=60.0,
    )

    result = _run(mgr.score(_baseline_ctx(), "def s(): pass"))

    assert result.cycles is None
    assert result.cycle_source is None
    assert result.error_feedback is not None
    assert "Rust simulator failed" in result.error_feedback
    assert "simulate crashed" in result.error_feedback
    assert list(store.iter_records()) == []


def test_deterministic_split_start_pass_resets_shared_budget(tmp_path):
    mgr, budget, _s, _ = _make_deterministic_split(tmp_path, total_seconds=10.0)
    _run(budget.consume(7.0))
    assert budget.consumed_seconds == pytest.approx(7.0)
    _run(mgr.start_pass(30.0))
    assert budget.total_seconds == pytest.approx(30.0)
    assert budget.consumed_seconds == 0.0


# ---------------------------------------------------------------------------
# AgentManager (PR4)
# ---------------------------------------------------------------------------


def _curation_reply(record_ids: list[str]) -> str:
    """Fake well-formed curation-agent JSON reply."""
    return (
        '```json\n'
        + json.dumps({"record_ids": record_ids})
        + '\n```'
    )


def _decision_reply(decision: str, reason: str = "ok") -> str:
    """Fake well-formed sim-decision-agent JSON reply."""
    return (
        '```json\n'
        + json.dumps({"decision": decision, "reason": reason})
        + '\n```'
    )


def _make_agent_manager(
    tmp_path: Path,
    *,
    score_fn=None,
    rust_evaluate_fn=None,
    total_seconds: float | None = None,
    curation_fn=None,
    decision_fn=None,
    fetch_candidates_fn=None,
    log_warning=None,
    telemetry_store=None,
    turn_artifact_dir_fn=None,
    max_curation_candidates: int | None = None,
    curation_k: int | None = None,
):
    if score_fn is None:
        score_fn = lambda _s: (100, 200)  # noqa: E731
    if rust_evaluate_fn is None:
        rust_evaluate_fn = lambda _s: (500, 7.0)  # noqa: E731
    if curation_fn is None:
        async def curation_fn(_conv):  # picks every id verbatim
            from src.autotune2.sim_manager import _extract_agent_text  # noqa
            raise AssertionError(
                "default curation_fn was invoked but no candidates were "
                "supplied; the test should override curation_fn when it "
                "supplies candidates"
            )
    if decision_fn is None:
        async def decision_fn(_conv):
            return _decision_reply("rust")
    if fetch_candidates_fn is None:
        fetch_candidates_fn = lambda _src: []  # noqa: E731  cold start
    budget = TimeBudget(total_seconds=total_seconds)
    store = CalibrationStore(path=tmp_path / "calibration.jsonl")
    sources_dir = tmp_path / "calibration_sources"
    warnings: list[str] = []
    kwargs = dict(
        score_fn=score_fn, rust_evaluate_fn=rust_evaluate_fn,
        time_budget=budget, calibration_store=store,
        sources_dir=sources_dir, kernel="k", preset="p",
        hw_config_hash="0123456789abcdef" * 1, compute_bw=100_000,
        run_id="test_run",
        curation_agent_fn=curation_fn, decision_agent_fn=decision_fn,
        fetch_candidates_fn=fetch_candidates_fn,
        log_warning=(log_warning if log_warning is not None
                     else warnings.append),
        telemetry_store=telemetry_store,
        turn_artifact_dir_fn=turn_artifact_dir_fn,
    )
    if max_curation_candidates is not None:
        kwargs["max_curation_candidates"] = max_curation_candidates
    if curation_k is not None:
        kwargs["curation_k"] = curation_k
    mgr = AgentManager(**kwargs)
    return mgr, budget, store, sources_dir, warnings


def test_agent_manager_runs_rust_on_decision_rust_cold_start(tmp_path):
    """Cold start (no candidates) + decision='rust' → rust call, calibration
    record appended, cycles from rust."""
    rust_calls = []
    def rust(src):
        rust_calls.append(src)
        return (777, 3.5)

    mgr, _b, store, _, warnings = _make_agent_manager(
        tmp_path, rust_evaluate_fn=rust, total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def s(): pass"))
    assert result.cycle_source == "rust"
    assert result.cycles == 777
    assert result.rust_dur_ms == pytest.approx(3.5)
    assert len(rust_calls) == 1
    assert len(list(store.iter_records())) == 1
    assert warnings == [], f"unexpected warnings: {warnings!r}"


def test_agent_manager_falls_back_when_decision_analytical(tmp_path):
    """Decision='analytical' → no rust call, no calibration record."""
    rust_calls = []
    def rust(src):
        rust_calls.append(src)
        return (777, 3.5)

    async def decision_fn(_conv):
        return _decision_reply("analytical", "regression looks unlikely")

    mgr, _b, store, _, warnings = _make_agent_manager(
        tmp_path, rust_evaluate_fn=rust,
        decision_fn=decision_fn, total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def s(): pass"))
    assert result.cycle_source == "analytical"
    assert result.cycles == 100
    assert rust_calls == []
    assert list(store.iter_records()) == []
    assert warnings == []


def test_agent_manager_budget_exhausted_skips_agents_entirely(tmp_path):
    """When the per-pass budget is at zero we don't even call the agents —
    no point asking whether to spend budget we don't have."""
    decision_calls = []
    async def decision_fn(_conv):
        decision_calls.append(_conv)
        return _decision_reply("rust")
    mgr, _b, store, _, _ = _make_agent_manager(
        tmp_path, decision_fn=decision_fn, total_seconds=0.0,
    )
    result = _run(mgr.score(_ctx(), "def s(): pass"))
    assert result.cycle_source == "analytical"
    assert decision_calls == [], (
        "decision agent was invoked despite an exhausted budget — that "
        "burns LLM tokens for a decision the manager can't execute on"
    )


def test_agent_manager_uses_curation_when_candidates_present(tmp_path):
    """With non-empty candidates, curation is called and its picks flow
    through to the decision agent's evidence block."""
    from src.autotune2.prompts import CurationCandidate

    candidates = [
        CurationCandidate(
            record_id=f"rec{i:02x}",
            composed_source=f"# fake source {i}\n",
            analytical_cycles=100 + i,
            rust_cycles=200 + i,
            kernel="k", preset="p",
        )
        for i in range(6)
    ]

    curation_calls = []
    async def curation_fn(conv):
        curation_calls.append(conv)
        # Pick the first DEFAULT_CURATION_K records by id.
        ids = [c.record_id for c in candidates[: AgentManager.DEFAULT_CURATION_K]]
        return _curation_reply(ids)

    decision_payloads = []
    async def decision_fn(conv):
        decision_payloads.append(conv[0]["content"])
        return _decision_reply("rust", "evidence supports rust")

    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path,
        fetch_candidates_fn=lambda _src: candidates,
        curation_fn=curation_fn, decision_fn=decision_fn,
        total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def target(): pass"))
    assert result.cycle_source == "rust"
    assert len(curation_calls) == 1
    assert len(decision_payloads) == 1
    # All curated record ids appear in the decision prompt's evidence block.
    for rid in [c.record_id for c in candidates[: AgentManager.DEFAULT_CURATION_K]]:
        assert rid in decision_payloads[0]
    assert warnings == []


def test_agent_manager_falls_back_to_analytical_on_curation_parse_failure(
    tmp_path,
):
    """When the curation agent returns malformed JSON, the parser asserts;
    the AgentManager catches at the call-site boundary, logs a warning,
    and degrades the variant to analytical."""
    from src.autotune2.prompts import CurationCandidate

    candidates = [
        CurationCandidate(
            record_id="rec1", composed_source="# c1\n",
            analytical_cycles=10, rust_cycles=20,
            kernel="k", preset="p",
        ),
    ]

    async def bad_curation(_conv):
        return "no fenced block at all, the agent is confused"

    rust_calls = []
    mgr, _b, store, _, warnings = _make_agent_manager(
        tmp_path,
        fetch_candidates_fn=lambda _src: candidates,
        curation_fn=bad_curation,
        rust_evaluate_fn=lambda _s: rust_calls.append(_s) or (1, 1.0),
        total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def t(): pass"))
    assert result.cycle_source == "analytical"
    assert result.cycles == 100
    assert rust_calls == [], "rust must not run when curation failed"
    assert list(store.iter_records()) == []
    assert len(warnings) == 1
    assert "agent decision path failed" in warnings[0]


def test_agent_manager_falls_back_on_decision_parse_failure(tmp_path):
    """A malformed decision reply (valid JSON, invalid `decision` value)
    degrades to analytical, not crash."""
    async def bad_decision(_conv):
        return '```json\n{"decision": "magic", "reason": "bug"}\n```'

    mgr, _b, store, _, warnings = _make_agent_manager(
        tmp_path, decision_fn=bad_decision, total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def t(): pass"))
    assert result.cycle_source == "analytical"
    assert list(store.iter_records()) == []
    assert len(warnings) == 1


def test_agent_manager_falls_back_on_decision_rpc_failure(tmp_path):
    """Network / RPC errors (the agent_fn raising) degrade to analytical,
    matching the JSON-failure path."""
    async def boom(_conv):
        raise ConnectionError("simulated transport failure")

    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path, decision_fn=boom, total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def t(): pass"))
    assert result.cycle_source == "analytical"
    assert len(warnings) == 1
    assert "ConnectionError" in warnings[0]


def test_agent_manager_propagates_analytical_failure_as_feedback(tmp_path):
    """Analytical-scorer crash short-circuits before any agent call —
    same behaviour as RustAll / DeterministicSplit."""
    decision_calls = []
    async def decision_fn(_conv):
        decision_calls.append(_conv)
        return _decision_reply("rust")
    def boom(_src):
        raise RuntimeError("scorer exploded")
    mgr, _b, _s, _, _ = _make_agent_manager(
        tmp_path, score_fn=boom, decision_fn=decision_fn,
        total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def t(): pass"))
    assert result.error_feedback is not None
    assert "Analytical scorer failed" in result.error_feedback
    assert decision_calls == [], "decision agent must not run after analytical crashed"


def test_agent_manager_converts_rust_simulator_failure_to_score_feedback(
    tmp_path,
):
    def bad_rust(_src):
        raise AssertionError("evaluate_kernel failed at stage 'simulate'")

    mgr, _b, store, _, warnings = _make_agent_manager(
        tmp_path, rust_evaluate_fn=bad_rust, total_seconds=60.0,
    )

    result = _run(mgr.score(_ctx(), "def t(): pass"))

    assert result.cycles is None
    assert result.cycle_source is None
    assert result.error_feedback is not None
    assert "Rust simulator failed" in result.error_feedback
    assert "evaluate_kernel failed" in result.error_feedback
    assert list(store.iter_records()) == []
    assert warnings == []


def test_agent_manager_start_pass_resets_shared_budget(tmp_path):
    mgr, budget, _s, _, _ = _make_agent_manager(tmp_path, total_seconds=10.0)
    _run(budget.consume(8.0))
    _run(mgr.start_pass(40.0))
    assert budget.total_seconds == pytest.approx(40.0)
    assert budget.consumed_seconds == 0.0


# ---------------------------------------------------------------------------
# AgentManager capacity knobs (PR6 — ctor args)
# ---------------------------------------------------------------------------


def test_agent_manager_rejects_invalid_capacity_knobs(tmp_path):
    """The ctor asserts ``1 <= curation_k <= max_curation_candidates`` so
    misconfigured callers fail at construction rather than at first call.
    """
    with pytest.raises(AssertionError, match="max_curation_candidates"):
        _make_agent_manager(tmp_path, max_curation_candidates=0)
    with pytest.raises(AssertionError, match="curation_k"):
        _make_agent_manager(tmp_path, max_curation_candidates=10, curation_k=0)
    with pytest.raises(AssertionError, match="curation_k"):
        _make_agent_manager(
            tmp_path, max_curation_candidates=4, curation_k=5,
        )


def test_agent_manager_respects_curation_max_candidates(tmp_path):
    """When the fetcher returns more candidates than the configured cap,
    only the first ``max_curation_candidates`` reach the curation agent.
    """
    from src.autotune2.prompts import CurationCandidate

    # 20 candidates available; cap is 3 → only 3 should make it into the
    # curation prompt and the curation agent only ranks among those.
    all_candidates = [
        CurationCandidate(
            record_id=f"rec{i:02d}",
            composed_source=f"# fake source {i}\n",
            analytical_cycles=100 + i, rust_cycles=200 + i,
            kernel="k", preset="p",
        )
        for i in range(20)
    ]

    curation_calls = []
    async def curation_fn(conv):
        curation_calls.append(conv[0]["content"])
        # Curate from the first 3 — the ones the cap should have let
        # through. ``curation_k`` defaults to min(4, 3) == 3.
        return _curation_reply([c.record_id for c in all_candidates[:3]])

    async def decision_fn(_conv):
        return _decision_reply("analytical", "no rust needed")

    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path,
        fetch_candidates_fn=lambda _src: all_candidates,
        curation_fn=curation_fn, decision_fn=decision_fn,
        total_seconds=60.0,
        max_curation_candidates=3, curation_k=3,
    )
    _run(mgr.score(_ctx(), "def s(): pass"))
    # Records 0..2 visible in the curation prompt; record 03+ not.
    assert "rec00" in curation_calls[0]
    assert "rec02" in curation_calls[0]
    assert "rec03" not in curation_calls[0]
    assert warnings == []


def test_agent_manager_respects_curation_k(tmp_path):
    """``curation_k`` controls how many record_ids the curation agent
    must return — the parser asserts cardinality.
    """
    from src.autotune2.prompts import CurationCandidate

    candidates = [
        CurationCandidate(
            record_id=f"rec{i:02d}", composed_source=f"# s {i}\n",
            analytical_cycles=10 + i, rust_cycles=20 + i,
            kernel="k", preset="p",
        )
        for i in range(8)
    ]

    async def curation_fn(_conv):
        # k=2 was configured; return exactly 2.
        return _curation_reply(["rec00", "rec01"])

    async def decision_fn(_conv):
        return _decision_reply("analytical", "ok")

    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path,
        fetch_candidates_fn=lambda _src: candidates,
        curation_fn=curation_fn, decision_fn=decision_fn,
        total_seconds=60.0,
        max_curation_candidates=8, curation_k=2,
    )
    result = _run(mgr.score(_ctx(), "def s(): pass"))
    assert result.cycle_source == "analytical"
    assert warnings == []


# ---------------------------------------------------------------------------
# AgentManager telemetry (PR6)
# ---------------------------------------------------------------------------


def _load_telemetry(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_agent_manager_writes_telemetry_on_rust_decision(tmp_path):
    """On the rust-decision happy path, one telemetry row is written with
    ``decision='rust'``, the curated record ids, and both per-call timings.
    """
    from src.autotune2.agent_telemetry import AgentDecisionStore
    from src.autotune2.prompts import CurationCandidate

    candidates = [
        CurationCandidate(
            record_id=f"rec{i:02d}", composed_source=f"# s{i}\n",
            analytical_cycles=10 + i, rust_cycles=20 + i,
            kernel="k", preset="p",
        )
        for i in range(5)
    ]

    async def curation_fn(_conv):
        return _curation_reply(["rec00", "rec01", "rec02", "rec03"])

    async def decision_fn(_conv):
        return _decision_reply("rust", "evidence supports rust")

    tel_path = tmp_path / "agent_decisions.jsonl"
    telemetry = AgentDecisionStore(path=tel_path)

    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path,
        fetch_candidates_fn=lambda _src: candidates,
        curation_fn=curation_fn, decision_fn=decision_fn,
        total_seconds=60.0,
        telemetry_store=telemetry,
    )
    result = _run(mgr.score(_ctx(), "def t(): return 1"))
    assert result.cycle_source == "rust"

    rows = _load_telemetry(tel_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "sim_decision"
    assert row["decision"] == "rust"
    assert row["reason"] == "evidence supports rust"
    assert row["curated_record_ids"] == ["rec00", "rec01", "rec02", "rec03"]
    assert row["num_candidates_available"] == 5
    assert row["curation_dur_ms"] >= 0.0
    assert row["decision_dur_ms"] >= 0.0
    assert row["picked_variant_index"] == -1  # sim_decision stage
    assert row["composed_source_hash"] == hashlib.sha256(
        b"def t(): return 1"
    ).hexdigest()
    assert warnings == []


def test_agent_manager_writes_per_turn_curation_and_decision_artifacts(
    tmp_path,
):
    """When AgentManager invokes both LLM helpers, each call is persisted
    under the caller's turn directory for post-hoc prompt inspection.
    """
    from src.autotune2.prompts import CurationCandidate
    from src.autotune2.search import AgentResponse

    candidates = [
        CurationCandidate(
            record_id=f"rec{i:02d}", composed_source=f"# s{i}\n",
            analytical_cycles=10 + i, rust_cycles=20 + i,
            kernel="k", preset="p",
        )
        for i in range(4)
    ]

    async def curation_fn(_conv):
        return AgentResponse(
            text=_curation_reply(["rec00", "rec01", "rec02", "rec03"]),
            reasoning="curation reasoning",
        )

    async def decision_fn(_conv):
        return AgentResponse(
            text=_decision_reply("analytical", "model already accurate"),
            reasoning="decision reasoning",
        )

    turn_dir = tmp_path / "root" / "attempt_0" / "turn_5"
    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path,
        fetch_candidates_fn=lambda _src: candidates,
        curation_fn=curation_fn,
        decision_fn=decision_fn,
        total_seconds=60.0,
        turn_artifact_dir_fn=lambda ctx: turn_dir,
    )

    result = _run(mgr.score(_ctx(turn_index=5), "def t(): pass"))

    assert result.cycle_source == "analytical"
    curator_dir = turn_dir / "sim_manager" / "curator"
    decision_dir = turn_dir / "sim_manager" / "sim_manager"
    assert "## Target composed source" in (
        curator_dir / "user_prompt.txt"
    ).read_text()
    assert "rec00" in (curator_dir / "response.txt").read_text()
    assert (curator_dir / "reasoning.txt").read_text() == "curation reasoning"
    assert "variant_kind=variant" in (
        decision_dir / "user_prompt.txt"
    ).read_text()
    assert "model already accurate" in (
        decision_dir / "response.txt"
    ).read_text()
    assert (decision_dir / "reasoning.txt").read_text() == "decision reasoning"
    assert warnings == []


def test_agent_manager_repairs_curation_schema_error_once(tmp_path):
    from src.autotune2.prompts import CurationCandidate

    candidates = [
        CurationCandidate(
            record_id=f"rec{i:02d}", composed_source=f"# s{i}\n",
            analytical_cycles=10 + i, rust_cycles=20 + i,
            kernel="k", preset="p",
        )
        for i in range(4)
    ]
    curation_convos = []

    async def curation_fn(conv):
        curation_convos.append([dict(m) for m in conv])
        if len(curation_convos) == 1:
            return _curation_reply(["missing", "rec01", "rec02", "rec03"])
        assert "failed schema validation" in conv[-1]["content"]
        assert "missing" in conv[-1]["content"]
        return _curation_reply(["rec00", "rec01", "rec02", "rec03"])

    async def decision_fn(_conv):
        return _decision_reply("analytical", "repaired curation")

    turn_dir = tmp_path / "root" / "attempt_0" / "turn_0"
    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path,
        fetch_candidates_fn=lambda _src: candidates,
        curation_fn=curation_fn,
        decision_fn=decision_fn,
        total_seconds=60.0,
        turn_artifact_dir_fn=lambda ctx: turn_dir,
    )

    result = _run(mgr.score(_ctx(), "def t(): pass"))

    assert result.cycle_source == "analytical"
    assert len(curation_convos) == 2
    assert (turn_dir / "sim_manager" / "curator_repair_1"
            / "response.txt").exists()
    assert warnings == []


def test_agent_manager_repairs_decision_schema_error_once(tmp_path):
    decision_convos = []

    async def decision_fn(conv):
        decision_convos.append([dict(m) for m in conv])
        if len(decision_convos) == 1:
            return '{"decision": "maybe", "reason": "invalid"}'
        assert "failed schema validation" in conv[-1]["content"]
        return _decision_reply("rust", "repaired decision")

    turn_dir = tmp_path / "root" / "attempt_0" / "turn_0"
    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path,
        decision_fn=decision_fn,
        total_seconds=60.0,
        turn_artifact_dir_fn=lambda ctx: turn_dir,
    )

    result = _run(mgr.score(_ctx(), "def t(): pass"))

    assert result.cycle_source == "rust"
    assert len(decision_convos) == 2
    assert (turn_dir / "sim_manager" / "sim_manager_repair_1"
            / "response.txt").exists()
    assert warnings == []


def test_agent_manager_writes_decision_artifact_on_cold_start(tmp_path):
    """Cold-start skips curation, but the sim-decision prompt/response
    still lands under the turn's sim_manager directory.
    """
    async def decision_fn(_conv):
        return _decision_reply("analytical", "cold start skip")

    turn_dir = tmp_path / "root" / "attempt_0" / "turn_0"
    mgr, _b, _s, _, _ = _make_agent_manager(
        tmp_path,
        decision_fn=decision_fn,
        total_seconds=60.0,
        turn_artifact_dir_fn=lambda ctx: turn_dir,
    )

    result = _run(mgr.score(_ctx(), "def t(): pass"))

    assert result.cycle_source == "analytical"
    assert not (turn_dir / "sim_manager" / "curator").exists()
    decision_dir = turn_dir / "sim_manager" / "sim_manager"
    assert (decision_dir / "user_prompt.txt").exists()
    assert "cold start skip" in (decision_dir / "response.txt").read_text()


def test_agent_manager_writes_failure_artifact_when_curation_parse_fails(
    tmp_path,
):
    """If curation returns an invalid id, the sim-decision agent is skipped;
    the turn directory should say that explicitly instead of containing only
    a curator subdirectory.
    """
    from src.autotune2.prompts import CurationCandidate

    candidates = [
        CurationCandidate(
            record_id="rec00", composed_source="# s0\n",
            analytical_cycles=10, rust_cycles=20,
            kernel="k", preset="p",
        )
    ]

    async def curation_fn(_conv):
        return _curation_reply(["missing"])

    decision_calls = []

    async def decision_fn(_conv):
        decision_calls.append(_conv)
        return _decision_reply("rust")

    turn_dir = tmp_path / "root" / "attempt_0" / "turn_0"
    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path,
        fetch_candidates_fn=lambda _src: candidates,
        curation_fn=curation_fn,
        decision_fn=decision_fn,
        total_seconds=60.0,
        turn_artifact_dir_fn=lambda ctx: turn_dir,
    )

    result = _run(mgr.score(_ctx(), "def t(): pass"))

    assert result.cycle_source == "analytical"
    assert decision_calls == []
    assert warnings
    base_dir = turn_dir / "sim_manager"
    error = json.loads((base_dir / "error.json").read_text())
    assert error["error_type"] == "AssertionError"
    assert "parse_curation_response" in error["message"]
    decision_status = (
        base_dir / "sim_manager" / "status.txt"
    ).read_text()
    assert "SKIPPED" in decision_status
    assert "sim-decision agent did not complete" in decision_status


def test_agent_manager_writes_telemetry_on_analytical_decision(tmp_path):
    """Analytical-decision rows record ``decision='analytical'`` and the
    agent's reason but do NOT write a calibration record.
    """
    from src.autotune2.agent_telemetry import AgentDecisionStore

    async def decision_fn(_conv):
        return _decision_reply("analytical", "model already accurate")

    tel_path = tmp_path / "agent_decisions.jsonl"
    telemetry = AgentDecisionStore(path=tel_path)

    mgr, _b, store, _, _ = _make_agent_manager(
        tmp_path, decision_fn=decision_fn,
        total_seconds=60.0, telemetry_store=telemetry,
    )
    result = _run(mgr.score(_ctx(), "def t(): pass"))
    assert result.cycle_source == "analytical"

    rows = _load_telemetry(tel_path)
    assert len(rows) == 1
    assert rows[0]["decision"] == "analytical"
    assert rows[0]["reason"] == "model already accurate"
    assert rows[0]["curated_record_ids"] == []
    # Cold-start: no curation call happened, so dur_ms is the -1 sentinel.
    assert rows[0]["curation_dur_ms"] == -1.0
    assert rows[0]["decision_dur_ms"] >= 0.0
    # No rust call ⇒ no calibration record.
    assert list(store.iter_records()) == []


def test_agent_manager_writes_telemetry_on_budget_exhausted(tmp_path):
    """When the budget is exhausted, AgentManager skips both agent calls
    but still emits a telemetry row tagged
    ``decision='analytical_budget_exhausted'`` so the audit log is
    complete.
    """
    from src.autotune2.agent_telemetry import AgentDecisionStore

    decision_calls = []
    async def decision_fn(_conv):
        decision_calls.append(_conv)
        return _decision_reply("rust")

    tel_path = tmp_path / "agent_decisions.jsonl"
    telemetry = AgentDecisionStore(path=tel_path)

    mgr, budget, _s, _, _ = _make_agent_manager(
        tmp_path, decision_fn=decision_fn,
        total_seconds=5.0, telemetry_store=telemetry,
    )
    _run(budget.consume(6.0))  # over-consume to exhaust
    assert budget.remaining_seconds <= 0

    result = _run(mgr.score(_ctx(), "def t(): pass"))
    assert result.cycle_source == "analytical"
    assert decision_calls == [], (
        "decision agent must not run when budget is exhausted"
    )
    rows = _load_telemetry(tel_path)
    assert len(rows) == 1
    assert rows[0]["decision"] == "analytical_budget_exhausted"
    assert rows[0]["curation_dur_ms"] == -1.0
    assert rows[0]["decision_dur_ms"] == -1.0


def test_agent_manager_writes_telemetry_on_fallback(tmp_path):
    """A decision-agent RPC failure triggers the outer fallback; the
    telemetry row records ``decision='fallback'`` with the exception
    summary as the reason.
    """
    from src.autotune2.agent_telemetry import AgentDecisionStore

    async def decision_fn(_conv):
        raise RuntimeError("RPC down")

    tel_path = tmp_path / "agent_decisions.jsonl"
    telemetry = AgentDecisionStore(path=tel_path)

    mgr, _b, _s, _, warnings = _make_agent_manager(
        tmp_path, decision_fn=decision_fn,
        total_seconds=60.0, telemetry_store=telemetry,
    )
    result = _run(mgr.score(_ctx(), "def t(): pass"))
    assert result.cycle_source == "analytical"
    rows = _load_telemetry(tel_path)
    assert len(rows) == 1
    assert rows[0]["decision"] == "fallback"
    assert "RuntimeError" in rows[0]["reason"]
    assert "RPC down" in rows[0]["reason"]
    assert warnings, "fallback path must log a warning"


def test_agent_manager_no_telemetry_when_store_is_none(tmp_path):
    """Telemetry is opt-in; passing ``telemetry_store=None`` keeps the
    AgentManager behaviour identical to PR5.
    """
    mgr, _b, _s, _, _ = _make_agent_manager(
        tmp_path, telemetry_store=None, total_seconds=60.0,
    )
    result = _run(mgr.score(_ctx(), "def t(): pass"))
    assert result.cycle_source == "rust"  # default decision_fn returns rust
    # No telemetry file should exist alongside the calibration store.
    assert not (tmp_path / "agent_decisions.jsonl").exists()
