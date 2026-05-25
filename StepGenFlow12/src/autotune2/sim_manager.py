"""Simulation manager for autotune2.

The simulation manager intercepts every variant the search loop wants to
score and decides which estimator records the cycle count: the cheap
analytical timing model, or the more accurate (but slower) rust cycle-
approximate simulator. ``on_chip`` always comes from the analytical
model — rust does not surface on-chip bytes today — so only the cycles
axis is swappable.

This module ships the interception seam (``SimulationManager`` protocol)
plus the trivial ``AnalyticalOnly`` implementation that matches the
historical in-loop behavior. ``RustAll`` / ``DeterministicSplit`` /
``AgentManager`` live in their own modules (added in later PRs); all of
them satisfy the same protocol so ``search.py`` does not care which one
is wired up.

Design notes
------------
* ``score`` is async so agent-backed managers can ``await`` an LLM call
  without forcing the search loop to thread sync/async layers. Sync
  managers like ``AnalyticalOnly`` simply do not ``await``.
* ``SimulationResult.error_feedback`` carries the analytical-scorer-
  crashed-mid-flight message that ``_safe_score`` used to return — the
  search loop checks ``error_feedback is not None`` exactly where it
  used to check ``score_err is not None``.
* ``breakdown`` is part of the result rather than a side-channel: the
  per-node on-chip memory report is cheap to compute when the
  analytical scorer's per-source cache is already warm, so the manager
  always returns it alongside cycles instead of forcing the caller to
  re-fetch it via a ``.breakdown`` attribute on the score function.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import json
import hashlib
import time
import traceback as _tb
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Protocol


# (composed_source) -> (cycles, on_chip_bytes). Same shape as the legacy
# ``ScoreFn`` in ``compose.py``; re-declared here so impl modules don't
# need to import ``compose``.
ScoreFn = Callable[[str], tuple[int, int]]

# (composed_source, optional metadata kwargs) -> (rust_cycles, rust_dur_ms).
# Same shape as ``runtime.RustEvaluateFn``; re-declared here so the rust
# manager implementation doesn't have to import ``runtime`` (which would
# pull in the StepDB / step_tl path setup at import time).
RustEvaluateFn = Callable[..., tuple[int, float]]


class RustOutputMismatch(Exception):
    """Rust simulator finished but its produced output tensor diverged
    from the PyTorch reference. Raised by the closure built by
    ``runtime.build_rust_evaluate_fn`` when ``evaluate_kernel`` returns
    ``stage="correctness", success=False``. Lives here (not in
    ``runtime``) so the search-loop managers that catch it can stay
    independent of ``runtime``'s StepDB / step_tl path setup.
    """

    def __init__(self, *, cycles: float, max_diff: float, message: str) -> None:
        super().__init__(message)
        self.cycles = cycles
        self.max_diff = max_diff
        self.message = message


@dataclass(frozen=True)
class SimContext:
    """Identifies the call site of a ``SimulationManager.score`` call.

    PR1 managers ignore every field; later managers (the LLM-backed
    decision agent in particular) use them to render per-call context.
    Adding a field here is non-breaking as long as a default is given.
    """

    node_path: str
    variant_kind: str  # "baseline" | "variant"
    is_root: bool
    attempt_index: int = -1  # -1 for baseline scoring (no attempt context)
    turn_index: int = -1  # -1 for baseline scoring
    turn_artifact_dir: Path | None = None


@dataclass
class SimulationResult:
    """One simulation manager decision, executed.

    Successful path: ``cycles`` and ``on_chip`` are set, ``cycle_source``
    is ``"analytical"`` or ``"rust"``, ``error_feedback`` is None. The
    search loop admits an entry with ``(cycles, on_chip)`` tagged by
    ``cycle_source``.

    Failure path: ``cycles`` and ``on_chip`` are None, ``cycle_source``
    is None, ``error_feedback`` is a non-empty LLM-actionable string
    describing what blew up. The search loop converts this into next-
    turn feedback and skips the variant.
    """

    cycles: int | None
    on_chip: int | None
    cycle_source: str | None
    rust_dur_ms: float | None = None
    breakdown: str = ""
    error_feedback: str | None = None


class SimulationManager(Protocol):
    """Per-variant scoring policy.

    Implementations decide which simulator runs and what cycle source
    is recorded. The search loop hands one composed-source string to
    ``score`` per verified variant and trusts whatever comes back.
    """

    async def start_pass(
        self, time_budget_seconds: float | None,
    ) -> None: ...

    async def score(
        self, ctx: SimContext, composed_source: str,
    ) -> SimulationResult: ...


def _turn_relative_label(ctx: SimContext) -> str:
    if ctx.turn_artifact_dir is None:
        if ctx.variant_kind == "baseline":
            return "pass1_baseline"
        return (
            f"{ctx.variant_kind}/attempt_{ctx.attempt_index}/"
            f"turn_{ctx.turn_index}"
        )
    parts = tuple(Path(ctx.turn_artifact_dir).parts)
    node_parts = tuple(p for p in ctx.node_path.split("/") if p)
    for i in range(len(parts) - len(node_parts), -1, -1):
        if parts[i:i + len(node_parts)] == node_parts:
            tail = parts[i + len(node_parts):]
            if tail:
                return "/".join(tail)
            break
    return "/".join(parts[-3:])


def _invoke_rust_evaluate(
    rust_evaluate_fn: RustEvaluateFn,
    composed_source: str,
    *,
    ctx: SimContext,
) -> tuple[int, float]:
    try:
        sig = inspect.signature(rust_evaluate_fn)
    except (TypeError, ValueError):
        return rust_evaluate_fn(composed_source)
    params = sig.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_kwargs or "node_path" in params or "run_label" in params:
        return rust_evaluate_fn(
            composed_source,
            node_path=ctx.node_path,
            run_label=_turn_relative_label(ctx),
        )
    return rust_evaluate_fn(composed_source)


# ---------------------------------------------------------------------------
# TimeBudget — wall-clock accounting for one manager scope
# ---------------------------------------------------------------------------


class TimeBudget:
    """Wall-clock budget for one simulation-manager scope.

    Rust simulator runs take seconds to hours and may be dispatched
    concurrently from an autotune node's attempt fan-out, so an
    asyncio-aware budget object lets every attempt for that manager
    consult one ``remaining_seconds`` and consume against one accounting
    record. Node-local per-pass semantics are implemented via
    ``reset(seconds)``: the driver calls it indirectly through
    ``SimulationManager.start_pass`` when a node begins searching.

    ``total_seconds=None`` means "unlimited"; ``remaining_seconds``
    returns ``inf`` in that case so callers using the canonical
    ``remaining <= 0`` cutoff just keep going.

    Concurrency model
    -----------------
    ``consume`` is awaited under an ``asyncio.Lock`` so the
    ``_consumed`` accumulator can't lose a write under the search
    fan-out. ``reset`` is synchronous and should happen before attempts
    for this manager call ``consume``.
    """

    # Number of recent rust durations averaged into
    # ``recent_avg_seconds``. Small window because the rust runtime
    # spectrum is so wide (seconds → hours) that older measurements
    # rapidly become irrelevant to the current call's "how long will
    # this take?" question.
    _RECENT_WINDOW = 8

    def __init__(self, total_seconds: float | None = None) -> None:
        assert total_seconds is None or (
            isinstance(total_seconds, (int, float)) and total_seconds >= 0
        ), (
            f"TimeBudget: total_seconds must be None or a non-negative "
            f"number, got {total_seconds!r}"
        )
        self._total: float | None = (
            None if total_seconds is None else float(total_seconds)
        )
        self._consumed: float = 0.0
        self._recent: collections.deque[float] = collections.deque(
            maxlen=self._RECENT_WINDOW,
        )
        self._lock = asyncio.Lock()

    def reset(self, total_seconds: float | None) -> None:
        """Re-arm the budget for a new pass.

        Called once per pass by the simulation manager's ``start_pass``.
        Identical concurrent resets are safe (every node task calls
        with the same seconds); diverging concurrent resets would race
        and are not supported.
        """
        assert total_seconds is None or (
            isinstance(total_seconds, (int, float)) and total_seconds >= 0
        ), (
            f"TimeBudget.reset: total_seconds must be None or a non-negative "
            f"number, got {total_seconds!r}"
        )
        self._total = (
            None if total_seconds is None else float(total_seconds)
        )
        self._consumed = 0.0
        self._recent.clear()

    async def consume(self, seconds: float) -> None:
        """Charge ``seconds`` against the budget (under an asyncio lock)."""
        assert isinstance(seconds, (int, float)) and seconds >= 0, (
            f"TimeBudget.consume: seconds must be a non-negative number, "
            f"got {seconds!r}"
        )
        async with self._lock:
            self._consumed += float(seconds)
            self._recent.append(float(seconds))

    @property
    def remaining_seconds(self) -> float:
        """Seconds left in this pass; ``inf`` when the budget is unlimited.

        ``inf`` for unlimited means callers using the standard
        ``remaining_seconds <= 0`` cutoff work without a special case
        for None — they simply never trip the cutoff.
        """
        if self._total is None:
            return float("inf")
        return self._total - self._consumed

    @property
    def total_seconds(self) -> float | None:
        return self._total

    @property
    def consumed_seconds(self) -> float:
        return self._consumed

    @property
    def recent_avg_seconds(self) -> float:
        """Mean of the last ``_RECENT_WINDOW`` consumed durations.

        Returns 0.0 when no calls have been recorded yet, so a freshly
        reset budget reports zero rather than NaN. Decision agents use
        this to estimate how many more rust calls fit in the
        remaining budget.
        """
        if not self._recent:
            return 0.0
        return sum(self._recent) / len(self._recent)


# ---------------------------------------------------------------------------
# AnalyticalOnly — wraps the existing analytical scorer; no decisions made
# ---------------------------------------------------------------------------


class AnalyticalOnly:
    """Always invoke the analytical scorer; never call rust.

    Behavior-equivalent to the legacy ``score_fn`` injection point —
    every variant gets the analytical estimate, exceptions inside
    ``analyze_timing`` become LLM feedback (mirroring the old
    ``_safe_score`` helper), and ``cycle_source`` is always
    ``"analytical"``.
    """

    def __init__(self, score_fn: ScoreFn) -> None:
        self._score_fn = score_fn

    async def start_pass(
        self, time_budget_seconds: float | None,
    ) -> None:
        # AnalyticalOnly never consults a budget — analytical scoring
        # is cheap, runs synchronously, and never benefits from being
        # gated. Method exists so the manager satisfies the Protocol.
        return None

    async def score(
        self, ctx: SimContext, composed_source: str,
    ) -> SimulationResult:
        # Mirrors the legacy ``_safe_score`` body — the timing-model
        # executor can raise on shape regimes outside its uniform path
        # (non-equal-bucket flat_reassemble / flat_partition, etc.).
        # Catching here converts the crash into LLM feedback rather
        # than aborting the autotune2 run.
        try:
            cycles, on_chip = self._score_fn(composed_source)
        except Exception:
            return SimulationResult(
                cycles=None,
                on_chip=None,
                cycle_source=None,
                error_feedback=_analytical_failure_feedback(_tb.format_exc()),
            )
        breakdown_fn = getattr(self._score_fn, "breakdown", None)
        breakdown = breakdown_fn(composed_source) if breakdown_fn else ""
        return SimulationResult(
            cycles=cycles,
            on_chip=on_chip,
            cycle_source="analytical",
            breakdown=breakdown,
        )


def _rust_correctness_failure_feedback(*, cycles: float, max_diff: float, message: str) -> str:
    """LLM-actionable message when the rust functional sim produces a
    tensor that diverges from the PyTorch gold reference.

    A wrong-output graph almost always also reports a meaningless cycle
    count (typical failure mode: a structural bug drains the pipeline
    early, so cycles look "winning" but the output is truncated). The
    feedback names the cycle and max_diff so the agent can spot the
    pattern, and explicitly tells it the cycle is unusable.
    """
    return (
        "## Rust functional output diverged from gold\n\n"
        "The DSL translated, the STeP graph built, and the rust simulator "
        f"ran to completion (cycles={int(cycles) if cycles == cycles else 'NaN'}, "
        f"max_diff={max_diff:g}) — but the produced output tensor did not "
        "match the PyTorch reference. Cycle count from this run is **not** "
        "trustworthy: a structurally broken graph often drains in a tiny "
        "number of cycles (e.g. an undersized accumulator that silently "
        "truncates a matmul reduction), which makes the variant look like a "
        "winner while producing wrong data. Error follows:\n\n"
        "```\n" + message + "\n```\n\n"
        "Re-examine the data path — particularly accumulator tile shapes, "
        "stream-rank alignment between zipped inputs, and any operator "
        "whose output tile shape differs from its inputs."
    )


def _analytical_failure_feedback(err: str) -> str:
    """LLM-actionable message produced when the analytical scorer raises.

    Verbatim copy of the string the legacy ``_safe_score`` returned so
    the resulting feedback is byte-for-byte identical to the previous
    behavior.
    """
    return (
        "## Analytical scorer failed on this variant\n\n"
        "The DSL translated and the STeP graph built, but the timing "
        "model's analyzer raised while propagating concrete values "
        "through the graph. This is usually a shape regime the "
        "functional executor can't handle uniformly (e.g. an op like "
        "`flat_reassemble` / `flat_partition` whose per-token or "
        "per-expert buckets must be equal-sized for the executor's "
        "internal stack-and-reshape path). Error follows:\n\n"
        "```\n" + err + "```\n\n"
        "Consider an alternative implementation that avoids the "
        "failing op pattern, or adjust your contracts so the "
        "downstream stream shapes are uniform across the dynamic "
        "axes the failing op spans."
    )


# ---------------------------------------------------------------------------
# RustAll — analytical + rust per variant, gated by a per-pass time budget
# ---------------------------------------------------------------------------


class RustAll:
    """Run analytical AND rust per variant, recording the rust cycles.

    For every successful analytical score, ``RustAll`` also invokes the
    injected rust simulator. The recorded cycle count is the rust
    number; ``on_chip`` is still the analytical estimate (rust doesn't
    surface on-chip bytes). When the per-pass ``TimeBudget`` is
    exhausted, RustAll degrades to ``AnalyticalOnly`` for the remainder
    of the pass — the variant is admitted with the analytical cycle
    count and ``cycle_source="analytical"``. Per design decision #4 in
    HANDOFF.md the cutoff is "hard" only in the sense that no new rust
    call starts after ``remaining_seconds <= 0``; calls already in
    flight run to completion.

    Each successful rust call appends a ``CalibrationRecord`` to the
    injected store so the cross-kernel calibration library accumulates
    paired (analytical, rust) measurements for the LLM curation /
    decision agents in later PRs.

    Composed-source persistence
    ---------------------------
    ``CalibrationRecord.composed_source_path`` is a path on disk, not
    the source text. Recording inline would blow past the JSONL
    atomic-append limit (see ``calibration.py``); RustAll writes the
    composed source into ``sources_dir`` under a content-addressed
    filename (``sha256(source).py``), then records that path in the
    record. Same source from two different runs → same file (idempotent
    write) → calibration records de-dup naturally on the source-text
    axis.
    """

    def __init__(
        self,
        *,
        score_fn: ScoreFn,
        rust_evaluate_fn: RustEvaluateFn,
        time_budget: "TimeBudget",
        calibration_store,  # CalibrationStore (avoid cyclic import)
        sources_dir: Path,
        kernel: str,
        preset: str,
        hw_config_hash: str,
        compute_bw: int,
        run_id: str,
    ) -> None:
        self._score_fn = score_fn
        self._rust_evaluate_fn = rust_evaluate_fn
        self._time_budget = time_budget
        self._store = calibration_store
        self._sources_dir = Path(sources_dir)
        self._kernel = kernel
        self._preset = preset
        self._hw_hash = hw_config_hash
        self._compute_bw = int(compute_bw)
        self._run_id = run_id

    async def start_pass(
        self, time_budget_seconds: float | None,
    ) -> None:
        """Re-arm this manager's node-local time budget."""
        self._time_budget.reset(time_budget_seconds)

    async def score(
        self, ctx: SimContext, composed_source: str,
    ) -> SimulationResult:
        # Analytical first — needed for ``on_chip``, the breakdown, and
        # as the fallback cycle estimate when the budget is exhausted.
        # Mirrors AnalyticalOnly's exception-to-feedback path verbatim:
        # if analytical scoring crashes, surface the same byte-for-byte
        # feedback string and skip the rust call entirely (a variant
        # the analytical model can't even shape-check is unlikely to
        # produce useful rust evidence).
        try:
            cycles, on_chip = self._score_fn(composed_source)
        except Exception:
            return SimulationResult(
                cycles=None,
                on_chip=None,
                cycle_source=None,
                error_feedback=_analytical_failure_feedback(_tb.format_exc()),
            )
        breakdown_fn = getattr(self._score_fn, "breakdown", None)
        breakdown = breakdown_fn(composed_source) if breakdown_fn else ""

        # Budget-exhausted fallback. The check is racy across concurrent
        # tasks (one task may pass the check, another may bring
        # ``remaining`` below zero before the first task's rust call
        # actually starts), and that's fine: the handoff explicitly
        # allows "Rust calls that started before the cutoff run to
        # completion (no mid-call kill)." A single extra in-flight rust
        # call past the cutoff is not worth the complexity of an
        # atomic "reserve + execute + commit" sequence.
        if self._time_budget.remaining_seconds <= 0:
            return SimulationResult(
                cycles=cycles,
                on_chip=on_chip,
                cycle_source="analytical",
                breakdown=breakdown,
            )

        # Rust call — blocking subprocess inside ``rust_evaluate_fn``;
        # offload to a thread so the asyncio event loop can keep
        # interleaving other node tasks (which is the whole point of
        # the search fan-out).
        t0 = time.perf_counter()
        try:
            rust_cycles, rust_dur_ms = await asyncio.to_thread(
                _invoke_rust_evaluate,
                self._rust_evaluate_fn,
                composed_source,
                ctx=ctx,
            )
        except RustOutputMismatch as exc:
            await self._time_budget.consume(time.perf_counter() - t0)
            return SimulationResult(
                cycles=None, on_chip=None, cycle_source=None,
                error_feedback=_rust_correctness_failure_feedback(
                    cycles=exc.cycles, max_diff=exc.max_diff, message=exc.message,
                ),
            )
        elapsed = time.perf_counter() - t0
        await self._time_budget.consume(elapsed)

        # Persist the (analytical, rust) pair for cross-kernel
        # calibration. Source is content-addressed so duplicates de-dup
        # on disk; the record carries a path reference, never inline
        # text, to stay under the calibration JSONL atomic-write limit.
        source_path = _write_calibration_source(
            self._sources_dir, composed_source,
        )
        _append_calibration(
            store=self._store, ctx=ctx, source_path=source_path,
            kernel=self._kernel, preset=self._preset,
            hw_hash=self._hw_hash, compute_bw=self._compute_bw,
            run_id=self._run_id,
            analytical_cycles=cycles, analytical_on_chip=on_chip,
            rust_cycles=int(rust_cycles), rust_dur_ms=float(rust_dur_ms),
        )

        return SimulationResult(
            cycles=int(rust_cycles),
            on_chip=on_chip,
            cycle_source="rust",
            rust_dur_ms=float(rust_dur_ms),
            breakdown=breakdown,
        )


# ---------------------------------------------------------------------------
# DeterministicSplit — rule-driven analytical/rust mix per variant (PR4)
# ---------------------------------------------------------------------------


class DeterministicSplit:
    """Rule-driven analytical-vs-rust split per variant.

    Today's only rule is ``"rust_baselines"``:
      - Every variant tagged ``variant_kind == "baseline"`` is rust-
        evaluated (the baseline anchors the Pareto for the rest of the
        pass — its cycles are load-bearing, so we want ground truth).
      - Every other variant takes the analytical-only path.

    Identical wiring to ``RustAll`` otherwise: same constructor args, same
    node-local ``TimeBudget``, same calibration-store write-through on rust
    calls, same budget-exhausted fallback to analytical. Adding new rules
    is a switch in ``_should_use_rust``; the rule string is checked at
    construction so unknown rules fail loud.
    """

    _SUPPORTED_RULES = frozenset({"rust_baselines"})

    def __init__(
        self,
        *,
        score_fn: ScoreFn,
        rust_evaluate_fn: RustEvaluateFn,
        time_budget: "TimeBudget",
        calibration_store,
        sources_dir: Path,
        kernel: str,
        preset: str,
        hw_config_hash: str,
        compute_bw: int,
        run_id: str,
        rule: str = "rust_baselines",
    ) -> None:
        assert rule in self._SUPPORTED_RULES, (
            f"DeterministicSplit: rule {rule!r} is not supported; expected "
            f"one of {sorted(self._SUPPORTED_RULES)!r}"
        )
        self._score_fn = score_fn
        self._rust_evaluate_fn = rust_evaluate_fn
        self._time_budget = time_budget
        self._store = calibration_store
        self._sources_dir = Path(sources_dir)
        self._kernel = kernel
        self._preset = preset
        self._hw_hash = hw_config_hash
        self._compute_bw = int(compute_bw)
        self._run_id = run_id
        self._rule = rule

    async def start_pass(
        self, time_budget_seconds: float | None,
    ) -> None:
        """Re-arm this manager's node-local time budget."""
        self._time_budget.reset(time_budget_seconds)

    def _should_use_rust(self, ctx: SimContext) -> bool:
        # Single rule today; extend the dispatch when more land.
        if self._rule == "rust_baselines":
            return ctx.variant_kind == "baseline"
        raise AssertionError(
            f"DeterministicSplit._should_use_rust: unreachable — rule "
            f"{self._rule!r} accepted by ctor but unhandled here"
        )

    async def score(
        self, ctx: SimContext, composed_source: str,
    ) -> SimulationResult:
        # Analytical first — needed for on_chip + breakdown + fallback.
        try:
            cycles, on_chip = self._score_fn(composed_source)
        except Exception:
            return SimulationResult(
                cycles=None,
                on_chip=None,
                cycle_source=None,
                error_feedback=_analytical_failure_feedback(_tb.format_exc()),
            )
        breakdown_fn = getattr(self._score_fn, "breakdown", None)
        breakdown = breakdown_fn(composed_source) if breakdown_fn else ""

        if not self._should_use_rust(ctx):
            return SimulationResult(
                cycles=cycles, on_chip=on_chip,
                cycle_source="analytical", breakdown=breakdown,
            )

        if self._time_budget.remaining_seconds <= 0:
            return SimulationResult(
                cycles=cycles, on_chip=on_chip,
                cycle_source="analytical", breakdown=breakdown,
            )

        t0 = time.perf_counter()
        try:
            rust_cycles, rust_dur_ms = await asyncio.to_thread(
                _invoke_rust_evaluate,
                self._rust_evaluate_fn,
                composed_source,
                ctx=ctx,
            )
        except RustOutputMismatch as exc:
            await self._time_budget.consume(time.perf_counter() - t0)
            return SimulationResult(
                cycles=None, on_chip=None, cycle_source=None,
                error_feedback=_rust_correctness_failure_feedback(
                    cycles=exc.cycles, max_diff=exc.max_diff, message=exc.message,
                ),
            )
        elapsed = time.perf_counter() - t0
        await self._time_budget.consume(elapsed)

        source_path = _write_calibration_source(
            self._sources_dir, composed_source,
        )
        _append_calibration(
            store=self._store, ctx=ctx, source_path=source_path,
            kernel=self._kernel, preset=self._preset,
            hw_hash=self._hw_hash, compute_bw=self._compute_bw,
            run_id=self._run_id,
            analytical_cycles=cycles, analytical_on_chip=on_chip,
            rust_cycles=int(rust_cycles), rust_dur_ms=float(rust_dur_ms),
        )

        return SimulationResult(
            cycles=int(rust_cycles), on_chip=on_chip,
            cycle_source="rust",
            rust_dur_ms=float(rust_dur_ms), breakdown=breakdown,
        )


# ---------------------------------------------------------------------------
# Shared calibration write helpers (used by RustAll, DeterministicSplit,
# AgentManager — refactored out so the per-class methods stay focused on
# the decision logic, not the persistence boilerplate)
# ---------------------------------------------------------------------------


def _write_calibration_source(sources_dir: Path, composed_source: str) -> Path:
    """Write ``composed_source`` under a sha256-addressed filename.

    Idempotent: repeated calls with the same source hit the same path,
    so concurrent writers race benignly (every writer produces identical
    bytes).
    """
    digest = hashlib.sha256(composed_source.encode("utf-8")).hexdigest()
    sources_dir.mkdir(parents=True, exist_ok=True)
    path = sources_dir / f"{digest}.py"
    if not path.exists():
        path.write_text(composed_source, encoding="utf-8")
    return path


def _append_calibration(
    *,
    store,
    ctx: SimContext,
    source_path: Path,
    kernel: str,
    preset: str,
    hw_hash: str,
    compute_bw: int,
    run_id: str,
    analytical_cycles: int,
    analytical_on_chip: int,
    rust_cycles: int,
    rust_dur_ms: float,
) -> None:
    from src.autotune2.calibration import CalibrationRecord
    rust_int = int(rust_cycles)
    assert rust_int > 0, (
        f"_append_calibration: rust_cycles must be > 0 to compute "
        f"error_pct relative to ground truth, got {rust_int!r}"
    )
    record = CalibrationRecord(
        node_path=ctx.node_path,
        is_root=ctx.is_root,
        kernel=kernel,
        preset=preset,
        composed_source_path=str(source_path),
        analytical_cycles=int(analytical_cycles),
        analytical_on_chip=int(analytical_on_chip),
        rust_cycles=rust_int,
        rust_dur_ms=float(rust_dur_ms),
        hw_config_hash=hw_hash,
        compute_bw=int(compute_bw),
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_id=run_id,
        error_pct=100.0 * (int(analytical_cycles) - rust_int) / rust_int,
    )
    store.append(record)


# ---------------------------------------------------------------------------
# AgentManager — in-loop LLM decides rust vs analytical per variant (PR4)
# ---------------------------------------------------------------------------


# Conversational callable for the curation + decision agents. Same shape
# as ``src/autotune2/search.AgentFn`` but re-declared here so sim_manager
# stays decoupled from search-driver internals.
AgentCallFn = Callable[[list[dict]], Awaitable[object]]
# (composed_source) -> [CurationCandidate, ...] — provided by the
# AgentManager owner so the manager doesn't have to reach into the
# CalibrationStore directly (lets the run-script filter by hw_config_hash,
# cap candidate count, etc.).
FetchCandidatesFn = Callable[[str], list]
TurnArtifactDirFn = Callable[[SimContext], Path | None]


class AgentManager:
    """LLM-backed simulation manager (HANDOFF design decision #8).

    Per scored variant, ``AgentManager``:
      1. invokes the cheap **curation agent** to pick the K calibration
         records most predictive of this variant's analytical-vs-rust
         relationship,
      2. invokes the **simulation-decision agent** with those records +
         the budget state + the variant's analytical estimate to get a
         ``rust`` / ``analytical`` decision,
      3. executes that decision — either spending Rust budget on a real
         measurement (and writing a fresh calibration record), or
         returning the cheap analytical number.

    Fall-back policy (HANDOFF design decision #1 + user-confirmed PR4
    choice "fall back, log a warning"): any failure on the agent call
    path — RPC errors, JSON parse failures, missing fenced blocks, agent
    returning a non-string — is caught at the call-site boundary, logged
    via ``log_warning``, and degrades the variant to analytical-only.
    The fenced-JSON parsers themselves stay fail-loud (CLAUDE.md style);
    only this outer wrapper degrades.

    Budget semantics: when ``time_budget.remaining_seconds <= 0`` we skip
    both agent calls entirely (no point asking the LLM whether to spend
    budget we don't have) and return analytical. When the curated set is
    empty (cold-start: no prior records for this hw_config_hash) we run
    the decision agent with an empty evidence block — the agent's
    system prompt covers the cold-start case explicitly.

    Cardinality knobs (``max_curation_candidates``, ``curation_k``) and
    the optional ``telemetry_store`` are ctor args so the runner can
    tune them via CLI without monkey-patching.
    """

    # Defaults for the cardinality knobs — overridable via ctor.
    # 50 candidates: anything larger and the curation prompt fills with
    # low-signal records; anything smaller and the LLM can't see enough
    # divergence patterns to discriminate. Starting point — revisit
    # with telemetry. 4 picks: the decision agent reads all K, and a
    # small K keeps the decision prompt short for the in-loop call.
    DEFAULT_MAX_CURATION_CANDIDATES = 50
    DEFAULT_CURATION_K = 4

    def __init__(
        self,
        *,
        score_fn: ScoreFn,
        rust_evaluate_fn: RustEvaluateFn,
        time_budget: "TimeBudget",
        calibration_store,
        sources_dir: Path,
        kernel: str,
        preset: str,
        hw_config_hash: str,
        compute_bw: int,
        run_id: str,
        curation_agent_fn: AgentCallFn,
        decision_agent_fn: AgentCallFn,
        fetch_candidates_fn: FetchCandidatesFn,
        log_warning: Callable[[str], None] = print,
        max_curation_candidates: int = DEFAULT_MAX_CURATION_CANDIDATES,
        curation_k: int = DEFAULT_CURATION_K,
        telemetry_store=None,  # AgentDecisionStore | None — None disables telemetry
        turn_artifact_dir_fn: TurnArtifactDirFn | None = None,
    ) -> None:
        assert max_curation_candidates >= 1, (
            f"AgentManager: max_curation_candidates must be >= 1, "
            f"got {max_curation_candidates!r}"
        )
        assert 1 <= curation_k <= max_curation_candidates, (
            f"AgentManager: curation_k must be in [1, max_curation_candidates="
            f"{max_curation_candidates}], got {curation_k!r}"
        )
        self._score_fn = score_fn
        self._rust_evaluate_fn = rust_evaluate_fn
        self._time_budget = time_budget
        self._store = calibration_store
        self._sources_dir = Path(sources_dir)
        self._kernel = kernel
        self._preset = preset
        self._hw_hash = hw_config_hash
        self._compute_bw = int(compute_bw)
        self._run_id = run_id
        self._curation_fn = curation_agent_fn
        self._decision_fn = decision_agent_fn
        self._fetch_candidates_fn = fetch_candidates_fn
        self._log_warning = log_warning
        self._max_curation_candidates = int(max_curation_candidates)
        self._curation_k = int(curation_k)
        self._telemetry_store = telemetry_store
        self._turn_artifact_dir_fn = turn_artifact_dir_fn

    async def start_pass(
        self, time_budget_seconds: float | None,
    ) -> None:
        self._time_budget.reset(time_budget_seconds)

    async def score(
        self, ctx: SimContext, composed_source: str,
    ) -> SimulationResult:
        # Analytical first — identical shape to RustAll / DeterministicSplit.
        try:
            cycles, on_chip = self._score_fn(composed_source)
        except Exception:
            return SimulationResult(
                cycles=None, on_chip=None, cycle_source=None,
                error_feedback=_analytical_failure_feedback(_tb.format_exc()),
            )
        breakdown_fn = getattr(self._score_fn, "breakdown", None)
        breakdown = breakdown_fn(composed_source) if breakdown_fn else ""

        # Telemetry inputs that every path below shares. ``source_hash``
        # is the same digest the calibration sources_dir uses, so a
        # telemetry row joins to a calibration row (when one exists) by
        # equality on this field.
        source_hash = hashlib.sha256(composed_source.encode("utf-8")).hexdigest()

        # Budget exhausted: degrade to analytical, skip both agent calls
        # entirely. Still log the decision — "we skipped because of the
        # budget" is exactly the kind of audit signal #2 is for.
        if self._time_budget.remaining_seconds <= 0:
            self._emit_telemetry(
                ctx=ctx, source_hash=source_hash,
                decision="analytical_budget_exhausted",
                reason="budget exhausted before agent could decide",
                curated_ids=[], num_candidates=0,
                curation_dur_ms=-1.0, decision_dur_ms=-1.0,
            )
            return SimulationResult(
                cycles=cycles, on_chip=on_chip,
                cycle_source="analytical", breakdown=breakdown,
            )

        # Decide rust vs analytical via the two-step agent dance. The
        # returned ``DecisionOutcome`` carries the call-level telemetry
        # that ``_emit_telemetry`` will record below.
        outcome = await self._decide(
            ctx=ctx, composed_source=composed_source,
            analytical_cycles=cycles, analytical_on_chip=on_chip,
        )
        self._emit_telemetry(
            ctx=ctx, source_hash=source_hash,
            decision=outcome.decision, reason=outcome.reason,
            curated_ids=outcome.curated_ids,
            num_candidates=outcome.num_candidates_available,
            curation_dur_ms=outcome.curation_dur_ms,
            decision_dur_ms=outcome.decision_dur_ms,
        )
        if outcome.decision == "analytical":
            return SimulationResult(
                cycles=cycles, on_chip=on_chip,
                cycle_source="analytical", breakdown=breakdown,
            )
        # "fallback" decisions are degraded to analytical with the
        # telemetry row carrying ``decision="fallback"`` for the audit.
        if outcome.decision == "fallback":
            return SimulationResult(
                cycles=cycles, on_chip=on_chip,
                cycle_source="analytical", breakdown=breakdown,
            )

        # outcome.decision == "rust": execute, account, persist calibration.
        assert outcome.decision == "rust", (
            f"AgentManager.score: unexpected decision {outcome.decision!r}"
        )
        t0 = time.perf_counter()
        try:
            rust_cycles, rust_dur_ms = await asyncio.to_thread(
                _invoke_rust_evaluate,
                self._rust_evaluate_fn,
                composed_source,
                ctx=ctx,
            )
        except RustOutputMismatch as exc:
            await self._time_budget.consume(time.perf_counter() - t0)
            return SimulationResult(
                cycles=None, on_chip=None, cycle_source=None,
                error_feedback=_rust_correctness_failure_feedback(
                    cycles=exc.cycles, max_diff=exc.max_diff, message=exc.message,
                ),
            )
        elapsed = time.perf_counter() - t0
        await self._time_budget.consume(elapsed)

        source_path = _write_calibration_source(
            self._sources_dir, composed_source,
        )
        _append_calibration(
            store=self._store, ctx=ctx, source_path=source_path,
            kernel=self._kernel, preset=self._preset,
            hw_hash=self._hw_hash, compute_bw=self._compute_bw,
            run_id=self._run_id,
            analytical_cycles=cycles, analytical_on_chip=on_chip,
            rust_cycles=int(rust_cycles), rust_dur_ms=float(rust_dur_ms),
        )
        return SimulationResult(
            cycles=int(rust_cycles), on_chip=on_chip,
            cycle_source="rust",
            rust_dur_ms=float(rust_dur_ms), breakdown=breakdown,
        )

    async def _decide(
        self,
        *,
        ctx: SimContext,
        composed_source: str,
        analytical_cycles: int,
        analytical_on_chip: int,
    ) -> "DecisionOutcome":
        """Curation + decision call sequence with fall-back-on-failure.

        Returns a ``DecisionOutcome``: ``decision`` is one of
        ``"rust"`` / ``"analytical"`` / ``"fallback"``, plus the
        per-call timings and curated-record ids the score-level
        telemetry writer needs. On any exception inside the agent flow,
        returns a ``"fallback"`` outcome and logs a warning.
        """
        try:
            return await self._decide_inner(
                ctx=ctx, composed_source=composed_source,
                analytical_cycles=analytical_cycles,
                analytical_on_chip=analytical_on_chip,
            )
        except Exception as e:
            # User-confirmed PR4 fallback (HANDOFF design): agent failures
            # degrade to analytical with a warning, not a hard crash.
            # We catch broadly here because the failure could be at any
            # layer: RPC, JSON parse, candidate fetch, prompt build. The
            # parsers themselves stay strict — they only get a chance to
            # raise because this outer block exists.
            self._log_warning(
                f"AgentManager: agent decision path failed for "
                f"node={ctx.node_path!r} variant_kind={ctx.variant_kind!r}; "
                f"falling back to analytical. Error: {e!r}"
            )
            self._write_failure_artifacts(ctx=ctx, error=e)
            return DecisionOutcome(
                decision="fallback",
                reason=f"agent failure ({type(e).__name__}): {e!s}"[:400],
                curated_ids=[],
                num_candidates_available=0,
                curation_dur_ms=-1.0,
                decision_dur_ms=-1.0,
            )

    async def _decide_inner(
        self,
        *,
        ctx: SimContext,
        composed_source: str,
        analytical_cycles: int,
        analytical_on_chip: int,
    ) -> "DecisionOutcome":
        from src.autotune2.prompts import (
            build_curation_user_prompt,
            build_sim_decision_user_prompt,
            parse_curation_response,
            parse_sim_decision_response,
        )

        # Snapshot budget state once so curation + decision see consistent
        # numbers (a concurrent rust call could update the budget mid-flight
        # otherwise — not catastrophic, just noisy in logs).
        remaining = self._time_budget.remaining_seconds
        recent_avg = self._time_budget.recent_avg_seconds
        consumed = self._time_budget.consumed_seconds

        candidates = list(
            self._fetch_candidates_fn(composed_source)
        )[: self._max_curation_candidates]
        num_available = len(candidates)
        turn_artifact_dir = self._turn_artifact_dir(ctx)

        curation_dur_ms = -1.0
        if candidates:
            curation_k = min(self._curation_k, len(candidates))
            curation_user_prompt = build_curation_user_prompt(
                target_source=composed_source,
                candidates=candidates,
                k=curation_k,
            )
            t0 = time.perf_counter()
            curation_reply = await self._curation_fn(
                [{"role": "user", "content": curation_user_prompt}]
            )
            curation_dur_ms = (time.perf_counter() - t0) * 1000.0
            self._write_agent_artifacts(
                turn_artifact_dir=turn_artifact_dir,
                agent_dir_name="curator",
                user_prompt=curation_user_prompt,
                agent_response=curation_reply,
            )
            curation_text = _extract_agent_text(curation_reply)
            candidate_ids = [c.record_id for c in candidates]
            try:
                picked_ids = parse_curation_response(
                    curation_text,
                    candidate_ids=candidate_ids,
                    k=curation_k,
                )
            except (AssertionError, json.JSONDecodeError) as e:
                repair_prompt = _schema_repair_prompt(
                    error=e,
                    expected_json=(
                        '{"record_ids": ["<record_id_1>", '
                        '"<record_id_2>", "..."]}'
                    ),
                    allowed_ids=candidate_ids,
                )
                repair_conversation = [
                    {"role": "user", "content": curation_user_prompt},
                    {"role": "assistant", "content": curation_text},
                    {"role": "user", "content": repair_prompt},
                ]
                t1 = time.perf_counter()
                curation_reply = await self._curation_fn(repair_conversation)
                curation_dur_ms += (time.perf_counter() - t1) * 1000.0
                self._write_agent_artifacts(
                    turn_artifact_dir=turn_artifact_dir,
                    agent_dir_name="curator_repair_1",
                    user_prompt=repair_prompt,
                    agent_response=curation_reply,
                )
                curation_text = _extract_agent_text(curation_reply)
                picked_ids = parse_curation_response(
                    curation_text,
                    candidate_ids=candidate_ids,
                    k=curation_k,
                )
            by_id = {c.record_id: c for c in candidates}
            curated = [by_id[i] for i in picked_ids]
            curated_ids = list(picked_ids)
        else:
            # Cold start: no records for this hw_config yet. Skip the
            # curation call; the decision agent's system prompt covers
            # the empty-evidence case.
            curated = []
            curated_ids = []

        decision_user_prompt = build_sim_decision_user_prompt(
            node_path=ctx.node_path,
            is_root=ctx.is_root,
            variant_kind=ctx.variant_kind,
            attempt_index=ctx.attempt_index,
            turn_index=ctx.turn_index,
            composed_source=composed_source,
            analytical_cycles=analytical_cycles,
            analytical_on_chip=analytical_on_chip,
            remaining_seconds=remaining,
            recent_rust_avg_sec=recent_avg,
            consumed_seconds=consumed,
            curated=curated,
        )
        t0 = time.perf_counter()
        decision_reply = await self._decision_fn(
            [{"role": "user", "content": decision_user_prompt}]
        )
        decision_dur_ms = (time.perf_counter() - t0) * 1000.0
        self._write_agent_artifacts(
            turn_artifact_dir=turn_artifact_dir,
            agent_dir_name="sim_manager",
            user_prompt=decision_user_prompt,
            agent_response=decision_reply,
        )
        decision_text = _extract_agent_text(decision_reply)
        try:
            decision, reason = parse_sim_decision_response(decision_text)
        except (AssertionError, json.JSONDecodeError) as e:
            repair_prompt = _schema_repair_prompt(
                error=e,
                expected_json=(
                    '{"decision": "rust|analytical", '
                    '"reason": "<25 word reason>"}'
                ),
                allowed_ids=[],
            )
            repair_conversation = [
                {"role": "user", "content": decision_user_prompt},
                {"role": "assistant", "content": decision_text},
                {"role": "user", "content": repair_prompt},
            ]
            t1 = time.perf_counter()
            decision_reply = await self._decision_fn(repair_conversation)
            decision_dur_ms += (time.perf_counter() - t1) * 1000.0
            self._write_agent_artifacts(
                turn_artifact_dir=turn_artifact_dir,
                agent_dir_name="sim_manager_repair_1",
                user_prompt=repair_prompt,
                agent_response=decision_reply,
            )
            decision_text = _extract_agent_text(decision_reply)
            decision, reason = parse_sim_decision_response(decision_text)
        return DecisionOutcome(
            decision=decision,
            reason=reason,
            curated_ids=curated_ids,
            num_candidates_available=num_available,
            curation_dur_ms=curation_dur_ms,
            decision_dur_ms=decision_dur_ms,
        )

    def _emit_telemetry(
        self,
        *,
        ctx: SimContext,
        source_hash: str,
        decision: str,
        reason: str,
        curated_ids: list[str],
        num_candidates: int,
        curation_dur_ms: float,
        decision_dur_ms: float,
    ) -> None:
        """Write one telemetry row if a store is wired; otherwise no-op.

        Fail-loud on a bad store, same as ``CalibrationStore.append``:
        if the telemetry sink can't accept writes (disk full, bad
        permissions, schema violation), we want to discover it during
        the run, not hours later when reviewing logs.
        """
        if self._telemetry_store is None:
            return
        from src.autotune2.agent_telemetry import AgentDecisionRecord
        record = AgentDecisionRecord(
            stage="sim_decision",
            run_id=self._run_id,
            kernel=self._kernel,
            preset=self._preset,
            hw_config_hash=self._hw_hash,
            node_path=ctx.node_path,
            is_root=ctx.is_root,
            timestamp=datetime.now(timezone.utc).isoformat(),
            composed_source_hash=source_hash,
            variant_kind=ctx.variant_kind,
            attempt_index=ctx.attempt_index,
            turn_index=ctx.turn_index,
            decision=decision,
            reason=reason,
            curated_record_ids=list(curated_ids),
            num_candidates_available=int(num_candidates),
            curation_dur_ms=float(curation_dur_ms),
            decision_dur_ms=float(decision_dur_ms),
        )
        self._telemetry_store.append(record)

    def _turn_artifact_dir(self, ctx: SimContext) -> Path | None:
        if ctx.turn_artifact_dir is not None:
            return Path(ctx.turn_artifact_dir)
        if self._turn_artifact_dir_fn is None:
            return None
        path = self._turn_artifact_dir_fn(ctx)
        return None if path is None else Path(path)

    def _write_agent_artifacts(
        self,
        *,
        turn_artifact_dir: Path | None,
        agent_dir_name: str,
        user_prompt: str,
        agent_response: object,
    ) -> None:
        if turn_artifact_dir is None:
            return
        from src.autotune2.agent_telemetry import write_agent_call_artifacts

        write_agent_call_artifacts(
            Path(turn_artifact_dir) / "sim_manager" / agent_dir_name,
            user_prompt=user_prompt,
            agent_response=agent_response,
        )

    def _write_failure_artifacts(
        self,
        *,
        ctx: SimContext,
        error: Exception,
    ) -> None:
        turn_artifact_dir = self._turn_artifact_dir(ctx)
        if turn_artifact_dir is None:
            return
        from src.autotune2.agent_telemetry import write_agent_failure_artifacts

        write_agent_failure_artifacts(
            Path(turn_artifact_dir) / "sim_manager",
            error=error,
        )


@dataclass(frozen=True)
class DecisionOutcome:
    """One ``AgentManager._decide`` result with telemetry fields.

    ``decision`` is one of ``"rust"`` / ``"analytical"`` (agent
    succeeded) or ``"fallback"`` (an exception was caught at the outer
    boundary). Telemetry fields use ``-1.0`` / empty list to signal
    "did not run".
    """

    decision: str
    reason: str
    curated_ids: list[str]
    num_candidates_available: int
    curation_dur_ms: float
    decision_dur_ms: float


def _extract_agent_text(reply: object) -> str:
    """Get the assistant text out of an AgentResponse-or-str reply.

    Mirrors ``src/autotune2/search._coerce_agent_response`` but as a
    local helper so sim_manager doesn't pull in the search module's
    import graph (search imports prompts, prompts has no reverse
    dependency on sim_manager today — the cycle would be a regression).
    """
    if isinstance(reply, str):
        return reply
    text = getattr(reply, "text", None)
    assert isinstance(text, str), (
        f"_extract_agent_text: agent reply must be a string or carry a "
        f"string `.text` attribute (AgentResponse-shaped); got "
        f"{type(reply).__name__}={reply!r}"
    )
    return text


def _schema_repair_prompt(
    *,
    error: Exception,
    expected_json: str,
    allowed_ids: list[str],
) -> str:
    prompt = (
        "Your previous response failed schema validation.\n\n"
        f"Error:\n{type(error).__name__}: {error}\n\n"
        "Re-emit only a JSON object matching this schema. Do not include "
        "markdown, commentary, or code fences.\n\n"
        f"{expected_json}\n"
    )
    if allowed_ids:
        prompt += (
            "\nAllowed record_ids, copy exactly from this list:\n"
            + json.dumps(allowed_ids, indent=2)
            + "\n"
        )
    return prompt
