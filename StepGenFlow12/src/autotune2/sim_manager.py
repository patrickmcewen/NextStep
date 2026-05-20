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

import traceback as _tb
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol


# (composed_source) -> (cycles, on_chip_bytes). Same shape as the legacy
# ``ScoreFn`` in ``compose.py``; re-declared here so impl modules don't
# need to import ``compose``.
ScoreFn = Callable[[str], tuple[int, int]]


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

    async def score(
        self, ctx: SimContext, composed_source: str,
    ) -> SimulationResult: ...


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
