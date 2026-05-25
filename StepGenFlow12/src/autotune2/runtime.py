"""Phase 6 wire-up for autotune2.

Bridges the Phase 5 search driver to:
  - **Top-K rust promotion** at the root: take the analytical
    Pareto-best entries from the root's library and re-score them
    with the rust simulator at ``/workspace/NextStep/StepDB/evaluate.py``.
  - **Run summary** JSON: write the root's full Pareto-front plus the
    rust-validated winner to ``<ckpt_dir>/autotune2_summary.json``.
  - **Real-component factories** (lazy-import wrappers) for the
    Anthropic agent + 4-gate verification chain — kept thin so the
    test environment can inject mocks via the Phase 5 ``AgentFn`` /
    ``VerifierFn`` interfaces.

Top-K policy
------------
``pick_top_k_pareto_entries`` walks every cell in the root's library,
flattens to a single list, and keeps the K entries that are
non-dominated on (cycles, on_chip). Ties broken by lower cycles first,
then lower on_chip. When the unique non-dominated set is larger than
K, we drop the largest-cycle entries first (rationale: the rust
simulator catches cycle modeling errors more often than on-chip
errors; spending the rust budget on the highest-cycle Pareto wing is
lower marginal value).

Rust evaluator interface
------------------------
The injected ``rust_evaluate_fn`` has signature

    rust_evaluate_fn(composed_source: str) -> tuple[int, float]

returning ``(cycles, dur_ms)``. The argument is the autotuner's composed
*DSL* source (``def tiled_reference(...)`` plus descendant defs).
``build_rust_evaluate_fn`` runs ``src.dsl_to_step.translate`` on it before
handing the resulting ``def build_graph(...)`` STeP source to
``StepDB/evaluate.py::evaluate_kernel``, which requires ``build_graph``.
Production callers build the closure over the per-kernel ``work_dir`` and
the rust simulator's ``hbm_config`` / ``sim_config``; unit tests inject
deterministic fakes that ignore the translation step.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.autotune2.compose import compose_source
from src.autotune2.contracts import DesignEntry, NodeLibrary
from src.autotune2.pareto import dominates
from src.autotune2.search import (
    AutotuneResult,
    gather_descendants_postorder,
)


# (composed_source, optional metadata kwargs) -> (cycles, dur_ms)
RustEvaluateFn = Callable[..., tuple[int, float]]

# Raised by the ``build_rust_evaluate_fn`` closure when the rust functional
# sim produces a tensor that doesn't match the PyTorch gold reference;
# defined in ``sim_manager`` (where it's caught) to avoid an import cycle
# back through ``search`` -> ``sim_manager``.
from src.autotune2.sim_manager import RustOutputMismatch


def _safe_path_component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("._")
    return safe or "unnamed"


def _safe_path_parts(value: str | None) -> list[str]:
    if value is None or not str(value).strip():
        return ["rust_eval"]
    raw = Path(str(value))
    assert not raw.is_absolute(), (
        f"rust work label must be relative, got {value!r}"
    )
    out: list[str] = []
    for part in raw.parts:
        assert part not in ("", ".", ".."), (
            f"rust work label contains invalid path component {part!r}: "
            f"{value!r}"
        )
        out.append(_safe_path_component(part))
    return out or ["rust_eval"]


def _allocate_labeled_work_dir(
    *,
    root: Path,
    node_path: str | None,
    run_label: str | None,
) -> Path:
    base = root.joinpath(*_safe_path_parts(node_path)) if node_path else root
    label_parts = _safe_path_parts(run_label)
    parent = base.joinpath(*label_parts[:-1])
    leaf = label_parts[-1]
    suffix = 0
    while True:
        name = leaf if suffix == 0 else f"{leaf}__{suffix}"
        candidate = parent / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1


def _invoke_rust_evaluate(
    rust_evaluate_fn: RustEvaluateFn,
    composed_source: str,
    *,
    node_path: str | None = None,
    run_label: str | None = None,
) -> tuple[int, float]:
    """Call a rust evaluator with metadata when it accepts metadata.

    Unit tests and older integrations often use ``lambda src: ...`` fakes.
    Preserve that API while allowing the real evaluator to receive enough
    context to lay artifacts out under traceable directories.
    """
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
            node_path=node_path,
            run_label=run_label,
        )
    return rust_evaluate_fn(composed_source)


@dataclass
class RustPromotionResult:
    """One entry that was re-scored by the rust simulator."""

    entry: DesignEntry
    """The DesignEntry from the root's analytical library."""

    rust_cycles: int
    """Cycle count reported by the rust simulator."""

    rust_dur_ms: float
    """Wall-clock duration of the rust simulator run in milliseconds."""

    composed_source: str
    """The exact source string fed to the rust evaluator. Stored for
    reproducibility — large strings, but persistence is opt-in via
    ``write_autotune2_summary(include_sources=True)``."""


# ---------------------------------------------------------------------------
# Top-K selection
# ---------------------------------------------------------------------------


def pick_top_k_pareto_entries(
    root_library: NodeLibrary, k: int,
) -> list[DesignEntry]:
    """Flatten root's cells into a single (cycles, on_chip)-Pareto list,
    keep the K most-attractive non-dominated entries.

    "Most attractive" = lowest cycles first, then lowest on_chip. If
    fewer than K non-dominated entries exist, returns whatever the
    Pareto front has (no padding).
    """
    assert k >= 1, f"pick_top_k_pareto_entries: k must be >= 1, got {k!r}"

    all_entries: list[DesignEntry] = []
    for by_out in root_library.values():
        for cell in by_out.values():
            all_entries.extend(cell)
    assert all_entries, (
        "pick_top_k_pareto_entries: root_library has no entries; the search "
        "driver must seed at least the pass-1 baseline before promotion"
    )

    nondom: list[DesignEntry] = []
    for e in all_entries:
        if any(dominates(other, e) for other in all_entries if other is not e):
            continue
        nondom.append(e)
    nondom.sort(key=lambda e: (e.cycles, e.on_chip))
    return nondom[:k]


# ---------------------------------------------------------------------------
# Rust promotion
# ---------------------------------------------------------------------------


def _build_composed_source_for_entry(entry: DesignEntry) -> str:
    """Reconstruct the full composed source for a root entry.

    The root entry's own ``dsl`` is the ``tiled_reference`` body; its
    ``children_picks`` references descendant entries (recursively),
    whose ``children_picks`` reference their own descendants, and so on
    down to the leaves. We walk this chain in post-order via
    ``gather_descendants_postorder``.
    """
    descendants = gather_descendants_postorder(entry)
    return compose_source(
        parent_dsl=entry.dsl,
        descendant_dsls_postorder=descendants,
    )


def promote_top_k(
    *,
    root_library: NodeLibrary,
    k: int,
    rust_evaluate_fn: RustEvaluateFn,
) -> list[RustPromotionResult]:
    """Pick top-K and rust-evaluate each.

    Picks are dispatched concurrently via a thread pool — each rust
    evaluator call is dominated by a blocking ``subprocess.run`` on the
    rust sim, so threads parallelize the wall-clock cost cleanly. The
    closure built by ``build_rust_evaluate_fn`` writes per-call
    artifacts to a unique subdir so concurrent calls don't collide.
    Entries whose ``cycle_source == "rust"`` are not re-evaluated —
    their ``cycles`` was already produced by the same rust evaluator
    in-loop, and a redundant rerun would burn budget without changing
    the number. ``rust_dur_ms`` is read off the entry when present and
    defaults to 0.0 when the entry was produced by a manager that
    didn't record it (legacy snapshots, mocks).

    Results are returned sorted by rust_cycles (best first). The list
    length equals ``len(pick_top_k_pareto_entries(root_library, k))``;
    a rust failure is **not** silently skipped — the rust evaluator
    must succeed for every promoted entry (otherwise the calling
    pipeline has a bug worth surfacing).
    """
    from concurrent.futures import ThreadPoolExecutor

    picks = pick_top_k_pareto_entries(root_library, k)
    composed_sources = [_build_composed_source_for_entry(e) for e in picks]

    # Two groups: entries that need a fresh rust call vs entries whose
    # ``cycles`` was already produced by rust in-loop (cycle_source ==
    # "rust"). The latter group reuses ``entry.cycles`` directly so we
    # don't pay the rust budget twice for the same composed source.
    needs_rust_idx = [
        i for i, e in enumerate(picks) if e.cycle_source != "rust"
    ]
    needs_rust_sources = [composed_sources[i] for i in needs_rust_idx]
    fresh_results: dict[int, tuple[int, float]] = {}
    if needs_rust_sources:
        labeled_sources = [
            (
                composed_sources[i],
                f"top_k/{i}_{_safe_path_component(picks[i].provenance)}",
            )
            for i in needs_rust_idx
        ]
        # max_workers caps parallelism so a large top_k doesn't fork an
        # arbitrarily large number of rust subprocesses. K is typically
        # <=8 in practice; cap at the number of pending calls so we
        # never spin idle threads.
        with ThreadPoolExecutor(max_workers=len(needs_rust_sources)) as ex:
            for i, result in zip(
                needs_rust_idx,
                ex.map(
                    lambda item: _invoke_rust_evaluate(
                        rust_evaluate_fn,
                        item[0],
                        node_path="root",
                        run_label=item[1],
                    ),
                    labeled_sources,
                ),
            ):
                fresh_results[i] = result

    out: list[RustPromotionResult] = []
    for i, (entry, composed) in enumerate(zip(picks, composed_sources)):
        if i in fresh_results:
            cycles, dur_ms = fresh_results[i]
        else:
            # entry.cycles was rust-measured in-loop; reuse it. The
            # rust_dur_ms attribute isn't part of DesignEntry today
            # (it's only on SimulationResult), so default to 0.0 —
            # the summary's "rust duration" column then reflects "this
            # number came from a cached in-loop run, not a fresh rerun".
            cycles = entry.cycles
            dur_ms = 0.0
        out.append(RustPromotionResult(
            entry=entry,
            rust_cycles=cycles,
            rust_dur_ms=dur_ms,
            composed_source=composed,
        ))
    out.sort(key=lambda r: r.rust_cycles)
    return out


# ---------------------------------------------------------------------------
# Final pick — one rust-evaluated variant from the root Pareto
# ---------------------------------------------------------------------------


def _root_pareto_entries(root_library: NodeLibrary) -> list[DesignEntry]:
    """Flatten root cells and return the non-dominated entries on
    ``(cycles, on_chip)``. Shared by ``final_pick`` and
    ``final_pick_agent`` so both paths agree on the candidate set.
    """
    all_entries: list[DesignEntry] = []
    for by_out in root_library.values():
        for cell in by_out.values():
            all_entries.extend(cell)
    assert all_entries, (
        "_root_pareto_entries: root_library has no entries; the search "
        "driver must seed at least the pass-1 baseline before the final pick"
    )
    return [
        e for e in all_entries
        if not any(dominates(o, e) for o in all_entries if o is not e)
    ]


def _build_promotion_for_pick(
    *,
    picked: DesignEntry,
    rust_evaluate_fn: RustEvaluateFn,
    run_label: str = "final_pick",
) -> RustPromotionResult:
    """Compose the source for ``picked`` and either reuse its rust-measured
    cycles (when ``cycle_source == "rust"``) or invoke ``rust_evaluate_fn``
    exactly once. Shared by ``final_pick`` and ``final_pick_agent``.
    """
    composed = _build_composed_source_for_entry(picked)
    if picked.cycle_source == "rust":
        cycles = picked.cycles
        dur_ms = 0.0
    else:
        cycles, dur_ms = _invoke_rust_evaluate(
            rust_evaluate_fn,
            composed,
            node_path="root",
            run_label=run_label,
        )
    return RustPromotionResult(
        entry=picked,
        rust_cycles=cycles,
        rust_dur_ms=dur_ms,
        composed_source=composed,
    )


def final_pick(
    *,
    root_library: NodeLibrary,
    rust_evaluate_fn: RustEvaluateFn,
    strategy: str = "min_cycles",
) -> list[RustPromotionResult]:
    """Pick exactly one root-Pareto entry and rust-evaluate it.

    Replaces ``promote_top_k(k=N)`` with a single deterministic pick + one
    ground-truth rust evaluation — the "comparable across all sim_mode
    configs" number from HANDOFF design decision #6. The return shape is
    a single-element ``list[RustPromotionResult]`` so
    ``write_autotune2_summary`` keeps working unchanged.

    Strategies
    ----------
    ``min_cycles``: lowest recorded ``cycles`` from the root Pareto front,
        with ``on_chip`` as tiebreaker. When the library is mixed-source
        (any entry has ``cycle_source == "rust"``), the pick is restricted
        to rust-sourced entries — mixing analytical and rust cycles is
        comparing incommensurable numbers (HANDOFF design decision #5).
        When the picked entry was already rust-measured in-loop
        (``cycle_source == "rust"``), its ``cycles`` is reused and
        ``rust_dur_ms`` is 0.0; otherwise we invoke ``rust_evaluate_fn``
        once for ground truth.

    For ``strategy="agent"`` (FinalPickAgent — HANDOFF decision #8), see
    the ``final_pick_agent`` async function below. The agent path is a
    separate entry point because it must ``await`` the curation +
    final-pick LLM calls; keeping the deterministic ``min_cycles`` path
    synchronous avoids forcing every existing call site through an event
    loop just to use the historical default.
    """
    assert strategy == "min_cycles", (
        f"final_pick: only strategy='min_cycles' is supported by this "
        f"sync entry point, got {strategy!r}. For 'agent', call "
        f"final_pick_agent(...) — it's async because the curation + "
        f"final-pick LLM calls have to be awaited."
    )

    nondom = _root_pareto_entries(root_library)
    has_rust = any(e.cycle_source == "rust" for e in nondom)
    candidates = (
        [e for e in nondom if e.cycle_source == "rust"]
        if has_rust else nondom
    )
    candidates.sort(key=lambda e: (e.cycles, e.on_chip))
    picked = candidates[0]
    return [_build_promotion_for_pick(
        picked=picked, rust_evaluate_fn=rust_evaluate_fn,
        run_label=f"final_pick/{_safe_path_component(picked.provenance)}",
    )]


# ---------------------------------------------------------------------------
# Final pick — agent-driven (FinalPickAgent, HANDOFF design decision #8)
# ---------------------------------------------------------------------------


async def final_pick_agent(
    *,
    root_library: NodeLibrary,
    rust_evaluate_fn: RustEvaluateFn,
    curation_agent_fn,
    final_pick_agent_fn,
    fetch_candidates_fn,
    root_path: str,
    kernel: str,
    preset: str,
    curation_k: int = 4,
    curation_max_candidates: int = 50,
    log_warning: Callable[[str], None] = print,
    telemetry_store=None,  # AgentDecisionStore | None
    hw_config_hash: str = "",
    run_id: str = "",
) -> list[RustPromotionResult]:
    """Agent-driven end-of-run final pick.

    Walks the root Pareto front, asks the curation agent for the K most
    predictive calibration records per candidate (cold-start tolerated),
    feeds the full set to the final-pick agent, then either reuses
    ``entry.cycles`` (when the picked entry is ``cycle_source == "rust"``)
    or invokes ``rust_evaluate_fn`` exactly once. Returns the same
    single-element ``list[RustPromotionResult]`` shape ``final_pick``
    returns so ``write_autotune2_summary`` is path-independent.

    Fall-back policy (mirrors ``AgentManager`` in ``sim_manager``): any
    exception inside the agent call sequence — RPC error, parser
    assertion, malformed curation pick — is caught at the call-site
    boundary, logged via ``log_warning``, and the deterministic
    ``min_cycles`` pick is used instead so the run still produces a
    reportable number. The fenced-JSON parsers themselves stay strict.

    Skipping the agent entirely when the Pareto front has exactly one
    entry — there is nothing to pick. This also lets a degenerate root
    library (e.g. only the seeded baseline survived) avoid an agent
    round-trip for a foregone conclusion.
    """
    import time as _time
    from src.autotune2.prompts import (
        FinalPickCandidate,
        build_curation_user_prompt,
        build_final_pick_user_prompt,
        parse_curation_response,
        parse_final_pick_response,
    )
    from src.autotune2.sim_manager import _extract_agent_text

    assert 1 <= curation_k <= curation_max_candidates, (
        f"final_pick_agent: curation_k must be in [1, curation_max_candidates="
        f"{curation_max_candidates}], got {curation_k!r}"
    )

    nondom = _root_pareto_entries(root_library)
    nondom.sort(key=lambda e: (e.cycles, e.on_chip))

    if len(nondom) == 1:
        # Trivial Pareto — no decision to make. Promote it directly.
        _emit_final_pick_telemetry(
            telemetry_store=telemetry_store,
            root_path=root_path, kernel=kernel, preset=preset,
            hw_config_hash=hw_config_hash, run_id=run_id,
            composed_source=_build_composed_source_for_entry(nondom[0]),
            decision="pareto_short_circuit",
            reason="single non-dominated entry; no agent round-trip needed",
            curated_ids=[], num_candidates=0,
            curation_dur_ms=-1.0, decision_dur_ms=-1.0,
            picked_variant_index=0, num_pareto_entries=1,
        )
        return [_build_promotion_for_pick(
            picked=nondom[0], rust_evaluate_fn=rust_evaluate_fn,
            run_label=(
                "final_pick_agent/pareto_short_circuit/"
                f"{_safe_path_component(nondom[0].provenance)}"
            ),
        )]

    composed_sources = [_build_composed_source_for_entry(e) for e in nondom]

    # Telemetry fields we update along the way so the row reflects what
    # actually happened, regardless of which branch we exit through.
    picked_variant_index = -1
    pick_reason = ""
    total_curation_ms = 0.0
    decision_ms = -1.0
    # Per-candidate curated ids; we record the picked candidate's set
    # post-hoc so the telemetry row answers "what evidence did the
    # agent see for the variant it chose?"
    per_candidate_curated_ids: list[list[str]] = []
    per_candidate_num_available: list[int] = []
    telemetry_decision = "picked"

    try:
        # Curate per candidate so each renders with its own evidence.
        candidates: list[FinalPickCandidate] = []
        for idx, (entry, composed) in enumerate(
            zip(nondom, composed_sources),
        ):
            raw = list(fetch_candidates_fn(composed))[:curation_max_candidates]
            per_candidate_num_available.append(len(raw))
            if raw:
                k = min(curation_k, len(raw))
                curation_prompt = build_curation_user_prompt(
                    target_source=composed, candidates=raw, k=k,
                )
                t0 = _time.perf_counter()
                curation_reply = await curation_agent_fn(
                    [{"role": "user", "content": curation_prompt}]
                )
                total_curation_ms += (_time.perf_counter() - t0) * 1000.0
                picked_ids = parse_curation_response(
                    _extract_agent_text(curation_reply),
                    candidate_ids=[c.record_id for c in raw],
                    k=k,
                )
                by_id = {c.record_id: c for c in raw}
                curated = [by_id[i] for i in picked_ids]
                per_candidate_curated_ids.append(list(picked_ids))
            else:
                curated = []
                per_candidate_curated_ids.append([])
            candidates.append(FinalPickCandidate(
                variant_index=idx,
                cycles=entry.cycles,
                on_chip=entry.on_chip,
                cycle_source=entry.cycle_source,
                composed_source=composed,
                curated=curated,
            ))

        user_prompt = build_final_pick_user_prompt(
            root_path=root_path, kernel=kernel, preset=preset,
            candidates=candidates,
        )
        t0 = _time.perf_counter()
        pick_reply = await final_pick_agent_fn(
            [{"role": "user", "content": user_prompt}]
        )
        decision_ms = (_time.perf_counter() - t0) * 1000.0
        picked_idx, pick_reason = parse_final_pick_response(
            _extract_agent_text(pick_reply), num_candidates=len(candidates),
        )
        picked_variant_index = picked_idx
        picked = nondom[picked_idx]
    except Exception as e:
        # Fall back to the deterministic min_cycles pick so the run still
        # produces a reportable number — same shape as AgentManager's
        # in-loop fallback. Mixed-source restriction (rust-sourced
        # entries only when any exist) mirrors ``final_pick``.
        log_warning(
            f"final_pick_agent: agent path failed; falling back to "
            f"min_cycles. Error: {e!r}"
        )
        telemetry_decision = "fallback"
        pick_reason = f"agent failure ({type(e).__name__}): {e!s}"[:400]
        has_rust = any(e.cycle_source == "rust" for e in nondom)
        fallback_pool = (
            [e for e in nondom if e.cycle_source == "rust"]
            if has_rust else list(nondom)
        )
        fallback_pool.sort(key=lambda e: (e.cycles, e.on_chip))
        picked = fallback_pool[0]
        picked_variant_index = nondom.index(picked)

    # Picked-candidate context: use whatever per-candidate state we
    # accumulated for this index (may be empty if the failure occurred
    # before the picked candidate's curation ran).
    picked_curated = (
        per_candidate_curated_ids[picked_variant_index]
        if 0 <= picked_variant_index < len(per_candidate_curated_ids) else []
    )
    picked_num_available = (
        per_candidate_num_available[picked_variant_index]
        if 0 <= picked_variant_index < len(per_candidate_num_available) else 0
    )
    _emit_final_pick_telemetry(
        telemetry_store=telemetry_store,
        root_path=root_path, kernel=kernel, preset=preset,
        hw_config_hash=hw_config_hash, run_id=run_id,
        composed_source=composed_sources[picked_variant_index],
        decision=telemetry_decision,
        reason=pick_reason,
        curated_ids=picked_curated,
        num_candidates=picked_num_available,
        curation_dur_ms=(
            total_curation_ms if total_curation_ms > 0 else -1.0
        ),
        decision_dur_ms=decision_ms,
        picked_variant_index=picked_variant_index,
        num_pareto_entries=len(nondom),
    )

    return [_build_promotion_for_pick(
        picked=picked, rust_evaluate_fn=rust_evaluate_fn,
        run_label=(
            f"final_pick_agent/pick_{picked_variant_index}_"
            f"{_safe_path_component(picked.provenance)}"
        ),
    )]


def _emit_final_pick_telemetry(
    *,
    telemetry_store,
    root_path: str,
    kernel: str,
    preset: str,
    hw_config_hash: str,
    run_id: str,
    composed_source: str,
    decision: str,
    reason: str,
    curated_ids: list[str],
    num_candidates: int,
    curation_dur_ms: float,
    decision_dur_ms: float,
    picked_variant_index: int,
    num_pareto_entries: int,
) -> None:
    """Write one final-pick telemetry row if a store is wired.

    Fail-loud on a bad store, same as ``CalibrationStore.append`` —
    a side-channel observability store that silently swallows writes
    is worse than no store at all.
    """
    if telemetry_store is None:
        return
    import hashlib as _hashlib
    from datetime import datetime as _dt, timezone as _tz
    from src.autotune2.agent_telemetry import AgentDecisionRecord
    source_hash = _hashlib.sha256(composed_source.encode("utf-8")).hexdigest()
    record = AgentDecisionRecord(
        stage="final_pick",
        run_id=run_id,
        kernel=kernel,
        preset=preset,
        hw_config_hash=hw_config_hash,
        node_path=root_path,
        is_root=True,
        timestamp=_dt.now(_tz.utc).isoformat(),
        composed_source_hash=source_hash,
        variant_kind="",
        attempt_index=-1,
        turn_index=-1,
        decision=decision,
        reason=reason,
        curated_record_ids=list(curated_ids),
        num_candidates_available=int(num_candidates),
        curation_dur_ms=float(curation_dur_ms),
        decision_dur_ms=float(decision_dur_ms),
        picked_variant_index=int(picked_variant_index),
        num_pareto_entries=int(num_pareto_entries),
    )
    telemetry_store.append(record)


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


def write_autotune2_summary(
    *,
    autotune_result: AutotuneResult,
    rust_promotions: list[RustPromotionResult],
    out_path: Path,
    include_sources: bool = False,
    root_pick_strategy: str | None = None,
) -> None:
    """Emit a JSON summary of an autotune2 run.

    Fields:
      - ``root_path``: the root node's path
      - ``library_sizes``: backwards-compatible alias for
        ``library_cell_counts``
      - ``library_cell_counts``: ``{node_path: <num contract cells>}``
      - ``library_entry_counts``: ``{node_path: <num DesignEntry rows>}``
      - ``library_llm_entry_counts``: ``{node_path: <num non-baseline rows>}``
      - ``root_pareto``: the root's analytical Pareto entries as
        ``[{cycles, on_chip, provenance}, ...]``, sorted by cycles
      - ``rust_winners``: the rust-promoted entries with their
        analytical-vs-rust cycle comparison
      - ``best_rust_entry``: the lowest rust_cycles entry (or None if
        no promotions)
      - ``root_pick_strategy``: which root-pick path produced the
        ``rust_winners`` list — ``"final_pick:<strategy>"``,
        ``"top_k:<k>"``, or ``None`` when the caller didn't tag it.
        Lets downstream consumers tell which run mode produced the
        reported number.
      - ``best_composed_source``: only when ``include_sources=True``;
        the source string of the rust-best entry (can be megabytes for
        large kernels)
    """
    library_cell_counts = {
        path: sum(len(by_out) for by_out in lib.values())
        for path, lib in autotune_result.libraries.items()
    }
    library_entry_counts = {
        path: sum(len(cell) for by_out in lib.values() for cell in by_out.values())
        for path, lib in autotune_result.libraries.items()
    }
    library_llm_entry_counts = {
        path: sum(
            1
            for by_out in lib.values()
            for cell in by_out.values()
            for entry in cell
            if entry.provenance != "pass1_baseline"
        )
        for path, lib in autotune_result.libraries.items()
    }

    root_lib = autotune_result.root_library()
    root_pareto: list[dict] = []
    for by_out in root_lib.values():
        for cell in by_out.values():
            for entry in cell:
                root_pareto.append({
                    "cycles": entry.cycles,
                    "on_chip": entry.on_chip,
                    "provenance": entry.provenance,
                })
    root_pareto.sort(key=lambda d: (d["cycles"], d["on_chip"]))

    rust_winners: list[dict] = []
    for r in rust_promotions:
        rust_winners.append({
            "analytical_cycles": r.entry.cycles,
            "analytical_on_chip": r.entry.on_chip,
            "rust_cycles": r.rust_cycles,
            "rust_dur_ms": r.rust_dur_ms,
            "provenance": r.entry.provenance,
        })

    payload: dict = {
        "root_path": autotune_result.root_path,
        "library_sizes": library_cell_counts,
        "library_cell_counts": library_cell_counts,
        "library_entry_counts": library_entry_counts,
        "library_llm_entry_counts": library_llm_entry_counts,
        "root_pareto": root_pareto,
        "rust_winners": rust_winners,
        "best_rust_entry": rust_winners[0] if rust_winners else None,
        "root_pick_strategy": root_pick_strategy,
    }
    if include_sources and rust_promotions:
        payload["best_composed_source"] = rust_promotions[0].composed_source

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Real-component factory sketches
# ---------------------------------------------------------------------------


def build_rust_evaluate_fn(
    *,
    work_dir: Path,
    kernel_name: str,
    preset: str,
    tensors: dict | None = None,
    timing_only: bool = False,
    max_total_compute_bw: int | None = None,
    rust_sim_timeout_seconds: float | None = None,
    rust_timeout_cycles: int = 10**15,
) -> RustEvaluateFn:
    """Build a rust evaluator that writes a temp DSL file and invokes
    ``StepDB/evaluate.py``'s rust simulator subprocess.

    Lazy-imports the StepDB module so test environments without the rust
    toolchain can still import autotune2.runtime. The returned closure
    writes the composed source to ``work_dir / "step_impl.py"`` before
    each call (overwrites prior contents). ``timing_only=False`` (the
    default) runs the functional simulator and compares the produced
    output tensor against the PyTorch reference — necessary to catch
    structurally broken graphs that drain early and report a meaningless
    cycle count (e.g. an undersized accumulator silently truncating a
    matmul reduction). When the compare fails the closure raises
    ``RustOutputMismatch`` carrying the offending cycles and ``max_diff``;
    in-loop managers catch it and convert to LLM ``error_feedback``,
    while ``promote_top_k`` / ``final_pick`` let it propagate so a
    wrong-output winner can't be silently recorded. Pass
    ``timing_only=True`` only when correctness is checked elsewhere
    (e.g. a calibration-only script that just wants raw cycle numbers).

    ``max_total_compute_bw`` is forwarded to ``evaluate_kernel`` so each
    compute op's ``compute_bw`` is rescaled (in place) to sum to that
    budget before serialization — mirrors what the analytical scorer
    does via ``compose._rescale_compute_bw``. Pass the same value here
    that ``make_analytical_scorer`` got, or the analytical-vs-rust
    cycle comparison in the summary uses two different cost models.

    ``rust_sim_timeout_seconds`` is forwarded to StepDB's simulator
    subprocess timeout. If that timeout fires, the rust evaluator returns
    ``rust_timeout_cycles`` instead of raising, so the variant is recorded
    as a deliberately bad high-cycle measurement and the autotune run can
    keep moving.

    Each call writes its artifacts under a unique labeled subdirectory.
    In-loop callers pass ``node_path`` and a trace ``run_label`` so the
    hierarchy mirrors the autotune checkpoint that produced the variant,
    e.g. ``_rust_work/root/moe/pass_0/.../session_0/turn_1``. If the
    same label is reused, the final directory gets ``__<n>`` appended so
    concurrent evaluations never overwrite each other's ``step_impl.py`` /
    ``graph.pb`` / sim outputs.

    ``tensors`` is the default tensor dictionary passed into
    ``evaluate_kernel``. Non-root autotune nodes can override it per call
    via ``evaluate(..., tensors_override=node_tensors)`` so their synthetic
    wrappers resolve contract-local inputs such as ``normed_2`` or ``Q``
    instead of StepDB's root-kernel precompute dict.
    """
    import threading
    build_timeout_seconds = rust_sim_timeout_seconds
    build_timeout_cycles = int(rust_timeout_cycles)
    _counter = {"i": 0}
    _counter_lock = threading.Lock()

    def evaluate(
        composed_source: str,
        *,
        tensors_override: dict | None = None,
        node_path: str | None = None,
        run_label: str | None = None,
        rust_sim_timeout_seconds: float | None = None,
        rust_timeout_cycles: int | None = None,
    ) -> tuple[int, float]:
        import sys
        import time
        from pathlib import Path as _Path

        # StepDB and step_tl ship as flat directories (not pip-installed
        # packages); mirror orchestrator.py's path setup so `from evaluate
        # import ...` resolves and StepDB's own bare imports (`from loader
        # import ...`, `from sim import ...`) work inside evaluate_kernel.
        _deio_root = _Path(__file__).resolve().parents[3]  # NextStep/
        for p in (
            _deio_root / "StepDB",
            _deio_root / "step_tl" / "src",
            _deio_root / "step_tl" / "src" / "proto",
        ):
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)

        from evaluate import evaluate_kernel  # type: ignore  # StepDB/evaluate.py

        # The composed source is a DSL `tiled_reference` body; StepDB's
        # evaluate_kernel exec's the source as-is and requires it to define
        # `build_graph`. Run the deterministic DSL → STeP IR translator so
        # the file written into work_dir mirrors what evaluate_kernel exec's
        # and the same string is fed via `step_impl_source`.
        from src.dsl_to_step import translate as _dsl_to_step_translate
        step_source = _dsl_to_step_translate(composed_source)

        with _counter_lock:
            if run_label is None:
                idx = _counter["i"]
                _counter["i"] += 1
                run_label = f"rust_eval_{idx}"
            sub_work_dir = _allocate_labeled_work_dir(
                root=work_dir,
                node_path=node_path,
                run_label=run_label,
            )
        (sub_work_dir / "step_impl.py").write_text(step_source)
        t0 = time.perf_counter()
        result = evaluate_kernel(
            kernel_name=kernel_name,
            preset=preset,
            work_dir=str(sub_work_dir),
            timing_only=timing_only,
            step_impl_source=step_source,
            max_total_compute_bw=max_total_compute_bw,
            sim_timeout_seconds=(
                rust_sim_timeout_seconds
                if rust_sim_timeout_seconds is not None
                else build_timeout_seconds
            ),
            tensors_override=(
                tensors if tensors_override is None else tensors_override
            ),
        )
        dur_ms = (time.perf_counter() - t0) * 1000.0

        if (
            not result.success
            and result.stage == "simulate"
            and result.error_message is not None
            and "timed out" in result.error_message.lower()
        ):
            return int(
                rust_timeout_cycles
                if rust_timeout_cycles is not None
                else build_timeout_cycles
            ), dur_ms

        # Stage "correctness" is the only failure mode that means "the sim
        # ran and produced a number, but the produced output is wrong" —
        # raise a typed exception so callers can distinguish from sim-crash
        # failures and turn it into actionable LLM feedback.
        if not result.success and result.stage == "correctness":
            raise RustOutputMismatch(
                cycles=result.cycle_time if result.cycle_time is not None else float("nan"),
                max_diff=result.max_diff if result.max_diff is not None else float("nan"),
                message=(
                    f"rust functional output diverged from gold for "
                    f"kernel={kernel_name} preset={preset} at "
                    f"{sub_work_dir}: {result.error_message}"
                ),
            )
        assert result.success, (
            f"build_rust_evaluate_fn: evaluate_kernel failed at stage "
            f"{result.stage!r} for kernel={kernel_name} preset={preset}: "
            f"{result.error_message}"
        )
        assert result.cycle_time is not None and result.cycle_time > 0, (
            f"build_rust_evaluate_fn: evaluate_kernel returned "
            f"cycle_time={result.cycle_time!r} for kernel={kernel_name} "
            f"preset={preset}; expected a positive cycle count."
        )
        return int(result.cycle_time), dur_ms

    return evaluate


def build_real_agent_fn(*, llm_config: dict):
    """Build a per-node ``AgentFn`` factory.

    Returns ``Callable[[system_prompt], AgentFn]``. The autotune2
    search driver calls this once per node (``autotune(...,
    agent_factory=...)``) so each ``search_leaf`` / ``search_parent``
    gets an ``AgentFn`` whose system prompt is baked in at construction.

    The returned ``AgentFn`` has the conversational signature
    ``async (conversation) -> AgentResponse`` where ``conversation`` is
    the standard OpenAI-style ``[{"role": ..., "content": ...}]`` list.
    The response carries the assistant text, any reasoning-summary
    chunks the provider returned, and the ``Usage`` object — the search
    loop logs these into ``reasoning.txt`` / ``tokens.json`` per turn,
    matching pass-1's turn-dir layout.
    """
    from agents import ReasoningItem, Runner

    from src.agents import build_dynamic_run_config, make_autotune2_agent
    from src.autotune2.search import AgentResponse

    def factory(system_prompt: str):
        agent = make_autotune2_agent(llm_config, system_prompt)

        async def call(conversation: list) -> AgentResponse:
            result = await Runner.run(
                agent, conversation,
                run_config=build_dynamic_run_config(agent, conversation))
            reasoning_chunks: list[str] = []
            for item in result.new_items:
                if isinstance(item, ReasoningItem):
                    for summary in item.raw_item.summary:
                        reasoning_chunks.append(summary.text)
            return AgentResponse(
                text=result.final_output or "",
                reasoning="\n\n".join(reasoning_chunks),
                usage=result.context_wrapper.usage,
            )

        return call

    return factory


def _extract_node_def_block(composed_source: str, node_name: str) -> str:
    """Pull the ``def <node_name>(...)`` block out of a composed source.

    The composed source for a non-root node typically contains the
    autotuner-generated ``tiled_reference`` wrapper plus the LLM-emitted
    leaf def plus zero or more descendant defs. Compliance / judge gates
    must inspect *only* the LLM's code (the wrapper is autotuner-
    generated; descendants were verified in a prior pass), so we re-emit
    just the matching top-level def.
    """
    import ast as _ast
    tree = _ast.parse(composed_source)
    for fn in tree.body:
        if isinstance(fn, _ast.FunctionDef) and fn.name == node_name:
            return _ast.unparse(fn) + "\n"
    raise AssertionError(
        f"_extract_node_def_block: def {node_name}(...) not found in composed "
        f"source (top-level defs: "
        f"{[fn.name for fn in tree.body if isinstance(fn, _ast.FunctionDef)]!r})"
    )


def build_real_verifier_factory_fn(
    *,
    root_kernel: str,
    dims: dict,
    check_order: str = "correctness-first",
    judge_agent=None,
    compliance_override=None,
    extra_required_ops: tuple = (),
    log: Callable[[str], None] = print,
):
    """Per-node verifier factory.

    Returns ``make_verifier(node, parent_contract, node_tensors) ->
    VerifierFn``. The driver invokes it once per node so each
    ``search_leaf`` / ``search_parent`` receives a verifier that is
    scoped to *that* node's gold / call-args / tensors.

    Per-node behaviour:

      * **Root node** (``parent_contract is None``): full 4-gate cascade
        against ``compute_gold(root_kernel, dims, tensors)`` via
        ``_run_dsl_correctness`` with the canonical
        ``tiled_reference(dims, tensors)`` entry point. ``tensors`` here
        is the kernel-level ``root_tensors`` dict the driver passes in
        via ``node_tensors``. Numerical correctness is checked end-to-
        end exactly once at the root.

      * **Non-root node**: smoke + value check + cycles/on-chip
        evaluation. The LLM proposes ``parent_input_contracts`` per node
        (boundary contracts the autotuner-generated wrapper consumes
        when emitting offchip_loads), but the leaf body must still
        reproduce ``parent_contract.tiled_outputs`` for
        ``parent_contract.tiled_values`` — that's the gold the parent's
        pass-1 stub captured at exactly this call site, so it's the
        layout-independent ground truth for the leaf's math. Output
        contracts are NOT declared by the LLM — they're a function of
        what the DSL produces and so are derived from the built graph
        on success. The verifier:
          1. extracts the LLM-emitted ``def <node_name>(...)`` block
             from the composed source and runs compliance / judge on
             it with ``is_root=False``;
          2. runs a graph-build smoke test (``dsl_to_step.translate``
             + ``_exec_build_graph``) on the full composed source so
             that contract/leaf shape clashes (e.g. STeP frontend
             ``stride x out_shape_tiled exceeds buffer grid``
             assertions) become LLM feedback instead of crashing
             ``score_fn``;
          3. on smoke-test pass, runs a value check
             (``_value_check_nonroot``) that calls the leaf with
             ``parent_contract.tiled_values`` as positional args and
             compares the result element-wise to
             ``parent_contract.tiled_outputs``. This catches body math
             bugs (e.g. a chunked ``offchip_load_ref`` whose stride
             indexes the wrong row of a weight matrix) that the smoke
             tests pass over because they only check that the ops run,
             not what they compute;
          4. on full success, derives per-output TensorContracts from
             the built graph's OffChipStore nodes and surfaces them via
             ``VerifyResult.derived_output_contracts``.
        Variants that pass all three gates are admitted to the Pareto
        library on ``(cycles, on_chip)``.

    ``check_order`` matches the existing 3 modes. For non-root nodes
    the "correctness" slot in ``gate_order`` is filled by the graph-
    build smoke test and the "post_validator" slot is a no-op.
    """
    assert check_order in ("correctness-first", "compliance-first", "always-both"), (
        f"build_real_verifier_factory_fn: unknown check_order {check_order!r}; "
        f"expected one of correctness-first/compliance-first/always-both"
    )

    def make_verifier(node, parent_contract, node_tensors):
        if parent_contract is None:
            return _build_root_verifier(
                kernel_name=root_kernel,
                dims=dims,
                tensors=node_tensors,
                check_order=check_order,
                judge_agent=judge_agent,
                compliance_override=compliance_override,
                extra_required_ops=extra_required_ops,
                log=log,
            )
        return _build_non_root_verifier(
            root_kernel=root_kernel,
            node_path=node.path,
            node_name=node.name,
            parent_contract=parent_contract,
            tensors=node_tensors,
            dims=dims,
            check_order=check_order,
            judge_agent=judge_agent,
            compliance_override=compliance_override,
            extra_required_ops=extra_required_ops,
            log=log,
        )

    return make_verifier


def _build_root_verifier(
    *,
    kernel_name: str,
    dims: dict,
    tensors: dict,
    check_order: str,
    judge_agent,
    compliance_override,
    extra_required_ops: tuple,
    log: Callable[[str], None],
):
    """Root verifier: full 4-gate cascade against the kernel's compute_gold.

    Carved out of the original ``build_real_verifier_fn`` body — the
    behavior is unchanged for the root case, only the call surface
    moves under the per-node factory.
    """
    import tempfile
    from pathlib import Path as _Path

    from src.autotune2.search import VerifyResult
    from src.orchestrator import (
        _GateResult,
        _gate_compliance,
        _gate_correctness,
        _gate_judge,
        _gate_post_validator,
        _make_translation_post_validator,
    )

    post_validator = _make_translation_post_validator(
        kernel_name=kernel_name, dims=dims, tensors=tensors, log=log,
    )
    gate_order, break_on_fail = _gate_order_for(check_order)

    async def verify(composed_source: str) -> VerifyResult:
        scratch = _Path(tempfile.mkdtemp(prefix="autotune2_verify_root_"))
        feedbacks: list[str] = []
        correctness_verified = check_order != "compliance-first"
        compliance_invoked_judge = False

        for gate_name in gate_order:
            if gate_name == "correctness":
                res, _trace = await _gate_correctness(
                    composed_source, kernel_name, dims, tensors, "dsl",
                    scratch, log, entry_point="tiled_reference",
                )
                correctness_verified = (res.feedback is None)
            elif gate_name == "compliance":
                inline_judge_eligible = judge_agent is not None
                res = await _gate_compliance(
                    composed_source, "refactor_final", compliance_override,
                    judge_agent, tensors, scratch, log,
                    correctness_verified=correctness_verified,
                    is_root=True,
                    extra_required_ops=extra_required_ops,
                )
                if res.feedback is not None and inline_judge_eligible:
                    compliance_invoked_judge = True
            elif gate_name == "judge":
                if compliance_invoked_judge or judge_agent is None:
                    res = _GateResult(None, "PASS", 0)
                else:
                    res = await _gate_judge(
                        judge_agent, composed_source, tensors, scratch, log,
                        correctness_verified=correctness_verified,
                    )
            else:  # post_validator
                if not correctness_verified:
                    res = _GateResult(None, "PASS", 0)
                else:
                    res = _gate_post_validator(
                        post_validator, composed_source, scratch, log,
                    )
            if res.feedback is not None:
                feedbacks.append(res.feedback)
                if break_on_fail:
                    break

        if not feedbacks:
            return VerifyResult(passed=True, feedback="")
        return VerifyResult(
            passed=False, feedback="\n\n---\n\n".join(feedbacks),
        )

    return verify


def _build_non_root_verifier(
    *,
    root_kernel: str,
    node_path: str,
    node_name: str,
    parent_contract,
    tensors: dict,
    dims: dict,
    check_order: str,
    judge_agent,
    compliance_override,
    extra_required_ops: tuple,
    log: Callable[[str], None],
):
    """Non-root verifier — smoke + value-check + compliance + output-
    contract derivation.

    Autotune2's bottom-up tree DP evaluates non-root nodes for
    ``(cycles, on_chip)`` and for value correctness against the parent's
    recorded ``tiled_outputs``. The LLM is free to propose new
    ``parent_input_contracts`` (boundary contracts the wrapper consumes
    when emitting offchip_loads), but the leaf body must reproduce the
    pass-1 reference output for the pass-1 input values. Output
    contracts are derived from the built graph after the smoke tests
    pass (the wrapper's OffChipStore nodes carry the actual produced
    stream+tile shapes) and surfaced via
    ``VerifyResult.derived_output_contracts``.

    Concretely, the verifier:

      * extracts the LLM-emitted leaf def from the composed source
        (the wrapper + leaf concatenation produced by
        ``build_synthetic_wrapper_for_node`` + the LLM's ``parsed.dsl``)
        and runs the ``refactor_final`` regex compliance rules against
        it with ``is_root=False``. The wrapper's offchip_stores live
        outside this block so the single-store invariant doesn't fire;
      * runs the LLM judge against the same leaf block (when
        configured);
      * runs a *DSL eager-exec smoke test* — ``_exec_dsl_ref(composed,
        dims, tensors)`` — which catches torch-level runtime errors
        inside DSL ops (shape mismatches, dtype clashes) that the
        graph build would miss because they only surface once
        concrete tensors flow through user code. Same failure-as-
        feedback pattern pass1's ``_gate_correctness`` uses;
      * runs a *graph-build smoke test* on the full composed source —
        ``dsl_to_step.translate`` + ``_exec_build_graph(translated,
        dims, tensors)`` — and surfaces any STeP frontend assertion
        (stride mismatches, buffer-grid overflows, etc.) as actionable
        LLM feedback rather than letting it crash the score path. A
        clean build means ``score_fn`` can run ``analyze_timing``;
      * runs a *value check* against ``parent_contract.tiled_outputs``
        via ``_value_check_nonroot`` — invokes ``<node_name>(
        *tiled_values, out_shapes=...)`` directly and compares element-
        wise to the parent's recorded gold via the existing tuple-aware
        ``_gate_correctness`` / ``_compare_against_gold`` machinery.
        This catches math bugs in the body whose ops all run cleanly
        but produce wrong values (e.g. a chunked ``offchip_load_ref``
        whose ``stride`` / ``out_shape_tiled`` indexes the wrong tile
        per stream position — see the moe_dispatch leaf in
        checkpoints/2026-05-18-033705/.../moe_dispatch__root_moe_moe_dispatch).
        Without this gate such bugs only surface at the root composition,
        by which point the autotuner has already committed the variant
        to the Pareto library — leaving the root LLM no recovery path
        when budget rules out the non-buggy alternatives.
    """
    import tempfile
    import traceback as _tb
    from pathlib import Path as _Path

    from src.autotune2.search import VerifyResult
    from src.dsl_to_step import translate as _dsl_to_step_translate
    from src.orchestrator import (
        _GateResult,
        _gate_compliance,
        _gate_judge,
    )
    from src.tools import _exec_build_graph, _exec_dsl_ref

    gate_order, break_on_fail = _gate_order_for(check_order)

    async def verify(composed_source: str) -> VerifyResult:
        scratch = _Path(tempfile.mkdtemp(prefix="autotune2_verify_nonroot_"))
        feedbacks: list[str] = []
        compliance_invoked_judge = False
        graph_build_passed = False
        # Only the LLM-emitted leaf def is subject to compliance / judge —
        # the autotuner-generated wrapper has its own offchip_load /
        # offchip_store calls that would trip ``is_root=False``.
        leaf_def_block = _extract_node_def_block(composed_source, node_name)

        for gate_name in gate_order:
            if gate_name == "correctness":
                # Three-step correctness: DSL eager exec, then IR graph
                # build, then value compare against the parent's recorded
                # ``tiled_outputs``. They catch disjoint failure modes:
                # eager-exec catches torch-level runtime errors inside DSL
                # ops; graph-build catches STeP frontend assertions; value-
                # check catches math bugs in the LLM-emitted body whose
                # ops all run cleanly but produce wrong values (e.g. a
                # chunked ``offchip_load_ref`` whose ``stride`` /
                # ``out_shape_tiled`` doesn't index the right tile per
                # stream position). The value check mirrors pass1's
                # non-root correctness gate — see ``_value_check_nonroot``
                # — and admits the variant to the Pareto library only if
                # all three pass. Output contracts are derived from the
                # built graph (see ``_derive_output_contracts_from_graph``)
                # after this gate passes.
                res = _dsl_exec_smoke_test(
                    composed_source, dims, tensors,
                    exec_dsl_ref=_exec_dsl_ref,
                    traceback_mod=_tb,
                    scratch=scratch,
                    log=log,
                )
                if res.feedback is None:
                    res = _graph_build_smoke_test(
                        composed_source, dims, tensors,
                        translate_fn=_dsl_to_step_translate,
                        exec_build_graph=_exec_build_graph,
                        traceback_mod=_tb,
                        scratch=scratch,
                        log=log,
                    )
                graph_build_passed = (res.feedback is None)
                if graph_build_passed:
                    res = await _value_check_nonroot(
                        composed_source,
                        root_kernel=root_kernel,
                        node_path=node_path,
                        node_name=node_name,
                        parent_contract=parent_contract,
                        dims=dims,
                        tensors=tensors,
                        scratch=scratch,
                        log=log,
                    )
            elif gate_name == "compliance":
                inline_judge_eligible = judge_agent is not None
                res = await _gate_compliance(
                    leaf_def_block, "refactor_final", compliance_override,
                    judge_agent, tensors, scratch, log,
                    # The graph-build smoke test always runs first in
                    # correctness-first / always-both, so by the time
                    # compliance fires we know whether the build path is
                    # OK. In compliance-first mode the gate runs before
                    # the build check; tell the prompt context the build
                    # has not yet been verified.
                    correctness_verified=(check_order != "compliance-first"),
                    is_root=False,
                    extra_required_ops=extra_required_ops,
                )
                if res.feedback is not None and inline_judge_eligible:
                    compliance_invoked_judge = True
            elif gate_name == "judge":
                if compliance_invoked_judge or judge_agent is None:
                    res = _GateResult(None, "PASS", 0)
                else:
                    res = await _gate_judge(
                        judge_agent, leaf_def_block, tensors, scratch, log,
                        correctness_verified=(check_order != "compliance-first"),
                    )
            else:  # post_validator — intentional no-op for non-root
                res = _GateResult(None, "PASS", 0)
            if res.feedback is not None:
                feedbacks.append(res.feedback)
                if break_on_fail:
                    break

        if not feedbacks:
            assert graph_build_passed, (
                "_build_non_root_verifier: feedbacks is empty but the graph "
                "build gate never ran or never reported success — the "
                "non-root verifier requires a built graph to derive output "
                "contracts. Check ``gate_order`` includes 'correctness'."
            )
            derived = _derive_output_contracts_from_graph(
                composed_source, dims, tensors,
                translate_fn=_dsl_to_step_translate,
                exec_build_graph=_exec_build_graph,
                log=log,
            )
            return VerifyResult(
                passed=True, feedback="",
                derived_output_contracts=derived,
            )
        return VerifyResult(
            passed=False, feedback="\n\n---\n\n".join(feedbacks),
        )

    return verify


def _dsl_exec_smoke_test(
    composed_source: str,
    dims: dict,
    tensors: dict,
    *,
    exec_dsl_ref,
    traceback_mod,
    scratch,
    log: Callable[[str], None],
):
    """Run the composed source through the DSL eager executor; return a
    ``_GateResult`` (never raises).

    Mirrors pass1's ``_gate_correctness`` failure-handling shape: any
    exception out of ``tiled_reference(dims, tensors)`` becomes
    actionable LLM feedback rather than crashing the autotune2 run.
    Catches *torch-level runtime errors* — shape mismatches in
    ``torch.stack`` / ``torch.cat``, dtype clashes, asserts inside DSL
    ops — that the graph-build smoke test misses because they only
    surface once concrete tensors flow through the DSL functions (the
    graph builder never executes user code; it just inspects DSL ops
    syntactically + builds the IR).

    Failures here would otherwise propagate all the way into
    ``score_fn → analyze_timing → execute_values`` and crash the
    timing-model functional executor (e.g. ``_exec_flat_reassemble``).
    ``_safe_score`` would catch the crash there, but by then the LLM
    feedback points at the timing model's internals rather than at the
    DSL it actually wrote. Catching the equivalent failure at the DSL
    surface gives the LLM a tighter, more actionable signal.
    """
    from src.orchestrator import _GateResult, _error_summary, _write

    log("      Running DSL eager-exec smoke test...")
    try:
        exec_dsl_ref(composed_source, dims, tensors)
    except Exception:
        err = traceback_mod.format_exc()
        _write(scratch / "dsl_exec_error.txt", err)
        log(f"      [dsl-exec] FAILED: {_error_summary(err)}")
        return _GateResult(
            feedback=(
                "## DSL eager-exec smoke test failed\n\n"
                "Running your DSL through the eager executor "
                "(``tiled_reference(dims, tensors)``) raised before any "
                "IR / timing-model work began. This usually means a "
                "torch-level shape or dtype mismatch inside one of the "
                "DSL ops you called — e.g. per-expert ``flat_reassemble`` "
                "inputs whose dynamic-dim positioning doesn't match the "
                "op's expectations, or a ``binary_*`` op whose two "
                "operands have incompatible stream shapes. Error "
                "follows:\n\n"
                "```\n" + err + "```\n\n"
                "Fix the DSL so it executes cleanly on the supplied "
                "tensors before re-submitting."
            ),
            status="DSL_EXEC_FAIL",
            tokens=0,
        )

    log("      [dsl-exec] OK")
    return _GateResult(None, "PASS", 0)


async def _value_check_nonroot(
    composed_source: str,
    root_kernel: str,
    node_path: str,
    node_name: str,
    parent_contract,
    dims: dict,
    tensors: dict,
    *,
    scratch,
    log: Callable[[str], None],
):
    """Element-wise compare the LLM's leaf def output against the parent's
    recorded ``tiled_outputs``.

    Mirrors pass1's non-root correctness gate (see
    ``_synthesize_child_pass_node_async`` at orchestrator.py:2297-2316 and
    the ``call_args`` setup at orchestrator.py:2504-2512): inject
    ``parent_contract.tiled_outputs`` as the gold for this node's
    ``synth_name``, then invoke ``<node_name>(*tiled_values,
    out_shapes=...)`` via ``_gate_correctness`` so the existing tuple-
    aware ``_compare_against_gold`` machinery does the comparison.

    Calling the leaf with positional ``tiled_values`` deliberately
    bypasses the autotuner's wrapper — the wrapper's job is to materialize
    on-chip inputs from off-chip tensors at the contract layout, which the
    smoke tests already exercise. This check is purely "does the LLM-
    emitted def compute the right values for the inputs the parent's
    pass-1 actually fed it?" Layout / wrapper-vs-body mismatches surface
    in the smoke tests; math bugs (like the moe_dispatch leaf whose
    ``offchip_load_ref`` indexed the wrong rows of ``w_gate``) only
    surface here.

    Returns a ``_GateResult`` (``feedback=None`` on PASS).
    """
    from src.gold_cache import _inject_gold
    from src.orchestrator import (
        _GateResult, _gate_correctness, _synth_kernel_name,
    )
    from src.tools import _wrap_on_chip_call_args

    if not parent_contract.tiled_outputs:
        # Un-stamped legacy contract — no gold to compare against. Treat
        # as PASS so older flows that build contracts without populating
        # tiled_outputs (e.g. test fixtures) keep working.
        log("      [value-check] skipped (parent_contract.tiled_outputs empty)")
        return _GateResult(None, "PASS", 0)
    if not parent_contract.arg_is_raw:
        # ``_wrap_on_chip_call_args`` requires arg_is_raw to be populated;
        # without it we can't safely construct the call. Pass-1 always
        # stamps it before recursing (see orchestrator.py:2499), but be
        # defensive in case autotune2 is invoked on a fixture that didn't.
        log("      [value-check] skipped (parent_contract.arg_is_raw empty)")
        return _GateResult(None, "PASS", 0)

    synth_name = _synth_kernel_name(root_kernel, node_path)
    if parent_contract.out_is_tuple:
        gold = parent_contract.tiled_outputs
    else:
        assert len(parent_contract.tiled_outputs) == 1, (
            f"_value_check_nonroot: parent_contract.out_is_tuple=False but "
            f"tiled_outputs has {len(parent_contract.tiled_outputs)} entries"
        )
        gold = parent_contract.tiled_outputs[0]
    _inject_gold(synth_name, dims, gold)

    call_args = _wrap_on_chip_call_args(
        tuple(parent_contract.tiled_values),
        parent_contract.arg_specs,
        parent_contract.arg_is_raw,
        parent_contract.tiled_shapes,
    )
    call_kwargs = {"out_shapes": parent_contract.out_shapes}

    res, _trace = await _gate_correctness(
        composed_source, synth_name, dims, tensors, "dsl",
        scratch, log,
        entry_point=node_name,
        call_args=call_args,
        call_kwargs=call_kwargs,
    )
    return res


def _graph_build_smoke_test(
    composed_source: str,
    dims: dict,
    tensors: dict,
    *,
    translate_fn,
    exec_build_graph,
    traceback_mod,
    scratch,
    log: Callable[[str], None],
):
    """Translate + build the composed source; raise nothing, return _GateResult.

    On any exception out of ``translate_fn`` or ``exec_build_graph`` —
    including STeP frontend assertions like the
    ``stride x out_shape_tiled exceeds buffer grid`` shape-incompatibility
    that surfaces when the LLM-proposed input contract clashes with the
    leaf's internal bufferize / streamify expectations — we capture the
    traceback and return it as gate feedback. The score path can then
    skip this variant cleanly rather than crashing the entire
    autotune2 run.
    """
    from src.orchestrator import _GateResult, _error_summary, _write

    log("      Running graph-build smoke test...")
    try:
        translated = translate_fn(composed_source)
    except Exception:
        err = traceback_mod.format_exc()
        _write(scratch / "translate_error.txt", err)
        log(f"      [graph-build] translation FAILED: {_error_summary(err)}")
        return _GateResult(
            feedback=(
                "## Graph-build smoke test: DSL → STeP translation failed\n\n"
                "Your DSL could not be lowered into STeP IR — the translator "
                "expects DSL primitives in their canonical, statically-"
                "analyzable forms. Error follows:\n\n"
                "```\n" + err + "```\n\n"
                "Adjust the DSL so the constraint is satisfied."
            ),
            status="GRAPH_BUILD_FAIL_TRANSLATE",
            tokens=0,
        )

    try:
        exec_build_graph(translated, dims, tensors)
    except Exception:
        err = traceback_mod.format_exc()
        _write(scratch / "build_error.txt", err)
        log(f"      [graph-build] graph build FAILED: {_error_summary(err)}")
        return _GateResult(
            feedback=(
                "## Graph-build smoke test: STeP graph build failed\n\n"
                "Your DSL translated but the resulting STeP graph could not "
                "be constructed — usually this means the input stream "
                "shapes implied by your declared `parent_input_contracts` "
                "are incompatible with the leaf body's internal bufferize "
                "/ streamify operations (the leaf body's stride math no "
                "longer fits the wrapper-supplied stream shape). Error "
                "follows:\n\n"
                "```\n" + err + "```\n\n"
                "Either change `parent_input_contracts` so they match the "
                "leaf's internal shape assumptions, or rewrite the leaf "
                "body's internal ops to consume the new contract layout."
            ),
            status="GRAPH_BUILD_FAIL_EXEC",
            tokens=0,
        )

    log("      [graph-build] OK")
    return _GateResult(None, "PASS", 0)


def _derive_output_contracts_from_graph(
    composed_source: str,
    dims: dict,
    tensors: dict,
    *,
    translate_fn,
    exec_build_graph,
    log: Callable[[str], None],
):
    """Walk the built graph's ``OffChipStore`` nodes and return one
    identity-permutation ``TensorContract`` per output, keyed
    ``out_0``, ``out_1``, ....

    The synthetic wrapper emits ``offchip_store(promote_outer(out_i))``
    for every output, in declared order. Each store's predecessor
    ``PromoteOuter`` wraps the leaf's actual produced stream; we read
    ``(stream.shape, stream.stream_dtype.shape)`` and concatenate them
    (stream dims followed by the 2D tile) to obtain the contract's
    ``reshape``. ``permutation`` is identity — there is nothing to
    permute since the dims come straight off the live graph.

    The caller has already passed ``_graph_build_smoke_test``, so the
    rebuild here can only fail if something non-deterministic snuck in;
    we assert and let the failure surface as a hard crash if so.
    """
    from src.autotune2.contracts import TensorContract
    from step_py.ops import OffChipStore, PromoteOuter, get_stream

    log("      Deriving output contracts from built graph...")
    translated = translate_fn(composed_source)
    graph, _ = exec_build_graph(translated, dims, tensors)

    stores = sorted(
        (n for n in graph.nodes if isinstance(n, OffChipStore)),
        key=lambda n: n.instance_id,
    )
    contracts: dict[str, "TensorContract"] = {}
    for i, store in enumerate(stores):
        promote = store.input
        if isinstance(promote, tuple):
            promote = promote[0]
        assert isinstance(promote, PromoteOuter), (
            f"_derive_output_contracts_from_graph: OffChipStore {i}'s "
            f"predecessor is {type(promote).__name__}, expected PromoteOuter "
            f"(the wrapper emits 'offchip_store(promote_outer(out_i))' for "
            f"every output)."
        )
        leaf_stream = get_stream(promote.input)
        stream_shape = tuple(leaf_stream.shape)
        tile_shape = tuple(leaf_stream.stream_dtype.shape)
        combined = stream_shape + tile_shape
        contracts[f"out_{i}"] = TensorContract(
            reshape=combined,
            permutation=tuple(range(len(combined))),
        )

    log(f"      [derive] {len(contracts)} output contract(s) derived")
    return contracts


def _gate_order_for(check_order: str):
    """Resolve the gate execution order + break-on-fail policy."""
    if check_order == "correctness-first":
        return ("correctness", "compliance", "judge", "post_validator"), True
    if check_order == "compliance-first":
        return ("compliance", "judge", "correctness", "post_validator"), True
    return ("correctness", "compliance", "judge", "post_validator"), False
