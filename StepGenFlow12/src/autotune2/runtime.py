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

returning ``(cycles, dur_ms)``. Production callers build it as a
closure over the per-kernel ``work_dir`` and ``hbm_config`` / ``sim_config``
used by ``StepDB/evaluate.py``; unit tests inject deterministic fakes.
"""

from __future__ import annotations

import json
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


# (composed_source) -> (cycles, dur_ms)
RustEvaluateFn = Callable[[str], tuple[int, float]]


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

    Results are returned sorted by rust_cycles (best first). The list
    length equals ``len(pick_top_k_pareto_entries(root_library, k))``;
    a rust failure is **not** silently skipped — the rust evaluator
    must succeed for every promoted entry (otherwise the calling
    pipeline has a bug worth surfacing).
    """
    picks = pick_top_k_pareto_entries(root_library, k)
    out: list[RustPromotionResult] = []
    for entry in picks:
        composed = _build_composed_source_for_entry(entry)
        cycles, dur_ms = rust_evaluate_fn(composed)
        out.append(RustPromotionResult(
            entry=entry,
            rust_cycles=cycles,
            rust_dur_ms=dur_ms,
            composed_source=composed,
        ))
    out.sort(key=lambda r: r.rust_cycles)
    return out


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------


def write_autotune2_summary(
    *,
    autotune_result: AutotuneResult,
    rust_promotions: list[RustPromotionResult],
    out_path: Path,
    include_sources: bool = False,
) -> None:
    """Emit a JSON summary of an autotune2 run.

    Fields:
      - ``root_path``: the root node's path
      - ``library_sizes``: ``{node_path: <num cells>}``
      - ``root_pareto``: the root's analytical Pareto entries as
        ``[{cycles, on_chip, provenance}, ...]``, sorted by cycles
      - ``rust_winners``: the rust-promoted entries with their
        analytical-vs-rust cycle comparison
      - ``best_rust_entry``: the lowest rust_cycles entry (or None if
        no promotions)
      - ``best_composed_source``: only when ``include_sources=True``;
        the source string of the rust-best entry (can be megabytes for
        large kernels)
    """
    library_sizes = {
        path: sum(len(by_out) for by_out in lib.values())
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
        "library_sizes": library_sizes,
        "root_pareto": root_pareto,
        "rust_winners": rust_winners,
        "best_rust_entry": rust_winners[0] if rust_winners else None,
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
    timing_only: bool = True,
) -> RustEvaluateFn:
    """Build a rust evaluator that writes a temp DSL file and invokes
    ``StepDB/evaluate.py``'s rust simulator subprocess.

    Lazy-imports the StepDB module so test environments without the rust
    toolchain can still import autotune2.runtime. The returned closure
    writes the composed source to ``work_dir / "step_impl.py"`` before
    each call (overwrites prior contents). ``timing_only=True`` skips
    correctness comparison (faster; matches the autotuner's analytical
    role).
    """

    def evaluate(composed_source: str) -> tuple[int, float]:
        from StepDB.evaluate import evaluate_kernel  # type: ignore

        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "step_impl.py").write_text(composed_source)
        result = evaluate_kernel(
            kernel_name=kernel_name,
            preset=preset,
            work_dir=str(work_dir),
            timing_only=timing_only,
        )
        # EvalResult exposes .cycles and .duration_ms; fall back to the
        # raw run_graph tuple if the fields differ in this StepDB build.
        cycles = int(getattr(result, "cycles", 0))
        dur_ms = float(getattr(result, "duration_ms", 0.0))
        assert cycles > 0, (
            f"build_rust_evaluate_fn: rust evaluator returned cycles={cycles} "
            f"for kernel={kernel_name} preset={preset}; expected positive "
            f"cycle count. EvalResult={result!r}"
        )
        return cycles, dur_ms

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

    from src.agents import make_autotune2_agent
    from src.autotune2.search import AgentResponse

    def factory(system_prompt: str):
        agent = make_autotune2_agent(llm_config, system_prompt)

        async def call(conversation: list) -> AgentResponse:
            result = await Runner.run(agent, conversation)
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


def build_real_verifier_fn(
    *,
    kernel_name: str,
    dims: dict,
    tensors: dict,
    check_order: str = "correctness-first",
    judge_agent=None,
    compliance_override=None,
    extra_required_ops: tuple = (),
    log: Callable[[str], None] = print,
):
    """Wire pass-1's 4-gate verification cascade as a ``VerifierFn``.

    Returns ``async (composed_source) -> VerifyResult``. Each call runs
    the configured gate cascade on the composed source (parent DSL plus
    descendant DSLs concatenated post-order, terminating in
    ``tiled_reference``):

      1. ``_gate_correctness`` — exec under the ``"dsl"`` executor and
         compare to gold (``compute_gold(kernel_name, dims, tensors)``).
      2. ``_gate_compliance`` — refactor_final regex compliance over
         the composed source as a whole (``is_root=True`` since the
         composed source always ends in ``tiled_reference``).
      3. ``_gate_judge`` — LLM judge for structural feedback. Skipped
         when ``judge_agent`` is ``None`` or already invoked inline by
         gate 2.
      4. ``_gate_post_validator`` — deterministic translator round-trip
         (DSL → STeP → graph executor). Skipped when correctness fails
         (matches pass-1's cascade).

    ``check_order`` mirrors pass-1's three modes:

      - ``"correctness-first"`` (default): break on first failing gate.
      - ``"compliance-first"``: run compliance + judge before correctness;
        break on first fail.
      - ``"always-both"``: run every gate regardless and concatenate
        feedback. Higher cost per failed turn; tighter LLM signal.

    Per-gate artifacts (correctness_result.txt, translate_check/, etc.)
    are written to an ephemeral ``tempfile.mkdtemp()`` dir per call; the
    search loop's own per-turn checkpoint at
    ``ckpt_dir/attempt_<N>/turn_<M>/`` carries the user/response/status
    log for inspection.
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

    assert check_order in ("correctness-first", "compliance-first", "always-both"), (
        f"build_real_verifier_fn: unknown check_order {check_order!r}; "
        f"expected one of correctness-first/compliance-first/always-both"
    )

    post_validator = _make_translation_post_validator(
        kernel_name=kernel_name, dims=dims, tensors=tensors, log=log,
    )

    if check_order == "correctness-first":
        gate_order = ("correctness", "compliance", "judge", "post_validator")
        break_on_fail = True
    elif check_order == "compliance-first":
        gate_order = ("compliance", "judge", "correctness", "post_validator")
        break_on_fail = True
    else:  # always-both
        gate_order = ("correctness", "compliance", "judge", "post_validator")
        break_on_fail = False

    async def verify(composed_source: str) -> VerifyResult:
        scratch = _Path(tempfile.mkdtemp(prefix="autotune2_verify_"))
        feedbacks: list[str] = []
        # Default to True in correctness-first / always-both so compliance
        # gate prompts the LLM appropriately when run before correctness.
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
