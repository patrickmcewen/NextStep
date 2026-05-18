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
    max_total_compute_bw: int | None = None,
) -> RustEvaluateFn:
    """Build a rust evaluator that writes a temp DSL file and invokes
    ``StepDB/evaluate.py``'s rust simulator subprocess.

    Lazy-imports the StepDB module so test environments without the rust
    toolchain can still import autotune2.runtime. The returned closure
    writes the composed source to ``work_dir / "step_impl.py"`` before
    each call (overwrites prior contents). ``timing_only=True`` skips
    correctness comparison (faster; matches the autotuner's analytical
    role).

    ``max_total_compute_bw`` is forwarded to ``evaluate_kernel`` so each
    compute op's ``compute_bw`` is rescaled (in place) to sum to that
    budget before serialization — mirrors what the analytical scorer
    does via ``compose._rescale_compute_bw``. Pass the same value here
    that ``make_analytical_scorer`` got, or the analytical-vs-rust
    cycle comparison in the summary uses two different cost models.
    """

    def evaluate(composed_source: str) -> tuple[int, float]:
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

        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "step_impl.py").write_text(step_source)
        t0 = time.perf_counter()
        result = evaluate_kernel(
            kernel_name=kernel_name,
            preset=preset,
            work_dir=str(work_dir),
            timing_only=timing_only,
            step_impl_source=step_source,
            max_total_compute_bw=max_total_compute_bw,
        )
        dur_ms = (time.perf_counter() - t0) * 1000.0

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

      * **Non-root node**: cycles + on-chip-memory evaluation only.
        The LLM is allowed to propose new ``parent_input_contracts`` /
        ``parent_output_contracts`` per node, which decouples the
        node-local notion of a "correct" output from pass-1's recorded
        ``tiled_outputs`` — so we *don't* compare against those.
        Instead the verifier:
          1. extracts the LLM-emitted ``def <node_name>(...)`` block
             from the composed source and runs compliance / judge on
             it with ``is_root=False``;
          2. runs a graph-build smoke test (``dsl_to_step.translate``
             + ``_exec_build_graph``) on the full composed source so
             that contract/leaf shape clashes (e.g. STeP frontend
             ``stride x out_shape_tiled exceeds buffer grid``
             assertions) become LLM feedback instead of crashing
             ``score_fn``.
        Variants that pass the smoke test are admitted to the
        Pareto library on ``(cycles, on_chip)`` alone; functional
        correctness is recovered at composition time (the parent's /
        root's verifier, which sees the full chain, catches
        mismatches).

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

    async def verify(
        composed_source: str,
        output_contracts: dict | None = None,
    ) -> VerifyResult:
        del output_contracts  # root verifies against compute_gold directly
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
    """Non-root verifier — graph-build smoke + compliance, no gold compare.

    Autotune2's bottom-up tree DP evaluates non-root nodes for
    ``(cycles, on_chip)`` only — the LLM is free to propose new
    ``parent_input_contracts`` / ``parent_output_contracts`` so the
    notion of a "correct" sub-output is decoupled from pass-1's
    recorded ``tiled_outputs``. Numerical correctness is recovered at
    the parent / root composition step, where ``compute_gold`` checks
    the full kernel against the LLM-picked leaf variants.

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
        the value-correctness of the result is *not* checked here.

    No gold injection. No call_args / call_kwargs entry-point
    invocation. ``parent_contract`` is retained on the signature for
    API symmetry with future extensions but is otherwise unused.
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

    del root_kernel, node_path, parent_contract  # unused; see docstring

    gate_order, break_on_fail = _gate_order_for(check_order)

    async def verify(
        composed_source: str,
        output_contracts: dict | None = None,
    ) -> VerifyResult:
        scratch = _Path(tempfile.mkdtemp(prefix="autotune2_verify_nonroot_"))
        feedbacks: list[str] = []
        compliance_invoked_judge = False
        # Only the LLM-emitted leaf def is subject to compliance / judge —
        # the autotuner-generated wrapper has its own offchip_load /
        # offchip_store calls that would trip ``is_root=False``.
        leaf_def_block = _extract_node_def_block(composed_source, node_name)

        for gate_name in gate_order:
            if gate_name == "correctness":
                # Three-step correctness: DSL eager exec, IR graph build,
                # contract conformance. They catch disjoint failure
                # modes — torch-level runtime errors, STeP frontend
                # assertions, and stream/tile-vs-declared-contract
                # mismatches respectively — so subsequent checks still
                # run after each earlier one passes. The conformance
                # check is what catches the variant whose declared
                # ``reshape`` claims one layout while the DSL actually
                # produces another (e.g. a leading-singleton stream that
                # later trips ``Parallelize``'s ``shape[0] % n_consumers``
                # assertion in the parent composition).
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
                if res.feedback is None and output_contracts:
                    res = _contract_conformance_smoke_test(
                        composed_source, dims, tensors, output_contracts,
                        translate_fn=_dsl_to_step_translate,
                        exec_build_graph=_exec_build_graph,
                        traceback_mod=_tb,
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
            return VerifyResult(passed=True, feedback="")
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
                "be constructed — usually this means the input or output "
                "stream shapes implied by your declared "
                "`parent_input_contracts` / `parent_output_contracts` are "
                "incompatible with the leaf body's internal bufferize / "
                "streamify operations (the leaf body's stride math no "
                "longer fits the wrapper-supplied stream shape). Error "
                "follows:\n\n"
                "```\n" + err + "```\n\n"
                "Either change the contracts so they match the leaf's "
                "internal shape assumptions, or rewrite the leaf body's "
                "internal ops to consume the new contract layout."
            ),
            status="GRAPH_BUILD_FAIL_EXEC",
            tokens=0,
        )

    log("      [graph-build] OK")
    return _GateResult(None, "PASS", 0)


def _contract_conformance_smoke_test(
    composed_source: str,
    dims: dict,
    tensors: dict,
    output_contracts: dict,
    *,
    translate_fn,
    exec_build_graph,
    traceback_mod,
    scratch,
    log: Callable[[str], None],
):
    """Verify each declared output contract matches the built graph's
    actual stream+tile decomposition; return a ``_GateResult``.

    Background: ``TensorContract`` is just (reshape, permutation) over
    the vanilla shape and does not encode the STeP IR stream/tile split.
    Two leaves can both declare ``reshape=(64, 16, 32)`` while producing
    incompatible underlying layouts — e.g. ``stream=(64,) tile=(16,32)``
    vs ``stream=(1, 1024) tile=(1, 32)``. A downstream consumer that's
    authored against one layout will fail when fed the other (the most
    common failure is ``Parallelize``'s ``shape[0] % num_consumers``
    check tripping on a leading singleton; the variant is "honest" by
    vanilla shape but lies about the actual layout).

    This check enforces the convention that *the concatenation of the
    output stream shape and the output tile shape, after permutation,
    must equal* ``output_contracts[out_i].post_permute_shape()``.
    STeP tiles are always 2D, so the contract's last two reshape dims
    are interpreted as the tile shape and the prefix as the stream
    shape. Variants that violate this fail the gate with feedback
    telling the LLM the actual produced layout vs what it declared.

    Only outputs are checked — wrapper-constructed inputs come from
    ``offchip_load`` with shapes derived directly from the declared
    input_contracts, so they are constrained by construction.
    """
    from src.orchestrator import _GateResult, _error_summary, _write
    from step_py.ops import OffChipStore, PromoteOuter, get_stream

    log("      Running contract conformance smoke test...")
    try:
        translated = translate_fn(composed_source)
        graph, _ = exec_build_graph(translated, dims, tensors)
    except Exception:
        # The graph-build smoke test should have caught this; if we hit
        # it again here, surface as conformance failure with the trace.
        err = traceback_mod.format_exc()
        _write(scratch / "conformance_build_error.txt", err)
        log(f"      [conformance] re-build FAILED: {_error_summary(err)}")
        return _GateResult(
            feedback=(
                "## Contract conformance: graph re-build failed\n\n"
                "Unexpected — the graph-build smoke test passed but "
                "rebuilding for conformance inspection raised. Error:\n\n"
                "```\n" + err + "```"
            ),
            status="CONTRACT_CONFORMANCE_BUILD_FAIL",
            tokens=0,
        )

    stores = sorted(
        (n for n in graph.nodes if isinstance(n, OffChipStore)),
        key=lambda n: n.instance_id,
    )
    assert len(stores) == len(output_contracts), (
        f"_contract_conformance_smoke_test: expected {len(output_contracts)} "
        f"OffChipStore nodes (one per declared output), found {len(stores)}. "
        f"The synthetic wrapper should emit exactly one store per output."
    )

    mismatches: list[tuple[str, tuple, tuple, tuple, tuple]] = []
    for i, store in enumerate(stores):
        promote = store.input
        if isinstance(promote, tuple):
            promote = promote[0]
        assert isinstance(promote, PromoteOuter), (
            f"_contract_conformance_smoke_test: OffChipStore {i}'s predecessor "
            f"is {type(promote).__name__}, expected PromoteOuter (the wrapper "
            f"emits 'offchip_store(promote_outer(out_i))' for every output)."
        )
        leaf_stream = get_stream(promote.input)
        actual_stream = tuple(leaf_stream.shape)
        actual_tile = tuple(leaf_stream.stream_dtype.shape)
        actual_combined = actual_stream + actual_tile

        out_name = f"out_{i}"
        contract = output_contracts[out_name]
        expected_combined = contract.post_permute_shape()

        if actual_combined != expected_combined:
            mismatches.append((
                out_name, expected_combined, actual_combined,
                actual_stream, actual_tile,
            ))

    if mismatches:
        lines = [
            "## Contract conformance check failed",
            "",
            "Your declared `parent_output_contracts` do not match the actual "
            "stream + tile shape your DSL produces for one or more outputs. "
            "A `TensorContract`'s `reshape` (after applying `permutation`) "
            "must equal the concatenation of the output's stream shape and "
            "its tile shape — in that order. STeP tiles are 2D, so the "
            "last two dims of `reshape` are interpreted as the tile and the "
            "prefix as the stream.",
            "",
            "Why this matters: a downstream consumer that's authored against "
            "the layout you advertised will rely on `shape[0]`-style checks "
            "(e.g. `Parallelize`'s divisibility assertion) that silently "
            "fail when the upstream produces a different stream-vs-tile "
            "split — even if total element count matches.",
            "",
            "Mismatches:",
        ]
        for name, expected, actual, stream, tile in mismatches:
            lines.append(
                f"- `{name}`: declared `reshape={expected!r}` (post-permutation), "
                f"actual stream+tile = `{actual!r}` "
                f"(stream={stream!r}, tile={tile!r})"
            )
        lines.extend([
            "",
            "Fix by either (a) changing your DSL so the produced stream and "
            "tile decomposition matches the declared contract, or (b) "
            "changing the declared `reshape` to truthfully describe what your "
            "DSL emits. Note: downstream nodes will see your declared layout "
            "and break if it lies.",
        ])
        feedback = "\n".join(lines) + "\n"
        _write(scratch / "conformance_mismatch.txt", feedback)
        names = ",".join(m[0] for m in mismatches)
        log(f"      [conformance] FAILED for outputs: {names}")
        return _GateResult(
            feedback=feedback,
            status="CONTRACT_NONCONFORMANCE",
            tokens=0,
        )

    log("      [conformance] OK")
    return _GateResult(None, "PASS", 0)


def _gate_order_for(check_order: str):
    """Resolve the gate execution order + break-on-fail policy."""
    if check_order == "correctness-first":
        return ("correctness", "compliance", "judge", "post_validator"), True
    if check_order == "compliance-first":
        return ("compliance", "judge", "correctness", "post_validator"), True
    return ("correctness", "compliance", "judge", "post_validator"), False
