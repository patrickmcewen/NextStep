"""Orchestrator for StepGenFlow — per-kernel two-phase pipeline.

Phase 1 (lowering): a single ``refactor_final`` LLM pass rewrites the PyTorch
reference into DSL form. Each turn is gated by the ``dsl`` executor against
gold, and (under ``--translator=auto``) also by a post-validator that runs the
deterministic translator and the resulting STeP graph against gold.

Phase 2 (translation): either the deterministic AST translator
(``--translator=auto``) or an LLM ``translate`` pass (``--translator=llm``) emits
``build_graph(dims, tensors)``; the result is gated by the ``graph`` executor.

The ``direct`` and ``direct_no_functional`` pipelines skip phase 1 and produce
``build_graph`` directly from PyTorch via a single LLM pass.

Bundle mode replaces the standalone DSL surface with a bundle's abstraction
(mounted as ``step_dsl``) and the deterministic translator with the bundle's
own ``transpiler.translate``; the pipeline collapses to a single
``refactor_final`` pass plus one deterministic translate.

Checkpoint structure:
  checkpoints/<timestamp>/<kernel>/outer_<N>/<pass_name>/turn_<M>/...
  checkpoints/<timestamp>/<kernel>/result.json
"""

import ast
import asyncio
import contextlib
import dataclasses
import io
import json
import os
import re
import shutil
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import torch
import yaml

# Enable shape-trace logging in step_dsl ops before tools.py exec's the scaffold.
# Each op prints its input/output shapes; the orchestrator captures the trace
# and feeds it to the LLM alongside any error.
os.environ.setdefault("STEP_DSL_TRACE", "1")

from src.dsl_to_step import translate as _dsl_to_step_translate
from agents import Runner

from src.agents import (make_judge_agent, make_bundle_judge_agent,
                        make_pass_agent)
from src.prompts import (LOWERING_PASSES, TRANSLATOR_PASSES, PIPELINES,
                         build_pass_user_prompt,
                         _format_tensors_description,
                         resolve_few_shot_examples)
from src.tools import (_exec_build_graph, _exec_dsl_ref,
                       _validate_functional_mod, enhance_emulator_error)
from src.gold_cache import _GOLD_CACHE, _gold_key, _get_gold, _inject_gold

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_DEIO_ROOT = Path(__file__).resolve().parent.parent.parent  # DEIOpt/
_STEPDB_DIR = _DEIO_ROOT / "StepDB"
_STEP_TL_SRC = _DEIO_ROOT / "step_tl" / "src"
_STEP_TL_PROTO = _STEP_TL_SRC / "proto"

for p in (_STEPDB_DIR, _STEP_TL_SRC, _STEP_TL_PROTO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from precompute import precompute_tensors  # noqa: E402  (StepDB/precompute.py)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Extract the last python code block from LLM output.

    Accepts ```python ... ``` and bare ``` ... ``` fences. Falls back to the
    whole response if it parses as valid Python — some bundle prompts instruct
    the model to omit fences entirely, and we don't want that to trip
    NO_CODE_EXTRACTED.
    """
    import ast
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        ast.parse(stripped)
    except SyntaxError:
        return ""
    return stripped


def _reasoning_text(run_result) -> str:
    """Concatenate any reasoning summaries attached to a RunResult.

    Returns "" when the model produced no reasoning items (non-reasoning
    models, or a turn where the provider didn't return the `reasoning`
    field). The orchestrator writes this separately from `response.txt`
    so chain-of-thought never contaminates code extraction.
    """
    from agents import ReasoningItem
    chunks: list[str] = []
    for item in run_result.new_items:
        if isinstance(item, ReasoningItem):
            for summary in item.raw_item.summary:
                chunks.append(summary.text)
    return "\n\n".join(chunks)


def _error_summary(err: str) -> str:
    """Extract a meaningful one-line summary from a traceback string."""
    for line in err.splitlines():
        stripped = line.strip()
        if stripped and any(tok in stripped for tok in ("Error:", "Error(", "assert ")):
            return stripped
    return err.splitlines()[-1].strip()


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> None:
    """Write content to a file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build_success_result(iteration, total_tool_calls, inner, all_inner_results, tiled_code):
    return {
        "success": True,
        "outer_iterations": iteration + 1,
        "total_tool_calls": total_tool_calls,
        "cycle_count": inner.get("cycle_count"),
        "final_diagnosis": None,
        "tiled_code": tiled_code,
        "traces": [{"code": r.get("code"), "tool_outputs": r.get("tool_outputs", [])} for r in all_inner_results],
    }


def _load_autotune_progress(autotune_kernel_dir: Path) -> dict:
    """Read `progress.json` written by run_autotune, or return {} if absent.

    `autotune_kernel_dir` is the inner kernel-named directory created by
    run_autotune for a given pass (i.e.
    `<outer_dir>/autotune/pass_<idx>_<agent>/<kernel_name>`). When
    run_autotune raises before writing baseline progress, the file may not
    exist.
    """
    path = autotune_kernel_dir / "progress.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# Deferred to avoid circular import (src.autotune imports from src.orchestrator).
# Populated lazily on first call to _run_outer_autotune; monkeypatch can
# replace this module-level name before the helper is invoked.
run_autotune = None


async def _run_outer_autotune(*, outer_dir: Path, kernel_name: str, preset: str,
                              llm_config: dict, autotune_options: dict,
                              log, tag: str) -> dict:
    """Run a sequence of autotuner passes against an outer's verified build_graph.

    Each pass's `best.py` becomes the next pass's baseline. Halts on:
      - a pass returning `feasible=False` (chain stops, status='halted')
      - an exception inside a pass (status='error', partial best recovered
        from progress.json so the harness still surfaces work-in-progress).
    """
    global run_autotune
    if run_autotune is None:
        from src.autotune import run_autotune as _ra
        run_autotune = _ra

    autotune_root = outer_dir / "autotune"
    passes = autotune_options["passes"]
    config = autotune_options["config"]
    assert passes, "autotune_options['passes'] must be a non-empty list"

    pass_results: list[dict] = []
    halt_reason: str | None = None
    halted_pass_index: int | None = None
    error_msg: str | None = None
    resume_from = str(outer_dir)

    for idx, spec in enumerate(passes):
        agent = spec["agent"]
        max_turns = spec.get("max_turns")
        feasibility = spec.get("feasibility")
        pass_dir = autotune_root / f"pass_{idx}_{agent}"
        log(f"{tag} autotune pass {idx} ({agent}) starting; resume_from={resume_from}")
        print(f"{tag} autotune pass {idx} ({agent}) starting")

        try:
            r = await run_autotune(
                kernel_name=kernel_name,
                preset=preset,
                llm_config=llm_config,
                autotune_config=config,
                resume_from=resume_from,
                max_turns=max_turns,
                checkpoint_dir=str(pass_dir),
                agent_variant=agent,
                feasibility=feasibility,
                log_prefix=f"{tag} ",
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            msg = f"{tag} autotune FAILED at pass {idx} ({agent}): {err}"
            log(msg)
            print(msg)
            progress = _load_autotune_progress(pass_dir / kernel_name)
            # Mirrors the success-path key set so callers iterating
            # pass_results don't have to special-case error entries.
            # Unknown values are None; `feasible` is False (a crashed pass
            # is by definition not feasible).
            partial = {
                "index": idx,
                "agent": agent,
                "status": "error",
                "error": err,
                "success": False,
                "kernel": kernel_name,
                "preset": preset,
                "checkpoint_dir": str(pass_dir / kernel_name),
                "resume_from": resume_from,
                "agent_variant": agent,
                "feasibility": feasibility,
                "feasible": False,
                "baseline_feasible": None,
                "baseline_cycles": progress.get("baseline_cycles"),
                "best_cycles": progress.get("best_cycles"),
                "speedup": None,
                "baseline_on_chip_bytes": progress.get("baseline_on_chip_bytes"),
                "baseline_off_chip_bytes": progress.get("baseline_off_chip_bytes"),
                "best_on_chip_bytes": progress.get("best_on_chip_bytes"),
                "best_off_chip_bytes": progress.get("best_off_chip_bytes"),
                "turns": progress.get("turn"),
            }
            pass_results.append(partial)
            halt_reason = "error"
            halted_pass_index = idx
            error_msg = err
            break

        pass_results.append({"index": idx, "agent": agent, "status": "ok", **r})

        if not r["feasible"]:
            log(f"{tag} autotune pass {idx} infeasible; halting chain")
            print(f"{tag} autotune pass {idx} infeasible; halting chain")
            halt_reason = "infeasible"
            halted_pass_index = idx
            break

        # Feed best.py forward as next pass's baseline.
        resume_from = str(pass_dir / kernel_name / "best.py")

    # Aggregate overall: pass 0's baseline -> last completed pass's best.
    first = pass_results[0]
    last = pass_results[-1]
    overall_baseline = first.get("baseline_cycles")
    overall_best = last.get("best_cycles")
    overall_speedup = (overall_baseline / overall_best
                       if overall_baseline and overall_best else None)
    overall_feasible = last.get("feasible", False)

    status = "ok" if halt_reason is None else (
        "halted" if halt_reason == "infeasible" else "error")

    return {
        "status": status,
        "passes": pass_results,
        "halt_reason": halt_reason,
        "halted_pass_index": halted_pass_index,
        "error": error_msg,
        "checkpoint_dir": str(autotune_root),
        "overall": {
            "baseline_cycles": overall_baseline,
            "best_cycles": overall_best,
            "speedup": overall_speedup,
            "feasible": overall_feasible,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_stepdb_config() -> dict:
    config_path = _STEPDB_DIR / "bench_config.yaml"
    assert config_path.exists(), f"bench_config.yaml not found at {config_path}"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Correctness checkers for each executor type
# ---------------------------------------------------------------------------

def _overall_match(report: str) -> bool:
    """Read the overall verdict from a ``_compare_against_gold`` report.

    The first line of every report is unambiguously ``match=True`` or
    ``match=False``; per-output detail blocks below it use ``tensor=PASS``
    / ``tensor=FAIL`` so a partial per-output match never leaks the
    substring ``match=True`` into a tuple report whose overall verdict is
    False. Always use this helper at gate sites — never substring-match.
    """
    if not report:
        return False
    return report.splitlines()[0].strip() == "match=True"


def _compare_against_gold(result, kernel_name, dims, label="result"):
    """Compare a tensor (or tuple/list of tensors) result against gold.

    Tuple/list outputs are supported because the planner can decompose a
    kernel into intermediate (non-root) sub-Models whose forward returns
    multiple tensors (e.g. ``return Q, K, V``). The root tree node always
    inherits the original kernel's single-tensor signature, so root-level
    comparisons stay single-tensor; tuple comparisons only kick in for
    intermediate per-node refactor passes.

    Shape strictness is uniformly relaxed: every comparison flattens both
    tensors and checks element-wise equality with matching ``numel()``.
    The kernel's externally-observable output is a STeP tile-stream
    (rank >= 3), so a vanilla 2D gold against an equivalent stream-shaped
    DSL/sim output shares row-major flat memory and matches element-wise.
    Layout-changing bugs (transposes, scrambles) still fail because they
    produce different elements at the same flat positions.

    Return-format invariant: the first line is always ``match=True`` or
    ``match=False`` (the overall verdict). Detail follows on subsequent
    lines and uses ``tensor=PASS``/``tensor=FAIL`` for per-output blocks
    so callers reading the verdict via ``_overall_match`` cannot be
    fooled by a partial match.
    """
    import torch

    gold = _get_gold(kernel_name, dims)

    gold_is_seq = isinstance(gold, (tuple, list))
    result_is_seq = isinstance(result, (tuple, list))
    if gold_is_seq != result_is_seq:
        return (
            f"match=False\n"
            f"STRUCTURE MISMATCH: gold is {type(gold).__name__}, "
            f"{label} is {type(result).__name__}"
        )
    if gold_is_seq:
        if len(gold) != len(result):
            return (
                f"match=False\n"
                f"LENGTH MISMATCH: gold has {len(gold)} outputs, "
                f"{label} has {len(result)}"
            )
        per_out = [
            _compare_one_tensor(g, r, kernel_name, dims, f"{label}[{i}]")
            for i, (g, r) in enumerate(zip(gold, result))
        ]
        all_match = all(p["match"] for p in per_out)
        out = f"match={all_match}"
        for i, p in enumerate(per_out):
            tag = "PASS" if p["match"] else "FAIL"
            out += f"\n--- output[{i}] (tensor={tag}) ---\n{p['report']}"
        return out
    res = _compare_one_tensor(gold, result, kernel_name, dims, label)
    return f"match={res['match']}\n{res['report']}"


def _compare_one_tensor(gold, result, kernel_name, dims, label):
    """Single-tensor comparison helper. Returns ``{"match": bool, "report": str}``.

    Shape strictness is uniformly relaxed across the flow: both tensors are
    flattened in row-major order before the element-wise compare, requiring
    only that ``numel`` matches. See ``_compare_against_gold`` for the
    rationale.

    The ``report`` body intentionally contains no ``match=True``/
    ``match=False`` substring — the per-tensor verdict is carried in the
    structured ``match`` field, and the overall verdict line is emitted
    once by ``_compare_against_gold``. This prevents a partial per-output
    PASS from greenlighting a substring-based gate check on a tuple
    report whose overall verdict is FAIL.
    """
    import torch

    if not hasattr(result, "shape"):
        report = (
            f"TYPE MISMATCH: gold is torch.Tensor, {label} is "
            f"{type(result).__name__}"
        )
        return {"match": False, "report": report}

    if gold.numel() != result.numel():
        report = (
            f"NUMEL MISMATCH: gold {tuple(gold.shape)} (numel={gold.numel()}) "
            f"vs {label} {tuple(result.shape)} (numel={result.numel()})"
        )
        return {"match": False, "report": report}

    gold_cmp = gold.reshape(-1)
    result_cmp = result.reshape(-1)

    max_err = (gold_cmp - result_cmp).abs().max().item()
    rel_err = max_err / (gold_cmp.abs().max().item() + 1e-12)
    match = rel_err < 1e-5

    shape_note = (
        f"output_shape={tuple(result.shape)}"
        if gold.shape == result.shape
        else (f"output_shape={tuple(result.shape)} "
              f"(flattened for compare; gold shape {tuple(gold.shape)})")
    )
    report = (
        f"max_abs_err={max_err:.2e}\nrel_err={rel_err:.2e}\n"
        f"{shape_note}"
    )
    if not match:
        diff = (gold_cmp - result_cmp).abs()
        worst_multi = torch.unravel_index(diff.argmax(), diff.shape)
        report += (
            f"\nworst_error_index={tuple(i.item() for i in worst_multi)}"
            f"\ngold_value={gold_cmp[worst_multi].item():.6e}"
            f"\n{label}_value={result_cmp[worst_multi].item():.6e}"
        )
    return {"match": match, "report": report}


def _run_dsl_correctness(code, kernel_name, dims, tensors, *,
                          extra_globals: dict | None = None,
                          entry_point: str = "tiled_reference",
                          call_args: tuple | None = None,
                          call_kwargs: dict | None = None):
    """Run a refactor-pass candidate against gold via the DSL executor.

    The DSL surface is directly runnable (the standalone ``step_dsl`` module
    or the bundle's mounted abstraction), so we exec the candidate and
    invoke the configured entry point. By default this is
    ``tiled_reference(dims, tensors)`` (root nodes); non-root nodes pass
    ``entry_point=<node_name>`` plus the contract-derived ``call_args`` and
    ``call_kwargs`` so the LLM-emitted
    ``<node_name>(<arg_1>, ..., *, out_shapes, out_perms=None)`` is
    exercised against parent-recorded inputs.

    For non-root nodes, ``call_kwargs["out_shapes"]`` is the contract the
    parent declared when calling this child's stub. We enforce it as a
    hard shape gate *before* the value check: the DSL output must equal
    the declared shape exactly. A numel-only check (correct for root,
    where ``offchip_store`` always emits 2D against possibly-ND gold)
    silently lets through implementations whose flat elements match gold
    but whose stream rank differs from the contract — those then crash
    pass-2 the moment a downstream consumer asserts on stream rank.
    """
    result = _exec_dsl_ref(code, dims, tensors, extra_globals=extra_globals,
                           entry_point=entry_point,
                           call_args=call_args,
                           call_kwargs=call_kwargs)
    declared = (call_kwargs or {}).get("out_shapes")
    if declared is not None:
        shape_violation = _check_declared_out_shapes(result, declared)
        if shape_violation is not None:
            return f"match=False\n{shape_violation}"
    return _compare_against_gold(result, kernel_name, dims, "dsl")


def _check_declared_out_shapes(result, declared) -> str | None:
    """Return a violation report if ``result``'s shape doesn't match the
    contract's declared ``out_shapes``, else None.

    ``declared`` is the parent's ``out_shapes=`` argument — a tuple of
    shape tuples, one per output. ``result`` is a Tensor when the ref
    returns a single output, or a tuple/list when it returns multiple.
    Single-output is encoded as ``len(declared) == 1`` whether the
    underlying ref returns ``Tensor`` or ``(Tensor,)``.
    """
    declared = tuple(tuple(int(d) for d in s) for s in declared)
    result_seq = result if isinstance(result, (tuple, list)) else (result,)
    if len(result_seq) != len(declared):
        return (
            f"DECLARED-SHAPE LENGTH MISMATCH: out_shapes declares "
            f"{len(declared)} output(s), dsl returned {len(result_seq)}"
        )
    for i, (r, d) in enumerate(zip(result_seq, declared)):
        if not hasattr(r, "shape"):
            return (
                f"DECLARED-SHAPE TYPE MISMATCH: out_shapes[{i}] declared {d}, "
                f"dsl returned {type(r).__name__}"
            )
        actual = tuple(int(s) for s in r.shape)
        if actual != d:
            return (
                f"DECLARED-SHAPE MISMATCH: out_shapes[{i}] declared {d}, "
                f"dsl returned {actual}"
            )
    return None


def _run_graph_correctness(code, kernel_name, dims, tensors, *,
                            extra_globals: dict | None = None):
    """Run a translate-pass candidate against gold via the simulator.

    The candidate is exec'd as ``build_graph(dims, tensors)`` and the
    resulting graph is dispatched through the STeP simulator. Simulator
    failures are re-raised with node + user-code context so the LLM gets
    actionable feedback.
    """
    from timing_and_emulator.functional import execute
    graph, output_op = _exec_build_graph(code, dims, tensors)
    try:
        sim = execute(graph, output_op)
    except Exception as exc:
        # Enhance emulator errors with node + user code context
        stripped = code.replace("import ", "# import ")  # strip imports for line matching
        enhanced = enhance_emulator_error(exc, stripped)
        raise type(exc)(enhanced) from exc
    return _compare_against_gold(sim, kernel_name, dims, "sim")


# Map executor type to correctness checker. ``dsl`` gates phase-1 refactor
# passes; ``graph`` gates phase-2 translate passes.
_CORRECTNESS_CHECKERS = {
    "dsl":   _run_dsl_correctness,
    "graph": _run_graph_correctness,
}


# ---------------------------------------------------------------------------
# Compliance checking — per-pass regex rules over the function body
# ---------------------------------------------------------------------------

# Per-pass compliance rules. Each pass is independent (no cumulative
# inheritance): the only refactor pass is ``refactor_final``, and the
# translate variants run as alternatives keyed by ``--pipeline``. Each entry
# carries the four standalone fields documented in design/pass_loop.md:
#   - allowed_torch:    set of allowed ``torch.X`` callables (empty = none)
#   - allowed_F:        set of allowed ``F.X`` callables (empty = none)
#   - banned_patterns:  list of (substring, fix-hint) pairs, each surfaced as a
#                       violation line that quotes the fix back to the model
#   - required_ops:     names that must appear textually in the function body
_PASS_RULES: dict[str, dict] = {
    # Phase 1: PyTorch -> DSL form. Output must be pure DSL.
    "refactor_final": {
        # ``torch.tensor`` is permitted exclusively for converting list[int]
        # positional inputs (per-batch sequence lengths, etc.) into a 1D int
        # tensor for DSL producers like ``metadata_gen``. Any other use is
        # caught downstream by the dataflow invariant (a fresh tensor whose
        # source isn't a DSL producer can't reach a DSL consumer).
        "allowed_torch": {"torch.tensor"},
        "allowed_F": set(),
        "banned_patterns": [
            (".unsqueeze(", "use promote(x, rank) or promote_outer(x)"),
            (".squeeze(",   "use flatten(x, rank, rank) or accum_retile_row/col"),
            (".expand(",    "use expand_ref(x, ref) or repeat_static(x, factor)"),
            (".sum(",       "use accum_add(x, rank=1) or unary_rowwise_sum(x)"),
            (".prod(",      "use accum_mul(x, rank=1)"),
            (".reshape(",   "use shape-modifying DSL calls (reshape_stream, flatten, bufferize+streamify)"),
            (".view(",      "use shape-modifying DSL calls (reshape_stream, flatten, bufferize+streamify)"),
            (".permute(",   "can express permutations using shape-modifying DSL calls like bufferize+streamify"),
            (".transpose(", "can express transposes using shape-modifying DSL calls like bufferize+streamify or via the transposed argument in binary ops"),
            (".flatten(",   "use the DSL `flatten(...)` standalone op, not the tensor method"),
            ("torch.matmul", "use binary_matmul(a, b)"),
            ("torch.exp",    "use unary_exp(x)"),
            ("torch.rsqrt",  "use unary_rsqrt(x)"),
            ("F.silu",       "use unary_silu(x)"),
            #("out_shape_tiled=(1,)",
            # "NEVER load as one giant tile — use proper streaming: out_shape_tiled=(B//tile_n,) or similar"),
        ],
        "required_ops": ["offchip_load", "offchip_store"],
    },
    # Phase 2 standard: DSL -> STeP graph. All DSL calls become STeP nodes.
    "translate": {
        "allowed_torch": set(),
        "allowed_F": set(),
        "banned_patterns": [
            ("offchip_load(",    "replace with LinearOffChipLoad(underlying, stride, out_shape_tiled, tile_row, tile_col, par_dispatch, transposed)"),
            ("offchip_store(",   "replace with OffChipStore(graph, input, par_dispatch=4)"),
            ("select_gen(",      "replace with SelectGen(is_multihot=..., tensor=..., n=...) - args match the DSL call"),
            ("metadata_gen(",    "replace with MetadataGen(tensor=tensor)"),
            ("binary_matmul(",   "replace with BinaryMap(graph, a, b, map_fn.Matmul(), False, 1024)"),
            ("binary_mul(",      "replace with BinaryMap(graph, a, b, map_fn.Mul(), False, 1024)"),
            ("binary_add(",      "replace with BinaryMap(graph, a, b, map_fn.Add(), False, 1024)"),
            ("binary_div(",      "replace with BinaryMap(graph, a, b, map_fn.Div(), False, 1024)"),
            ("binary_is_equal(", "replace with BinaryMap(graph, a, b, map_fn.IsEqual(), False, 1024)"),
            ("binary_map_accum(","replace with BinaryMapAccum(graph, a, b, map_accum_fn.Matmul(), init_fn.Zero(...), rank, False, 1024)"),
            ("unary_silu(",      "replace with UnaryMap(graph, x, map_fn.Silu(), False, 1024)"),
            ("unary_square(",    "replace with UnaryMap(graph, x, map_fn.Square(), False, 1024)"),
            ("unary_exp(",       "replace with UnaryMap(graph, x, map_fn.Exp(), False, 1024)"),
            ("unary_rsqrt(",     "replace with UnaryMap(graph, x, map_fn.Rsqrt(), False, 1024)"),
            ("unary_pow2(",      "replace with UnaryMap(graph, x, map_fn.Pow2(), False, 1024)"),
            ("unary_mul_imm(",   "replace with UnaryMap(graph, x, map_fn.MulImmediate(c), False, 1024)"),
            ("unary_add_imm(",   "replace with UnaryMap(graph, x, map_fn.AddImmediate(c), False, 1024)"),
            ("unary_sub_imm(",   "replace with UnaryMap(graph, x, map_fn.SubImmediate(c), False, 1024)"),
            ("unary_rowwise_sum(","replace with UnaryMap(graph, x, map_fn.RowWiseSum(), False, 1024)"),
            ("accum_add(",       "replace with Accum(graph, x, ..., accum_fn.Add(), ..., accum_rank=rank)"),
            ("accum_mul(",       "replace with Accum(graph, x, ..., accum_fn.Mul(), ..., accum_rank=rank)"),
            ("accum_retile_row(","replace with Accum(graph, x, ..., accum_fn.RetileRow(), ...)"),
            ("accum_retile_col(","replace with Accum(graph, x, ..., accum_fn.RetileCol(), ...)"),
            ("promote(",         "replace with Promote(graph, input, promote_rank=rank)"),
            ("promote_outer(",   "replace with PromoteOuter(graph, input)"),
            ("expand_ref(",      "replace with ExpandRef(graph, input, ref, expand_rank=...)"),
            ("repeat_ref(",      "replace with RepeatRef(graph, input, ref)"),
            ("repeat_static(",   "replace with RepeatStatic(graph, input, repeat_factor)"),
            ("flatten(",         "replace with Flatten(graph, input, min_rank, max_rank)"),
            ("reshape_stream(",  "replace with Reshape(graph, input, chunk_size, reshape_rank, write_back_mu=False)"),
            ("retile_streamify(","replace with RetileStreamify(graph, input, split_row, chunk=chunk)"),
            ("broadcast(",       "replace with Broadcast(graph, input, num_consumers=n)"),
            ("parallelize(",     "replace with Parallelize(graph, input, num_consumers=n)"),
            ("static_reassemble(","replace with StaticReassemble(graph, inputs, stream=shape)"),
            ("flat_partition(",  "replace with FlatPartition(graph, input, control, ...)"),
            ("flat_reassemble(", "replace with FlatReassemble(graph, inputs, control, ...)"),
            ("execute_values",   "remove mid-function execution; return (graph, output_op)"),
        ],
        "required_ops": ["LinearOffChipLoad", "OffChipStore"],
    },
    # Phase 2 direct: PyTorch -> STeP graph in one LLM pass; no DSL intermediate.
    "translate_full": {
        "allowed_torch": set(),
        "allowed_F": set(),
        "banned_patterns": [
            ("execute_values", "remove mid-function execution; return (graph, output_op)"),
        ],
        "required_ops": ["LinearOffChipLoad", "OffChipStore"],
    },
    "translate_full_no_functional": {
        "allowed_torch": set(),
        "allowed_F": set(),
        "banned_patterns": [
            ("execute_values", "remove mid-function execution; return (graph, output_op)"),
        ],
        "required_ops": ["LinearOffChipLoad", "OffChipStore"],
    },
}

_TRANSLATION_PASSES = {"translate", "translate_full", "translate_full_no_functional"}

# DSL alphabet — mirrors prompts/refactor_final_judge_system.txt's DSL_FUNCTIONS.
# Producers create on-chip streams from raw input or generate metadata streams;
# their inputs may be raw ``tensors[...]`` reads. Consumers operate on existing
# on-chip streams and require their tensor inputs to come from a producer,
# another consumer, a blackbox-child return, or a positional intermediate arg.
_DSL_PRODUCERS: frozenset[str] = frozenset({
    "offchip_load", "offchip_load_ref", "dyn_offchip_load",
    "random_offchip_load",
    "select_gen", "metadata_gen", "expert_addr_gen",
    "cache_read_addr_gen", "filter_last_tile",
})

_DSL_CONSUMERS: frozenset[str] = frozenset({
    # Binary
    "binary_matmul", "binary_mul", "binary_add", "binary_div", "binary_is_equal",
    "binary_set_offset", "binary_row_wise_append", "binary_cache_write_addr_gen",
    "binary_map_accum",
    # Unary
    "unary_silu", "unary_square", "unary_exp", "unary_rsqrt", "unary_pow2",
    "unary_mul_imm", "unary_add_imm", "unary_sub_imm", "unary_rowwise_sum",
    "unary_mask_row", "unary_select_to_scalar", "unary_to_const_int",
    # Accumulation
    "accum_add", "accum_mul", "accum_max",
    "accum_retile_row", "accum_retile_col", "accum_signal_req_all_read",
    # Stream shape
    "promote", "promote_outer", "flatten",
    "reshape_stream", "reshape_pad_stream",
    "expand_ref", "repeat_ref", "repeat_static",
    "streamify", "dyn_streamify", "bufferize", "retile_streamify",
    # Multi-output
    "broadcast", "parallelize", "static_reassemble",
    # Routing
    "eager_merge", "flat_partition", "flat_reassemble",
    # Flatmap
    "flatmap_filter_row_streamify", "flatmap_counter",
    # Sink
    "offchip_store", "random_offchip_store",
})

# Source classifications for a Name's binding, used by the dataflow walk.
_SRC_PRODUCER = "producer"
_SRC_CONSUMER = "consumer"
_SRC_BLACKBOX = "blackbox"
_SRC_INTERMEDIATE_ARG = "intermediate_arg"
_SRC_RAW_TENSORS = "raw_tensors_subscript"
# Positional arg of a non-root function whose parent's call site fed in a raw
# tensor (no DSL source op applied yet). Treated identically to
# ``_SRC_RAW_TENSORS`` by the consumer-source rule: must pass through a DSL
# producer (``offchip_load`` etc.) before reaching a DSL consumer.
_SRC_RAW_INTERMEDIATE_ARG = "raw_intermediate_arg"
_SRC_NON_TENSOR = "non_tensor"
_SRC_UNKNOWN = "unknown"

_ONCHIP_SOURCES = frozenset({
    _SRC_PRODUCER, _SRC_CONSUMER, _SRC_BLACKBOX, _SRC_INTERMEDIATE_ARG,
})

# Source classifications that flag a value as still-raw (off-chip) when fed
# directly into a DSL consumer.
_RAW_SOURCES = frozenset({_SRC_RAW_TENSORS, _SRC_RAW_INTERMEDIATE_ARG})

# Regex to find torch.XXX( and F.XXX( calls
_TORCH_CALL_RE = re.compile(r'\btorch\.(\w+)\s*\(')
_F_CALL_RE = re.compile(r'\bF\.(\w+)\s*\(')


def _strip_annotations(code: str) -> str:
    """Strip STeP annotation comments so they don't interfere with compliance checks."""
    return "\n".join(
        line for line in code.splitlines()
        if not line.strip().startswith("# >>> STeP:")
    )


def _extract_func_body(code: str) -> str:
    """Extract the body of build_graph or tiled_reference for compliance checking.

    For translation passes the build_graph body is the only scope we check —
    scaffold functions (DSL, functional.py) may legitimately use torch.* internally.
    """
    for func_name in ("build_graph", "tiled_reference"):
        marker = f"def {func_name}("
        idx = code.find(marker)
        if idx != -1:
            return code[idx:]
    return code  # fallback: check everything


class _GateResult(NamedTuple):
    """Per-turn gate result.

    feedback: None means the gate passed; a string means the gate failed and
        this string is the user-prompt feedback for the next turn.
    status: status.txt content if this gate determines the turn outcome.
    tokens: judge-call tokens (0 for non-LLM gates).
    """
    feedback: str | None
    status: str
    tokens: int = 0


def _call_name(call: ast.Call) -> str | None:
    """Return the bare callable name for ``foo(...)`` or ``x.foo(...)``."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _walk_no_nested_def(func: ast.FunctionDef):
    """Walk ``func``'s body without descending into nested ``FunctionDef`` bodies.

    Each top-level function is checked in its own frame so a Name that's a
    parameter of an inner helper isn't confused with one in the outer scope.
    """
    def _w(node):
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                continue
            yield from _w(child)
    for stmt in func.body:
        yield from _w(stmt)


def _classify_value(value: ast.AST,
                    name_to_source: dict[str, str],
                    blackbox_set: frozenset[str]) -> str:
    """Classify the data-flow source of an RHS expression."""
    if isinstance(value, ast.Call):
        name = _call_name(value)
        if name in _DSL_PRODUCERS:
            return _SRC_PRODUCER
        if name in _DSL_CONSUMERS:
            return _SRC_CONSUMER
        if name in blackbox_set:
            return _SRC_BLACKBOX
        # Method calls like ``x.reshape(...)`` propagate the receiver's source
        # so a forgotten load on a host-side reshape of ``tensors["x"]`` is
        # still flagged when consumed by a DSL op.
        if isinstance(value.func, ast.Attribute):
            return _classify_value(value.func.value, name_to_source, blackbox_set)
        return _SRC_NON_TENSOR
    if isinstance(value, ast.Subscript):
        if isinstance(value.value, ast.Name) and value.value.id == "tensors":
            return _SRC_RAW_TENSORS
        return _SRC_NON_TENSOR
    if isinstance(value, ast.Name):
        return name_to_source.get(value.id, _SRC_UNKNOWN)
    if isinstance(value, ast.Attribute):
        return _SRC_NON_TENSOR
    if isinstance(value, (ast.Constant, ast.BinOp, ast.UnaryOp,
                          ast.BoolOp, ast.Compare, ast.IfExp,
                          ast.Tuple, ast.List, ast.Set, ast.Dict)):
        return _SRC_NON_TENSOR
    return _SRC_UNKNOWN


def _bind_target(target: ast.AST, src: str,
                 name_to_source: dict[str, str]) -> None:
    """Bind LHS target(s) to a source classification.

    Tuple unpack ``(a, b, c) = call(...)``: each element gets the producer's
    source (an over-approximation — we don't slice tuple returns per element
    but the worst case is ``_SRC_NON_TENSOR`` which we accept silently).
    """
    if isinstance(target, ast.Name):
        name_to_source[target.id] = src
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _bind_target(elt, src, name_to_source)


def _subscript_key_repr(sub: ast.Subscript) -> str:
    s = sub.slice
    if isinstance(s, ast.Constant):
        return repr(s.value)
    return "..."


def _check_consumer_arg(arg: ast.AST,
                         name_to_source: dict[str, str],
                         blackbox_set: frozenset[str],
                         *, op: str) -> str | None:
    """Return a violation message if ``arg`` is clearly an off-chip raw read
    being fed into a DSL consumer; ``None`` otherwise.

    Conservative: only flags ``tensors["X"]`` (direct or via a Name that was
    bound to one). Unknown / non-tensor sources are accepted silently to
    avoid false positives on scalars and host-side helpers.
    """
    if isinstance(arg, ast.Subscript):
        if isinstance(arg.value, ast.Name) and arg.value.id == "tensors":
            key = _subscript_key_repr(arg)
            return (f"- `{op}` consumes a raw `tensors[{key}]` read; wrap it "
                    f"in `offchip_load(tensors[{key}], ...)` (or another "
                    f"source op) before passing to a DSL consumer")
        return None
    if isinstance(arg, ast.Name):
        src = name_to_source.get(arg.id, _SRC_UNKNOWN)
        if src in _ONCHIP_SOURCES:
            return None
        if src == _SRC_RAW_TENSORS:
            return (f"- `{op}` consumes `{arg.id}` which holds a raw "
                    f"`tensors[...]` read; load it via `offchip_load` (or "
                    f"another source op) before passing to a DSL consumer")
        if src == _SRC_RAW_INTERMEDIATE_ARG:
            return (f"- `{op}` consumes `{arg.id}` which is a raw positional "
                    f"arg (the parent's call site forwarded an off-chip "
                    f"tensor); load it via `offchip_load` (or another source "
                    f"op) before passing to a DSL consumer. (Raw args may "
                    f"still be passed straight to a child blackbox.)")
        return None
    if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
        # Method-chain like ``tensors["x"].reshape(...)`` — recurse on the receiver.
        return _check_consumer_arg(arg.func.value, name_to_source, blackbox_set, op=op)
    return None


def _build_name_to_source(func: ast.FunctionDef,
                           blackbox_set: frozenset[str],
                           raw_arg_names: frozenset[str]) -> dict[str, str]:
    """Build the per-function name → source classification map.

    Positional args in ``raw_arg_names`` bind to ``_SRC_RAW_INTERMEDIATE_ARG``
    instead of ``_SRC_INTERMEDIATE_ARG``; this is how the parent's static
    call-site classification flows into the child's dataflow check.
    """
    name_to_source: dict[str, str] = {}
    # Positional args other than dims/tensors are intermediate. By default
    # they're on-chip (the parent contract guarantees that for non-raw args);
    # those listed in ``raw_arg_names`` were forwarded raw at the call site.
    # Keyword-only args (out_shapes, out_perms) are scalars.
    for arg in func.args.args:
        if arg.arg in {"dims", "tensors", "self"}:
            continue
        if arg.arg in raw_arg_names:
            name_to_source[arg.arg] = _SRC_RAW_INTERMEDIATE_ARG
        else:
            name_to_source[arg.arg] = _SRC_INTERMEDIATE_ARG
    for arg in func.args.kwonlyargs:
        name_to_source[arg.arg] = _SRC_NON_TENSOR

    # Bind every assignment target to the source of its RHS.
    for node in _walk_no_nested_def(func):
        if isinstance(node, ast.Assign):
            src = _classify_value(node.value, name_to_source, blackbox_set)
            for tgt in node.targets:
                _bind_target(tgt, src, name_to_source)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            src = _classify_value(node.value, name_to_source, blackbox_set)
            _bind_target(node.target, src, name_to_source)

    return name_to_source


def _check_func_dataflow(func: ast.FunctionDef,
                          blackbox_set: frozenset[str],
                          raw_arg_names: frozenset[str] = frozenset()) -> list[str]:
    """Per-function dataflow check; see ``_check_dataflow_invariant``."""
    name_to_source = _build_name_to_source(func, blackbox_set, raw_arg_names)

    # Every DSL-consumer call's tensor arg must be on-chip.
    violations: list[str] = []
    for node in _walk_no_nested_def(func):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node)
        if call_name not in _DSL_CONSUMERS:
            continue
        for arg in node.args:
            v = _check_consumer_arg(arg, name_to_source, blackbox_set, op=call_name)
            if v is not None:
                violations.append(v)
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs unpacking — skip
                continue
            v = _check_consumer_arg(kw.value, name_to_source, blackbox_set, op=call_name)
            if v is not None:
                violations.append(v)
    return violations


def _check_dataflow_invariant(code: str, *,
                               blackbox_names: tuple[str, ...] = (),
                               raw_arg_names: frozenset[str] = frozenset()) -> list[str]:
    """AST-level dataflow check for ``refactor_final`` output.

    For every DSL consumer call (``binary_*``, ``unary_*``, ``offchip_store``,
    …), each tensor-typed positional or keyword argument must trace back to a
    valid on-chip source: a DSL producer (``offchip_load*``, ``select_gen``,
    …), another consumer, a blackbox-child return, or a positional
    intermediate function arg (one that isn't ``dims``/``tensors``/``self``).

    Raw ``tensors["X"]`` reads being fed into a DSL consumer are flagged.
    Blackbox-child call sites are exempt because their stubs accept either
    raw or tiled inputs and reshape internally.

    Subsumes the previous textual ``offchip_load`` requirement: a function
    that consumes only intermediate args or blackbox returns generates no
    violations even though it never calls ``offchip_load`` itself.
    """
    blackbox_set = frozenset(blackbox_names)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []  # exec'd elsewhere; don't double-flag here.

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            violations.extend(_check_func_dataflow(
                node, blackbox_set, raw_arg_names))
    # Deduplicate while preserving order.
    return list(dict.fromkeys(violations))


def _extract_call_site_rawness(
    code: str,
    *,
    child_names: tuple[str, ...],
    blackbox_names: tuple[str, ...],
    parent_raw_arg_names: frozenset[str] = frozenset(),
) -> dict[str, tuple[bool, ...]]:
    """Statically classify each child stub call site's positional args as raw / on-chip.

    Walks ``code``'s top-level functions and, for each child name in
    ``child_names``, finds the **first** call to that child (matching the
    Contract-on-first-call semantics in ``make_stub``) and returns a per-arg
    bool tuple: ``True`` means the parent's expression at that position
    classifies as raw (``_SRC_RAW_TENSORS`` or ``_SRC_RAW_INTERMEDIATE_ARG``).

    ``blackbox_names`` (typically the same as ``child_names``) is the set of
    names treated as blackbox callables for source classification.

    ``parent_raw_arg_names`` is the parent's own raw positional arg names
    (taken from the parent's contract). Forwarding a raw arg straight into a
    grandchild keeps it raw, so the parent's rawness must propagate when we
    classify expressions referring to those args.

    A child name absent from the returned dict means no call site was found
    in the parent code; callers should fail loudly in that case (the parent's
    compliance gate already requires that each stub it returned be tracked).
    """
    blackbox_set = frozenset(blackbox_names)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    targets = set(child_names)
    found: dict[str, tuple[bool, ...]] = {}
    for func in ast.walk(tree):
        if not isinstance(func, ast.FunctionDef):
            continue
        # Build the per-function name_to_source the same way the dataflow
        # check does, so rawness conclusions stay aligned with what the check
        # would enforce. Feed in the parent's own raw arg names so a forwarded
        # raw arg stays raw at the grandchild's call site.
        name_to_source = _build_name_to_source(
            func, blackbox_set, raw_arg_names=parent_raw_arg_names)
        for node in _walk_no_nested_def(func):
            if not isinstance(node, ast.Call):
                continue
            cname = _call_name(node)
            if cname not in targets or cname in found:
                continue
            arg_is_raw: list[bool] = []
            for arg in node.args:
                src = _classify_value(arg, name_to_source, blackbox_set)
                arg_is_raw.append(src in _RAW_SOURCES)
            found[cname] = tuple(arg_is_raw)
    return found


def _check_banned_ops(code: str, pass_name: str, *,
                       is_root: bool = True,
                       extra_required_ops: tuple[str, ...] = (),
                       raw_arg_names: frozenset[str] = frozenset()) -> list[str]:
    """Check whether ``code`` complies with this pass's output constraints.

    Returns a list of violation messages; empty means compliant. Each pass's
    rules are independent — there is no cumulative inheritance. For translation
    passes, only the ``build_graph`` body is checked so scaffold code (DSL
    functions, functional.py) may use torch.* internally without false
    positives.

    ``is_root=False`` drops sink ops (``offchip_store`` / ``OffChipStore``)
    from the required-ops set: only the root node writes results off-chip;
    intermediate nodes hand their tensors back to a parent rather than
    storing them.
    """
    rules = _PASS_RULES.get(pass_name)
    if rules is None:
        return []

    # Strip annotation comments before checking — they contain STeP node names
    # that would falsely satisfy required-op checks.
    code = _strip_annotations(code)
    if pass_name in _TRANSLATION_PASSES:
        code = _extract_func_body(code)

    violations: list[str] = []

    allowed_torch = rules["allowed_torch"]
    for match in _TORCH_CALL_RE.finditer(code):
        call = f"torch.{match.group(1)}"
        if call not in allowed_torch:
            violations.append(f"- `{call}()` is not allowed — replace with a canonical pattern")

    allowed_f = rules["allowed_F"]
    for match in _F_CALL_RE.finditer(code):
        call = f"F.{match.group(1)}"
        if call not in allowed_f:
            violations.append(f"- `{call}()` is not allowed — decompose into primitives")

    # Word-boundary anchor avoids false positives like "broadcast(" matching "infer_broadcast(".
    # Only prepend ``\b`` for patterns whose first character is a word char —
    # for dot-prefixed patterns (``.permute(``, ``.unsqueeze(``, ...) ``\b``
    # is the boundary between the previous char and ``.``, which silently
    # fails when the previous char is also non-word (e.g. ``).unsqueeze(``
    # — both ``)`` and ``.`` are non-word, so ``\b`` is False and the call
    # slips past the check). The leading literal ``.`` already prevents
    # partial-word matches, so the boundary is unnecessary in that case.
    for pattern, fix in rules["banned_patterns"]:
        prefix = r'\b' if pattern[:1].isalnum() or pattern[:1] == '_' else ''
        if re.search(prefix + re.escape(pattern), code):
            violations.append(f"- `{pattern}` still present — {fix}")

    sink_ops = {"offchip_store", "OffChipStore", "random_offchip_store"}
    # ``extra_required_ops`` (the planner's child blackbox names) is intentionally
    # NOT required to appear textually: blackbox calls are tools the implementer
    # may use, not a quota. They still flow into the dataflow walk below as
    # ``blackbox_names`` so call sites are exempt from the consumer-source rule.
    required_ops = list(rules["required_ops"])
    if pass_name == "refactor_final":
        # The AST dataflow walk subsumes the textual ``offchip_load`` check —
        # a node consuming only intermediate args / blackbox returns never
        # needs to call ``offchip_load`` itself. ``offchip_store`` is NOT
        # dropped here: the kernel's externally-observable output goes
        # off-chip via a single sink at the root, regardless of whether
        # the root is a leaf or a blackbox-only orchestrator. Children are
        # exempt (handled by the ``not is_root`` rule below).
        required_ops = [op for op in required_ops if op != "offchip_load"]
    for required in required_ops:
        if not is_root and required in sink_ops:
            continue
        if required not in code:
            violations.append(f"- `{required}` missing — this pass must introduce {required} nodes")

    if pass_name == "refactor_final":
        violations.extend(_check_dataflow_invariant(
            code, blackbox_names=tuple(extra_required_ops),
            raw_arg_names=raw_arg_names,
        ))

    # Deduplicate while preserving order
    return list(dict.fromkeys(violations))


def _check_bundle_compliance(code: str, compliance: dict, *,
                              is_root: bool = True) -> list[str]:
    """Bundle-mode compliance checker driven by the bundle's manifest config.

    Mirrors ``_check_banned_ops`` but pulls allowed/banned/required from the
    bundle's compliance dict rather than the hard-coded step_dsl tables.
    Empty allowlist = allowlist disabled.

    ``is_root=False`` drops sink ops (anything in
    ``compliance.get("sink_ops")`` or matching ``*offchip_store*`` /
    ``*OffChipStore*``) from the required-ops set.
    """
    code = _strip_annotations(code)
    code = _extract_func_body(code)

    allowed = compliance.get("allowed_ops") or []
    banned = compliance.get("banned_patterns") or []
    required = compliance.get("required_ops") or []

    violations: list[str] = []

    if allowed:
        # Any torch.X(...) or F.X(...) call whose suffix isn't in `allowed`
        # gets flagged. The allowlist names abstraction-level operators;
        # raw-torch passthrough is what we're trying to catch here.
        allowed_set = set(allowed)
        for match in _TORCH_CALL_RE.finditer(code):
            call_name = match.group(1)
            if call_name not in allowed_set and f"torch.{call_name}" not in allowed_set:
                violations.append(
                    f"- `torch.{call_name}()` is not in this bundle's allowed_ops "
                    "— replace with one of the abstraction's operators"
                )
        for match in _F_CALL_RE.finditer(code):
            call_name = match.group(1)
            if call_name not in allowed_set and f"F.{call_name}" not in allowed_set:
                violations.append(
                    f"- `F.{call_name}()` is not in this bundle's allowed_ops "
                    "— replace with one of the abstraction's operators"
                )

    for entry in banned:
        pattern = entry["pattern"]
        if re.search(r'\b' + re.escape(pattern), code):
            violations.append(f"- `{pattern}` still present — {entry['fix']}")

    declared_sinks = set(compliance.get("sink_ops") or [])
    for name in required:
        if not is_root and (
            name in declared_sinks
            or "offchip_store" in name
            or "OffChipStore" in name
        ):
            continue
        if name not in code:
            violations.append(
                f"- `{name}` missing — this bundle requires a call to {name}"
            )

    return list(dict.fromkeys(violations))


# ---------------------------------------------------------------------------
# Generic pass loop — works for both lowering and translator passes
# ---------------------------------------------------------------------------

async def _run_judge(judge_agent, code: str, turn_dir: Path,
                     log=print, context: str = "") -> tuple[str | None, int]:
    """Run the LLM judge on code. Returns (verdict, tokens_used).

    verdict is None if PASS, or a violation feedback string if REJECT.
    tokens_used is the total_tokens from the SDK RunResult (0 if usage unavailable).
    """
    judge_prompt = f"Review this code:\n\n{context}\n```python\n{code}\n```" if context else \
                   f"Review this code for compliance:\n\n```python\n{code}\n```"
    result = await Runner.run(judge_agent, [{"role": "user", "content": judge_prompt}])
    tokens_used = 0
    if result.context_wrapper.usage is not None:
        tokens_used = result.context_wrapper.usage.total_tokens
    judge_text = result.final_output or ""
    _write(turn_dir / "judge_response.txt", judge_text)
    judge_reasoning = _reasoning_text(result)
    if judge_reasoning:
        _write(turn_dir / "judge_reasoning.txt", judge_reasoning)

    if "VERDICT: PASS" in judge_text:
        return None, tokens_used

    # Extract violations from judge response
    if "VERDICT: REJECT" in judge_text:
        # Everything after VIOLATIONS: is the feedback
        idx = judge_text.find("VIOLATIONS:")
        if idx != -1:
            violations_text = judge_text[idx:]
        else:
            violations_text = judge_text[judge_text.find("VERDICT: REJECT"):]
        return violations_text, tokens_used

    # Ambiguous response — treat as reject
    log(f"      Judge gave ambiguous verdict, treating as reject")
    return judge_text, tokens_used


def _make_translation_post_validator(kernel_name: str, dims: dict,
                                     tensors: dict, log,
                                     translate_fn=None):
    """Build a refactor_final post-validator that runs deterministic translation.

    The validator returns ``None`` when the DSL code translates cleanly into a
    correct STeP graph, or a feedback string describing what went wrong (raised
    exception or graph mismatch). The returned string is appended to the next
    user prompt of the refactor loop, so translator-side constraints (e.g.
    ``select_gen`` must precede ``flat_partition``) get fixed by the refactor
    agent rather than failing later in a separate pass.

    ``translate_fn`` defaults to ``_dsl_to_step_translate``; bundle-dir mode
    passes the bundle's own ``transpiler.translate`` instead.
    """
    if translate_fn is None:
        translate_fn = _dsl_to_step_translate

    def validator(code: str, turn_dir: Path) -> str | None:
        check_dir = turn_dir / "translate_check"

        log(f"      [translate-check] running deterministic translator...")
        try:
            step_code = translate_fn(code)
        except Exception:
            err = traceback.format_exc()
            _write(check_dir / "error.txt", err)
            log(f"      [translate-check] translation FAILED: {_error_summary(err)}")
            return (
                "## Correctness: PASS, but deterministic translation failed\n\n"
                "Your DSL code is numerically correct, but the deterministic "
                "DSL→STeP translator could not lower it. The translator expects "
                "DSL primitives in their canonical, statically-analyzable forms — "
                "see the assertion / error message below for the specific "
                "constraint that is being violated.\n\n"
                "```\n" + err + "```\n\n"
                "Adjust the DSL code so this constraint is satisfied while "
                "keeping the output correct."
            )
        _write(check_dir / "step_extracted_code.py", step_code)

        log(f"      [translate-check] verifying STeP graph correctness...")
        try:
            result = _run_graph_correctness(step_code, kernel_name, dims, tensors)
        except Exception:
            err = traceback.format_exc()
            _write(check_dir / "graph_error.txt", err)
            log(f"      [translate-check] graph execution FAILED: {_error_summary(err)}")
            return (
                "## Correctness: PASS at DSL level, but STeP graph fails to execute\n\n"
                "Your DSL code translated into STeP IR, but the resulting graph "
                "fails to execute:\n\n"
                "```\n" + err + "```\n\n"
                "Adjust the DSL code so the lowered STeP graph executes correctly."
            )
        _write(check_dir / "graph_correctness.txt", result)

        if not _overall_match(result):
            log(f"      [translate-check] graph mismatch")
            return (
                "## Correctness: PASS at DSL level, but STeP graph is numerically wrong\n\n"
                "Your DSL code translated and the graph executed, but the output "
                "does not match the gold reference:\n\n"
                + result
                + "\n\nReview shape / stream invariants — usually this means a DSL "
                "op is being used in a way that is correct under DSL semantics but "
                "diverges from STeP IR semantics after deterministic lowering."
            )
        log(f"      [translate-check] OK")
        return None

    return validator


def _run_deterministic_translate(dsl_code: str, kernel_name: str,
                                 dims: dict, tensors: dict,
                                 outer_dir: Path, log,
                                 translate_fn=None) -> dict:
    """Translate DSL -> STeP build_graph deterministically (no LLM).

    Mirrors the on-disk layout of ``_run_pass_loop`` so checkpoints are
    interchangeable: ``<outer_dir>/translate/turn_0/{extracted_code.py,
    correctness_result.txt, status.txt}``.

    ``translate_fn`` defaults to ``_dsl_to_step_translate``; bundle-dir mode
    passes ``transpiler.translate`` from the bundle.
    """
    if translate_fn is None:
        translate_fn = _dsl_to_step_translate
    pass_dir = outer_dir / "translate"
    turn_dir = pass_dir / "turn_0"
    log(f"  Translation pass: translate (deterministic AST rewrite)")

    step_code = translate_fn(dsl_code)
    _write(turn_dir / "extracted_code.py", step_code)
    log(f"      Generated code: {len(step_code)} chars")

    log(f"      Running correctness check (graph)...")
    result = _run_graph_correctness(step_code, kernel_name, dims, tensors)
    _write(turn_dir / "correctness_result.txt", result)

    success = _overall_match(result)
    _write(turn_dir / "status.txt", "PASS" if success else "MISMATCH")
    log(f"  -> deterministic translate {'OK' if success else 'FAILED'}")
    return {"success": success, "code": step_code}


async def _gate_correctness(code, kernel_name, dims, tensors, executor,
                            turn_dir: Path, log, *,
                            extra_globals: dict | None = None,
                            entry_point: str = "tiled_reference",
                            call_args: tuple | None = None,
                            call_kwargs: dict | None = None) -> tuple[_GateResult, str]:
    """Run check_correctness with stdout captured for shape trace.

    Returns (gate_result, shape_trace). The shape trace is captured even on
    failure so the caller can append it to feedback for the LLM.

    ``entry_point``/``call_args``/``call_kwargs`` are only meaningful for the
    DSL executor (non-root pass1 invokes
    ``<node_name>(<arg_1>, ..., *, out_shapes, out_perms=None)``); the graph
    executor ignores them and stays on the canonical
    ``build_graph(dims, tensors)`` entry point.
    """
    check_correctness = _CORRECTNESS_CHECKERS[executor]
    log(f"      Running correctness check ({executor})...")
    extra_kwargs: dict = {}
    if executor == "dsl":
        extra_kwargs["entry_point"] = entry_point
        extra_kwargs["call_args"] = call_args
        extra_kwargs["call_kwargs"] = call_kwargs
    _trace_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_trace_buf):
            result = check_correctness(code, kernel_name, dims, tensors,
                                       extra_globals=extra_globals,
                                       **extra_kwargs)
        shape_trace = _trace_buf.getvalue()
        if shape_trace:
            _write(turn_dir / "shape_trace.txt", shape_trace)
        _write(turn_dir / "correctness_result.txt", result)

        if _overall_match(result):
            return _GateResult(None, "PASS", 0), shape_trace

        first_line = result.splitlines()[0] if result else "(empty)"
        return (
            _GateResult(
                feedback=f"## Correctness check result\n{result}",
                status=f"FAIL: {first_line}",
                tokens=0,
            ),
            shape_trace,
        )
    except Exception:
        shape_trace = _trace_buf.getvalue()
        if shape_trace:
            _write(turn_dir / "shape_trace.txt", shape_trace)
        err = traceback.format_exc()
        _write(turn_dir / "correctness_result.txt", f"ERROR:\n{err}")
        return (
            _GateResult(
                feedback=f"## Error running code\n{err}",
                status=f"FAIL: {_error_summary(err)}",
                tokens=0,
            ),
            shape_trace,
        )


def _build_judge_context(tensors, *, correctness_verified: bool) -> str:
    """Compose the judge prompt context (correctness-status preamble + tensors)."""
    if correctness_verified:
        ctx = (
            "## Correctness status\n"
            "This code has ALREADY been executed and its output matches the reference "
            "to within floating-point tolerance. Numerical correctness is verified. "
            "Only evaluate the three structural checks.\n\n"
        )
    else:
        ctx = (
            "## Correctness status\n"
            "This code has NOT yet been executed against the reference. Numerical "
            "correctness is not yet verified. Evaluate only the structural checks; "
            "correctness will be checked separately.\n\n"
        )
    if tensors is not None:
        ctx += "## Input tensors\n" + _format_tensors_description(tensors) + "\n\n"
    return ctx


async def _gate_compliance(code, pass_name, compliance_override, judge_agent,
                           tensors, turn_dir: Path, log,
                           *, correctness_verified: bool,
                           is_root: bool = True,
                           extra_required_ops: tuple[str, ...] = (),
                           raw_arg_names: frozenset[str] = frozenset()
                           ) -> _GateResult:
    """Regex compliance check.

    On failure, on `pass_name == "refactor_final"` with a non-None judge_agent,
    also runs the LLM judge for richer line-specific feedback (parity with the
    pre-refactor inline carve-out).
    """
    if compliance_override is not None:
        violations = _check_bundle_compliance(code, compliance_override,
                                              is_root=is_root)
    else:
        violations = _check_banned_ops(code, pass_name, is_root=is_root,
                                       extra_required_ops=extra_required_ops,
                                       raw_arg_names=raw_arg_names)

    if not violations:
        return _GateResult(None, "PASS", 0)

    log(f"      -> {len(violations)} compliance violation(s)")

    if pass_name in _TRANSLATION_PASSES:
        fix_hint = "Replace these with the corresponding STeP operations."
    else:
        fix_hint = "Replace these with the corresponding DSL function calls listed in the instructions."

    judge_feedback = ""
    judge_tokens = 0
    if judge_agent is not None and pass_name == "refactor_final":
        log(f"      Also running judge for richer feedback...")
        judge_ctx = _build_judge_context(tensors, correctness_verified=correctness_verified)
        judge_violations, judge_tokens = await _run_judge(
            judge_agent, code, turn_dir, log, context=judge_ctx)
        if judge_violations is not None:
            judge_feedback = "\n\n## Judge feedback (line-specific):\n\n" + judge_violations

    if correctness_verified:
        preamble = (
            "## Correctness: PASS\n\n"
            "Your code produces the correct output, but still contains "
            "disallowed operations:\n\n"
        )
        status = "CORRECT_BUT_NONCOMPLIANT"
    else:
        preamble = (
            "## Compliance check FAILED\n\n"
            "Your code uses disallowed operations:\n\n"
        )
        status = "NONCOMPLIANT"

    feedback = preamble + "\n".join(violations) + f"\n\n{fix_hint}" + judge_feedback
    return _GateResult(feedback, status, judge_tokens)


async def _gate_judge(judge_agent, code, tensors, turn_dir: Path, log,
                      *, correctness_verified: bool) -> _GateResult:
    """LLM judge over canonical-form structural checks."""
    if judge_agent is None:
        return _GateResult(None, "PASS", 0)

    log(f"      Running judge...")
    judge_ctx = _build_judge_context(tensors, correctness_verified=correctness_verified)
    judge_violations, judge_tokens = await _run_judge(
        judge_agent, code, turn_dir, log, context=judge_ctx)

    if judge_violations is None:
        return _GateResult(None, "PASS", judge_tokens)

    if correctness_verified:
        feedback = (
            "## Correctness: PASS\n\n"
            "Your code produces the correct output and uses allowed operations, "
            "but does not follow canonical form:\n\n"
            + judge_violations
            + "\n\nFix these structural issues while keeping the output correct."
        )
        status = "CORRECT_BUT_JUDGE_REJECTED"
    else:
        feedback = (
            "## Judge rejected\n\n"
            "Your code does not follow canonical form:\n\n"
            + judge_violations
            + "\n\nFix these structural issues."
        )
        status = "JUDGE_REJECTED"

    return _GateResult(feedback, status, judge_tokens)


def _gate_post_validator(post_validator, code, turn_dir: Path, log) -> _GateResult:
    """Deterministic translate-check (DSL -> STeP IR; runs graph against gold).

    Always runs after correctness has passed in both orderings, so the
    `CORRECT_BUT_*` status prefix is accurate regardless of `--check-order`.
    """
    if post_validator is None:
        return _GateResult(None, "PASS", 0)
    log(f"      Running post-validator...")
    feedback = post_validator(code, turn_dir)
    if feedback is None:
        return _GateResult(None, "PASS", 0)
    return _GateResult(feedback, "CORRECT_BUT_POST_VALIDATOR_REJECTED", 0)


def _build_stateless_user_prompt(original_user_prompt: str,
                                  last_code: str | None,
                                  last_feedback: str) -> str:
    """Compose a single-message user prompt for the next stateless turn.

    Used by ``_run_pass_loop`` in stateless mode: each turn ships only the
    original task + the most recent failed attempt + the most recent error,
    discarding the accumulating chat history. Keeps per-turn context O(1)
    and the prompt-cache prefix stable across turns. ``last_code`` is None
    on the no-code-extracted path; we just omit that section.
    """
    parts = [original_user_prompt, "\n\n---\n"]
    if last_code is not None:
        parts.append(
            "### Your Most Recent Attempt (FAILED)\n\n"
            f"```python\n{last_code.rstrip()}\n```\n\n"
        )
    parts.append(f"### Feedback on That Attempt\n\n{last_feedback}\n\n")
    parts.append(
        "Produce a corrected implementation. The original task and "
        "constraints are above; write a fresh, complete function."
    )
    return "".join(parts)


def _make_attempt_log(base_log, node_path: str, attempt_idx: int,
                      total_attempts: int):
    """Wrap ``log`` so every inner line carries its node-path + attempt tag.

    Pass1 and the planner spawn parallel ``_run_pass_loop`` instances per
    node and per attempt; without this their `[refactor_final] Turn N/M`,
    `Extracted code: …`, gate-result lines all interleave anonymously and
    you can't tell which checkpoint dir a given line came from.
    """
    if total_attempts <= 1:
        prefix = f"[{node_path}]"
    else:
        prefix = f"[{node_path} attempt_{attempt_idx}]"

    def wrapped(msg: str) -> None:
        stripped = msg.lstrip(" ")
        leading = msg[: len(msg) - len(stripped)]
        base_log(f"{leading}{prefix} {stripped}")

    return wrapped


async def _run_pass_loop(agent, pass_name, kernel_name, dims, max_turns,
                         ckpt_dir: Path, *, executor: str, tensors: dict,
                         prev_code=None, log=print,
                         judge_agent=None, dsl_code=None,
                         post_validator=None,
                         compliance_override: dict | None = None,
                         check_order: str,
                         prebuilt_user_prompt: str | None = None,
                         is_root: bool = True,
                         stateless: bool = False,
                         extra_required_ops: tuple[str, ...] = (),
                         extra_globals: dict | None = None,
                         entry_point: str = "tiled_reference",
                         call_args: tuple | None = None,
                         call_kwargs: dict | None = None,
                         raw_arg_names: frozenset[str] = frozenset(),
                         pre_turn_hook=None):
    """Run a single pass agent (lowering or translator).

    ``post_validator`` is an optional ``(code, turn_dir) -> str | None`` callable
    that runs after correctness + regex compliance + judge all pass. Returning a
    string treats the turn as failed and feeds that string back into the next
    user prompt — this is how deterministic translation surfaces errors back to
    the refactor pass.

    ``stateless`` (default False) controls how multi-turn context is handled.
    When False, the conversation accumulates: each turn appends the assistant
    response and the user feedback, so the model sees its full attempt history.
    When True, each turn rebuilds the conversation as a single user message
    containing the original prompt + the latest failed attempt + the latest
    feedback — no history beyond the most recent failure.

    ``pre_turn_hook`` is an optional zero-arg callable invoked at the top of
    each turn iteration, before any code or gate runs. Used by pass1 to clear
    per-turn state (e.g. ``ContractRecorder``s) so that side effects observed
    on a failed turn do not leak into the harvest of a later winning turn.

    Returns dict with success, code.
    """
    if prebuilt_user_prompt is not None:
        user_prompt = prebuilt_user_prompt
    else:
        user_prompt = build_pass_user_prompt(pass_name, kernel_name, dims,
                                             prev_code=prev_code,
                                             tensors=tensors,
                                             dsl_code=dsl_code)
    original_user_prompt = user_prompt
    conversation = [{"role": "user", "content": user_prompt}]

    pass_dir = ckpt_dir / pass_name
    # Save the system prompt actually used by the agent (agent.instructions
    # already contains any {few_shot_examples} substitutions from make_pass_agent).
    _write(pass_dir / "system_prompt.txt", agent.instructions)

    last_code = None
    last_feedback: str | None = None
    success = False
    total_tokens = 0

    for turn in range(max_turns):
        turn_dir = pass_dir / f"turn_{turn}"
        log(f"    [{pass_name}] Turn {turn + 1}/{max_turns}...")

        if pre_turn_hook is not None:
            pre_turn_hook()

        if stateless and turn > 0:
            assert last_feedback is not None, (
                "stateless mode: turn>0 must have last_feedback set by prior turn")
            conversation = [{"role": "user", "content":
                _build_stateless_user_prompt(
                    original_user_prompt, last_code, last_feedback)}]

        last_user_msg = conversation[-1]["content"] if conversation[-1]["role"] == "user" else ""
        _write(turn_dir / "user_prompt.txt", last_user_msg)

        from openai import BadRequestError as _BadRequestError
        try:
            run_result = await Runner.run(agent, conversation)
        except _BadRequestError as exc:
            log(f"      LLM rejected request ({exc}); aborting this {pass_name} attempt.")
            _write(turn_dir / "status.txt", f"LLM_BAD_REQUEST: {exc}")
            return {"success": False, "code": last_code, "total_tokens": total_tokens,
                    "last_messages": [{"role": "user", "content": last_user_msg},
                                       {"role": "assistant", "content": f"<LLM rejected: {exc}>"}]}
        if run_result.context_wrapper.usage is not None:
            total_tokens += run_result.context_wrapper.usage.total_tokens
        assistant_text = run_result.final_output or ""
        if not stateless:
            conversation.append({"role": "assistant", "content": assistant_text})
        _write(turn_dir / "response.txt", assistant_text)
        reasoning = _reasoning_text(run_result)
        if reasoning:
            _write(turn_dir / "reasoning.txt", reasoning)

        code = _extract_code(assistant_text)
        if not code:
            log(f"      No code block found ({len(assistant_text)} chars). Retrying.")
            _write(turn_dir / "status.txt", "NO_CODE_EXTRACTED")
            nocode_feedback = (
                "Your response did not contain extractable Python. Either wrap "
                "the implementation in a ```python ... ``` fence, OR make the "
                "entire response valid Python source with no surrounding prose "
                "(comments are fine). Your previous response failed both checks."
            )
            last_feedback = nocode_feedback
            if not stateless:
                conversation.append({"role": "user", "content": nocode_feedback})
            continue

        last_code = code
        _write(turn_dir / "extracted_code.py", code)
        log(f"      Extracted code: {len(code)} chars")

        # ----- Per-turn gate cascade -----
        assert check_order in ("correctness-first", "compliance-first", "always-both"), \
            f"Unknown check_order={check_order!r}"

        if check_order == "correctness-first":
            gate_order = ["correctness", "compliance", "judge", "post_validator"]
            correctness_verified = True
            break_on_fail = True
        elif check_order == "compliance-first":
            gate_order = ["compliance", "judge", "correctness", "post_validator"]
            correctness_verified = False
            break_on_fail = True
        else:  # "always-both"
            # Same order as correctness-first so the compliance/judge gates
            # see an accurate correctness_verified flag, but never short-
            # circuits — every applicable gate runs and their feedback is
            # concatenated into the next user turn.
            gate_order = ["correctness", "compliance", "judge", "post_validator"]
            correctness_verified = True
            break_on_fail = False

        shape_trace = ""
        turn_feedbacks: list[str] = []
        turn_statuses: list[str] = []
        turn_feedback: str | None = None
        turn_status: str | None = None
        success_this_turn = False
        compliance_invoked_judge = False

        try:
            for gate_name in gate_order:
                if gate_name == "correctness":
                    res, shape_trace = await _gate_correctness(
                        code, kernel_name, dims, tensors, executor,
                        turn_dir, log,
                        extra_globals=extra_globals,
                        entry_point=entry_point,
                        call_args=call_args,
                        call_kwargs=call_kwargs)
                    correctness_verified = (res.feedback is None)
                elif gate_name == "compliance":
                    # _gate_compliance also runs the judge inline when it
                    # fails on refactor_final with a judge_agent set; track
                    # that so the standalone judge gate below doesn't burn a
                    # duplicate LLM call for the same answer.
                    inline_judge_eligible = (
                        pass_name == "refactor_final" and judge_agent is not None
                    )
                    res = await _gate_compliance(
                        code, pass_name, compliance_override, judge_agent,
                        tensors, turn_dir, log,
                        correctness_verified=correctness_verified,
                        is_root=is_root,
                        extra_required_ops=extra_required_ops,
                        raw_arg_names=raw_arg_names)
                    if res.feedback is not None and inline_judge_eligible:
                        compliance_invoked_judge = True
                elif gate_name == "judge":
                    if compliance_invoked_judge:
                        res = _GateResult(None, "PASS", 0)
                    else:
                        res = await _gate_judge(
                            judge_agent, code, tensors, turn_dir, log,
                            correctness_verified=correctness_verified)
                else:  # "post_validator"
                    # Translate-check requires the DSL to be runnable; skip
                    # if correctness failed (matches existing cascade behavior
                    # in correctness-first/compliance-first).
                    if not correctness_verified:
                        res = _GateResult(None, "PASS", 0)
                    else:
                        res = _gate_post_validator(post_validator, code, turn_dir, log)
                total_tokens += res.tokens
                if res.feedback is not None:
                    turn_feedbacks.append(res.feedback)
                    turn_statuses.append(res.status)
                    if break_on_fail:
                        break

            if not turn_feedbacks:
                turn_status = "PASS"
                success_this_turn = True
            else:
                turn_feedback = "\n\n---\n\n".join(turn_feedbacks)
                turn_status = " ; ".join(turn_statuses)
        except Exception:
            err = traceback.format_exc()
            turn_feedback = f"## Error running code\n{err}"
            turn_status = f"FAIL: {_error_summary(err)}"

        _write(turn_dir / "status.txt", turn_status)
        if turn_status == "PASS" and judge_agent is not None:
            log(f"      -> PASS (judge approved)")
        else:
            log(f"      -> {turn_status}")

        if success_this_turn:
            success = True
            break

        feedback = turn_feedback

        # Shape-trace tail truncation (preserved verbatim from pre-refactor).
        if shape_trace:
            lines = shape_trace.splitlines()
            MAX_LINES = 200
            if len(lines) > MAX_LINES:
                trace_body = (
                    f"... ({len(lines) - MAX_LINES} earlier lines elided) ...\n"
                    + "\n".join(lines[-MAX_LINES:])
                )
            else:
                trace_body = "\n".join(lines)
            feedback += (
                "\n\n## STeP DSL shape trace\n"
                "Each line shows the input or output shape(s) of a step_dsl op "
                "call, in execution order. `stream(...)×tile(R,C)` means the "
                "tensor's stream shape is `(...)` and its tile shape is `(R,C)`. "
                "Use this to verify shape invariants (binary ops require "
                "identical stream shapes); when the run errored, the trace "
                "ends just before the failing op.\n"
                "```\n" + trace_body + "\n```"
            )

        # Import-hint augmentation predicates (only on correctness exceptions —
        # the predicate strings can appear in LLM prose otherwise).
        if feedback.startswith("## Error running code"):
            if "ModuleNotFoundError" in feedback or "ImportError" in feedback:
                feedback += (
                    "\n\n**IMPORTANT: Do NOT include any import statements in your code.** "
                    "All imports are injected automatically. Remove ALL import/from lines."
                )
            if "missing 1 required positional argument" in feedback:
                feedback += (
                    "\n\n**IMPORTANT: Most STeP ops require `graph` as the FIRST positional arg.** "
                    "Source ops (LinearOffChipLoad, SelectGen, MetadataGen) do NOT take graph. "
                    "ALL other ops take `graph` as their first argument: "
                    "`Promote(graph, input, promote_rank=2)` not `Promote(input, promote_rank=2)`."
                )
            if "FlatPartition" in feedback and ("not subscriptable" in feedback or "not iterable" in feedback):
                feedback += (
                    "\n\n**IMPORTANT: FlatPartition returns a single node, NOT a list.** "
                    "To access per-branch streams, pass a TUPLE `(partitioned, i)` as the input "
                    "to downstream ops. Example:\n"
                    "```python\n"
                    "partitioned = FlatPartition(graph, input_node, select_gen, ...)\n"
                    "# Access branch i:\n"
                    "branch_op = BinaryMap(graph, (partitioned, i), weight_load, ...)\n"
                    "```\n"
                    "Do NOT index `partitioned[i]` or iterate `for x in partitioned`."
                )

        feedback += (
            "\n\n**Fix the specific error above by making targeted changes to your "
            "previous code."
        )
        last_feedback = feedback
        if not stateless:
            conversation.append({"role": "user", "content": feedback})

    return {"success": success, "code": last_code, "total_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Per-node refactor walker (planner tree traversal)
# ---------------------------------------------------------------------------

def _synth_kernel_name(root_kernel: str, node_path: str) -> str:
    safe = node_path.replace("/", "_")
    return f"__plan_{root_kernel}_{safe}__"


def _extract_get_inputs_source(reference_code: str) -> str:
    """Return the literal source of the ``def get_inputs(dims):`` block."""
    import ast as _ast
    tree = _ast.parse(reference_code)
    for node in tree.body:
        if isinstance(node, _ast.FunctionDef) and node.name == "get_inputs":
            return _ast.get_source_segment(reference_code, node) or ""
    raise AssertionError("reference_code does not define get_inputs(dims)")


class _NodeFailure(Exception):
    def __init__(self, result: dict):
        self.result = result


async def _refactor_one_node(*, node, dims, root_kernel, ckpt_root,
                              agent_factory, max_turns, log,
                              children_dsls, node_attempts: int = 1,
                              non_root_sequential: bool = True,
                              translate_fn=None,
                              stateless: bool = False):
    """Refactor a single tree node. Returns the same dict shape as ``_run_pass_loop``.

    Gold is always computed from ``node.reference_code`` (the original Model,
    no child dependencies). The refactor agent sees ``node.refactored_code``
    when it exists, since that is the form decomposed into children.

    When ``node_attempts > 1``, spawns N parallel ``_run_pass_loop`` instances
    under ``<node_dir>/attempt_<i>/refactor_final/``; first success wins and
    the rest are cancelled. With ``node_attempts == 1`` the legacy single-
    attempt layout is used (artifacts go directly under ``<node_dir>``).

    If ``non_root_sequential`` is True and the node is non-root, the N attempts
    run one-at-a-time with early-exit on success, so attempts that aren't
    needed don't burn LLM tokens. Root nodes always run in parallel.
    """
    from src.planner import build_node_tensors, has_class_model
    from src.prompts import _get_precompute_source

    synth_name = _synth_kernel_name(root_kernel, node.path)

    ref_ns: dict = {}
    exec(node.reference_code, ref_ns)
    assert "compute_gold" in ref_ns, (
        f"node {node.path!r}: reference_code must define compute_gold(dims). "
        f"For LLM-synthesized children this is added by synthesize_reference_module; "
        f"for StepDB roots it ships with the seed kernel."
    )

    if has_class_model(node.reference_code):
        tensors = build_node_tensors(node.reference_code, dims)
        precompute_src = _extract_get_inputs_source(node.reference_code)
    else:
        assert node.path == "root", (
            f"only the root may be a function-based reference; node {node.path!r} "
            f"has no class Model but is not the root. LLM-emitted children always "
            f"go through synthesize_reference_module which produces a Model class."
        )
        tensors = precompute_tensors(root_kernel, dims)
        precompute_src = _get_precompute_source(root_kernel)

    import inspect as _inspect
    gold_arity = len(_inspect.signature(ref_ns["compute_gold"]).parameters)
    assert gold_arity in (1, 2), (
        f"node {node.path!r}: compute_gold must take (dims) or (dims, tensors); "
        f"got signature with {gold_arity} parameters"
    )
    with torch.no_grad():
        gold = (ref_ns["compute_gold"](dims) if gold_arity == 1
                else ref_ns["compute_gold"](dims, tensors))
    _inject_gold(synth_name, dims, gold)

    agent_facing_code = (
        node.refactored_code if node.refactored_code is not None
        else node.reference_code
    )
    user_prompt = build_pass_user_prompt(
        "refactor_final", synth_name, dims,
        tensors=tensors,
        reference_code_override=agent_facing_code,
        precompute_source_override=precompute_src,
    )
    if children_dsls:
        few_shot_block = "\n\n## Verified sub-task DSLs (reference material)\n"
        for child_path, dsl in children_dsls:
            few_shot_block += f"\n### {child_path}\n```python\n{dsl.rstrip()}\n```\n"
        few_shot_block += (
            "\n### How to compose these sub-routines\n"
            "Each verified sub-routine above was written as a self-contained DSL "
            "program: it loads its own inputs from `tensors[...]` via "
            "`offchip_load` / `offchip_load_ref` / `dyn_offchip_load` / "
            "`random_offchip_load` and (if it was the root of its subproblem) "
            "writes results via `offchip_store`.\n\n"
            "When you inline / compose these sub-routines into THIS node, those "
            "source-op calls may end up operating on values that are NOT raw "
            "input tensors anymore — they may be intermediate stream tensors "
            "produced by your own DSL ops (the new compute this node introduces "
            "on top of its children). That is illegal: `offchip_load*` must "
            "ONLY be applied to a raw `tensors['<name>']` value, never to the "
            "output of a previous DSL/STeP op. Re-loading already-streamed "
            "values would imply round-tripping through off-chip memory mid-"
            "program, which is not allowed.\n\n"
            "When you encounter this in a sub-routine you are inlining, you "
            "MUST either:\n"
            "1. Remove the offending `offchip_load*` call entirely and feed the "
            "upstream stream tensor directly into whatever consumed the load's "
            "output, OR\n"
            "2. Replace it with the appropriate stream-shape DSL ops "
            "(`promote`, `promote_outer`, `flatten`, `reshape_stream`, "
            "`reshape_pad_stream`, `expand_ref`, `repeat_ref`, `repeat_static`, "
            "`streamify`, `dyn_streamify`, `bufferize`, `retile_streamify`, "
            "`accum_retile_row`, `accum_retile_col`) so that the downstream "
            "computation sees the same stream/tile shape it would have seen "
            "had the load happened on a raw tensor.\n\n"
            "Symmetrically, sub-routines that ended in `offchip_store` should "
            "have that store dropped when their output is consumed by further "
            "DSL ops in this node — only the ROOT program writes results "
            "off-chip; intermediate stages hand stream tensors to their "
            "parent.\n"
        )
        user_prompt = user_prompt + few_shot_block

    node_dir = ckpt_root / "refactor" / node.path
    node_dir.mkdir(parents=True, exist_ok=True)
    agent = agent_factory(children_dsls)
    refactor_judge_agent = getattr(agent_factory, "__refactor_judge_agent__", None)

    is_root = (node.path == "root")

    post_validator = None
    if translate_fn is not None:
        post_validator = _make_translation_post_validator(
            synth_name, dims, tensors, log, translate_fn=translate_fn,
        )

    async def _one_attempt(attempt_idx: int) -> dict:
        attempt_dir = (
            node_dir if node_attempts == 1
            else node_dir / f"attempt_{attempt_idx}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_log = _make_attempt_log(
            log, node.path, attempt_idx, node_attempts)
        return await _run_pass_loop(
            agent, "refactor_final",
            kernel_name=synth_name, dims=dims, max_turns=max_turns,
            ckpt_dir=attempt_dir, executor="dsl", tensors=tensors,
            log=attempt_log,
            check_order="correctness-first",
            prebuilt_user_prompt=user_prompt,
            judge_agent=refactor_judge_agent,
            post_validator=post_validator,
            is_root=is_root,
            stateless=stateless,
        )

    run_sequential = (node_attempts > 1) and non_root_sequential and not is_root

    if node_attempts == 1:
        result = await _one_attempt(0)
    elif run_sequential:
        log(f"[planner] node {node.path!r}: running up to {node_attempts} sequential refactor attempts (non-root)")
        result = {"success": False}
        for i in range(node_attempts):
            r = await _one_attempt(i)
            if r.get("success"):
                log(f"[planner] node {node.path!r}: attempt {i} succeeded — skipping remaining {node_attempts - i - 1}")
                result = r
                break
            result = r
    else:
        log(f"[planner] node {node.path!r}: spawning {node_attempts} parallel refactor attempts")
        pending = {asyncio.create_task(_one_attempt(i)) for i in range(node_attempts)}
        last_failure: dict | None = None
        result = None
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                r = d.result()
                if r.get("success"):
                    log(f"[planner] node {node.path!r}: attempt succeeded — cancelling {len(pending)} pending")
                    for t in pending:
                        t.cancel()
                    result = r
                    pending = set()
                    break
                last_failure = r
        if result is None:
            result = last_failure or {"success": False}

    if not result.get("success"):
        result.setdefault("failing_node", _synth_kernel_name(root_kernel, node.path))
        result.setdefault("last_messages", [])
    return result


# ---------------------------------------------------------------------------
# Pass-1 plumbing (Task 8) — pre-order walk with blackbox stubs
# ---------------------------------------------------------------------------

def _build_node_index(tree, tensors: dict):
    """Walk the tree once, instantiate each node's Module, extract per-node
    NodeSignature.

    Returns a tuple ``(signatures, ref_modules)`` where:
    - ``signatures``: dict[node.path -> NodeSignature]
    - ``ref_modules``: dict[node.path -> nn.Module instance]

    For the root node, canonical inputs are derived from ``tensors`` matched
    against ``Model.forward``'s parameter names.  For non-root nodes, the
    parent's ``refactored_code`` is exec'd with each child's ``Model`` class
    injected as ``{CamelCase(child.name)}Model``; a ``register_forward_pre_hook``
    on each child instance captures the exact tiled args the parent passes.
    """
    import inspect as _inspect

    from src.node_signature import NodeSignature, extract_signature
    from src.planner import _camel_case

    signatures: dict = {}
    ref_modules: dict = {}

    from src.planner import has_class_model

    def _process_node(node, parent_canonical_inputs: dict | None):
        """Recursively process one node, then recurse into its children.

        ``parent_canonical_inputs`` is the already-resolved canonical input dict
        for THIS node (None only for the very first call, indicating root).
        """
        is_root = parent_canonical_inputs is None
        root_is_function_based = is_root and not has_class_model(node.reference_code)

        if root_is_function_based:
            # StepDB function-based root (compute_gold(dims, tensors), no Model class).
            # The root's signature is derived directly from `tensors`; no Module to
            # instantiate. Filter to actual tensors — `tensors` may also carry
            # scalar params (e.g. eps) that don't have a .shape.
            tensor_only = {n: t for n, t in tensors.items() if isinstance(t, torch.Tensor)}
            arg_names = tuple(tensor_only.keys())
            from src.node_signature import (
                NodeSignature as _NS, TensorArg as _TensorArg)
            arg_specs = tuple(
                _TensorArg(shape=tuple(tensor_only[n].shape)) for n in arg_names)
            # Function-based-root NodeSignature is a placeholder: it is created
            # solely so the dict has an entry for ``node.path``. The root's own
            # signature is never consumed downstream — only its CHILDREN's
            # signatures matter, and the function-based-root-with-children case
            # is already rejected at refactor_tree entry.
            signatures[node.path] = _NS(
                arg_names=arg_names, arg_specs=arg_specs,
                out_shapes=(), weight_names=(), out_is_tuple=False,
            )
            canonical_inputs = dict(tensors)
        else:
            # Standard path: instantiate Model and extract signature.
            ns: dict = {}
            exec(node.reference_code, ns)
            assert "Model" in ns, (
                f"node {node.path!r}: reference_code must define class Model(nn.Module)")
            model_instance = ns["Model"]()
            ref_modules[node.path] = model_instance

            if parent_canonical_inputs is None:
                forward_sig = _inspect.signature(model_instance.forward)
                arg_names = tuple(p for p in forward_sig.parameters if p != "self")
                canonical_inputs = {}
                for name in arg_names:
                    assert name in tensors, (
                        f"root forward param {name!r} not found in tensors "
                        f"(available: {sorted(tensors)})")
                    canonical_inputs[name] = tensors[name]
            else:
                canonical_inputs = parent_canonical_inputs

            sig = extract_signature(node.reference_code, canonical_inputs)
            signatures[node.path] = sig

        if not node.children:
            return

        # --- for non-leaf: capture child inputs by running parent's refactored forward ---
        assert node.refactored_code is not None, (
            f"non-leaf node {node.path!r} must have refactored_code to resolve child shapes")

        # Build namespace with each child's Model class under its camel-case name
        parent_ns: dict = {}
        child_instances: dict = {}  # child.path -> nn.Module instance
        for child in node.children:
            child_ns: dict = {}
            exec(child.reference_code, child_ns)
            assert "Model" in child_ns, (
                f"child {child.path!r}: reference_code must define class Model(nn.Module)")
            child_model_class = child_ns["Model"]
            # The parent's refactored_code references children as {CamelCase(name)}Model
            parent_ns[f"{_camel_case(child.name)}Model"] = child_model_class

        exec(node.refactored_code, parent_ns)
        assert "Model" in parent_ns, (
            f"node {node.path!r}: refactored_code must define class Model(nn.Module)")
        parent_model = parent_ns["Model"]()

        # Map child attribute name -> child path for hook dispatch
        child_by_name = {c.name: c for c in node.children}

        # Register pre-hooks on each child sub-module found as a direct attribute
        captured_inputs: dict = {}  # child.name -> tuple of args
        hooks = []
        for attr_name, submodule in parent_model.named_children():
            if attr_name in child_by_name:
                child_path = child_by_name[attr_name].path

                def make_hook(cpath):
                    def hook(module, args):
                        if cpath not in captured_inputs:
                            captured_inputs[cpath] = args
                    return hook

                h = submodule.register_forward_pre_hook(make_hook(child_path))
                hooks.append(h)
                child_instances[child_path] = submodule

        # Run parent's forward once on this node's canonical inputs
        arg_names = tuple(p for p in _inspect.signature(parent_model.forward).parameters
                          if p != "self")
        forward_args = tuple(canonical_inputs[n] for n in arg_names)
        with torch.no_grad():
            parent_model(*forward_args)

        for h in hooks:
            h.remove()

        # Verify all children were reached
        for child in node.children:
            assert child.path in captured_inputs, (
                f"child {child.name!r} (path {child.path!r}) was never called "
                f"during parent {node.path!r} forward — check refactored_code "
                f"assigns each child as a self.{child.name} attribute")

        # Recurse into children using their captured inputs
        for child in node.children:
            raw_args = captured_inputs[child.path]
            child_forward_sig = _inspect.signature(child_instances[child.path].forward)
            child_arg_names = tuple(p for p in child_forward_sig.parameters if p != "self")
            assert len(child_arg_names) == len(raw_args), (
                f"child {child.path!r}: forward declares {len(child_arg_names)} params "
                f"but hook captured {len(raw_args)} args")
            child_canonical = dict(zip(child_arg_names, raw_args))
            _process_node(child, child_canonical)

    _process_node(tree.root, None)
    return signatures, ref_modules


async def _refactor_one_node_pass1(*, node, parent_contract, children_meta,
                                    dims, root_kernel, ckpt_root,
                                    agent_factory, max_turns, log,
                                    node_attempts: int = 1,
                                    non_root_sequential: bool = True,
                                    stateless: bool = False,
                                    tensors: dict,
                                    plan_iter: int = 0):
    """Refactor a single tree node in Pass-1 (pre-order, blackbox-stub style).

    Structurally mirrors ``_refactor_one_node`` but uses Pass-1 prompt/agent
    builders and threads ``extra_globals`` (the blackbox stubs) + ``extra_required_ops``
    (the child names) into ``_run_pass_loop``.

    ``parent_contract`` is the Contract recorded by the grandparent's stub call, or
    None for the root node. ``children_meta`` is a list of
    ``(child_path, child_name, NodeSignature, ref_module)`` tuples; per-attempt
    ``ContractRecorder`` instances and stubs are built fresh inside
    ``_one_attempt`` so contract state from a failed sibling attempt cannot
    leak into the winner's harvest. The winner's recorders are returned in
    ``result["child_recorders"]``.
    """
    from src.agents import make_pass1_agent, make_pass1_judge_agent
    from src.blackbox_stub import ContractRecorder, make_stub
    from src.prompts import build_pass1_user_prompt
    from src.planner import build_node_tensors, has_class_model

    is_root = (node.path == "root")
    synth_name = _synth_kernel_name(root_kernel, node.path)

    # For non-root nodes, the LLM emits
    # ``def <node_name>(<arg_1>, ..., *, out_shapes, out_perms=None)`` — its
    # inputs are the parent's recorded ``tiled_values`` (positional, in
    # ``Contract.arg_names`` order) and its expected outputs are the parent's recorded
    # ``tiled_outputs`` (both captured by ``make_stub`` during the parent's
    # Pass-1 verification). Pulling gold straight from the contract ties the
    # child's check to exactly the call site the parent committed to, with
    # no second input-universe to keep in sync. Root nodes keep the
    # ``tiled_reference(dims, tensors)`` convention against ``compute_gold``.
    if is_root:
        node_tensors = tensors
        ref_ns: dict = {}
        exec(node.reference_code, ref_ns)
        assert "compute_gold" in ref_ns, (
            f"node {node.path!r}: reference_code must define compute_gold(dims)")

        import inspect as _inspect
        gold_arity = len(_inspect.signature(ref_ns["compute_gold"]).parameters)
        assert gold_arity in (1, 2), (
            f"node {node.path!r}: compute_gold must take (dims) or (dims, tensors); "
            f"got signature with {gold_arity} parameters")
        with torch.no_grad():
            gold = (ref_ns["compute_gold"](dims) if gold_arity == 1
                    else ref_ns["compute_gold"](dims, node_tensors))
    else:
        # Non-root: ``parent_contract.tiled_outputs`` is the gold the LLM
        # function must reproduce; we still surface vanilla ``node_tensors``
        # (when the planner emitted a ``Model``) for the prompt's tensor
        # description, which the agent uses for shape orientation only.
        assert parent_contract is not None, (
            f"non-root node {node.path!r} must have a parent_contract")
        if has_class_model(node.reference_code):
            node_tensors = build_node_tensors(node.reference_code, dims)
        else:
            node_tensors = tensors
        if parent_contract.out_is_tuple:
            gold = parent_contract.tiled_outputs
        else:
            assert len(parent_contract.tiled_outputs) == 1, (
                f"non-root node {node.path!r}: parent_contract reports "
                f"out_is_tuple=False but tiled_outputs has "
                f"{len(parent_contract.tiled_outputs)} entries")
            gold = parent_contract.tiled_outputs[0]
    _inject_gold(synth_name, dims, gold)

    # Build function signature string the LLM must produce
    if is_root:
        function_signature = "def tiled_reference(dims, tensors):"
    else:
        # Non-root: positional args are the tiled intermediate args from the contract
        sig_args = ", ".join(parent_contract.arg_names)
        function_signature = f"def {node.name}({sig_args}, *, out_shapes, out_perms=None):"

    # children_signatures (path, sig) pairs are kept for prompt-shape compat.
    children_signatures = [(m[0], m[2]) for m in children_meta]

    from src.node_signature import (
        ListOfIntArg as _ListOfIntArg,
        ListOfTensorArg as _ListOfTensorArg,
        TensorArg as _TensorArg,
        format_arg_spec as _format_arg_spec,
    )

    # Build child_blackbox_block and contract_block strings for the agent factory
    child_blackbox_block = ""
    if children_signatures:
        lines = []
        for child_path, child_sig in children_signatures:
            child_name = child_path.rsplit("/", 1)[-1]
            sig_args = ", ".join(child_sig.arg_names)
            lines.append(f"`{child_name}({sig_args}, *, out_shapes, out_perms=None)`")
            for aname, spec in zip(child_sig.arg_names, child_sig.arg_specs):
                lines.append(f"  - `{aname}` {_format_arg_spec(spec)}")
            if child_sig.out_is_tuple:
                lines.append(
                    f"  returns a tuple of {len(child_sig.out_shapes)} tensors "
                    f"(per-output vanilla shapes: {list(child_sig.out_shapes)}); "
                    f"call site must destructure: ``a_0, a_1, ... = {child_name}(...)``"
                )
            else:
                lines.append(
                    f"  returns a single tensor (vanilla shape {child_sig.out_shapes[0]})"
                )
        child_blackbox_block = "\n".join(lines)

    contract_block = ""
    if not is_root and parent_contract is not None:
        lines = []
        for aname, spec, tshape, raw in zip(
            parent_contract.arg_names,
            parent_contract.arg_specs,
            parent_contract.tiled_shapes,
            parent_contract.arg_is_raw,
        ):
            if raw:
                tag = (
                    "RAW (off-chip — must be loaded with `offchip_load` "
                    "before any DSL consumer; may be passed directly to a "
                    "child blackbox)"
                )
            else:
                tag = (
                    "on-chip (already produced by a sibling DSL op or "
                    "blackbox; pass directly to consumers)"
                )
            if isinstance(spec, _TensorArg):
                lines.append(
                    f"`{aname}`: vanilla shape {spec.shape}, "
                    f"tiled shape {tshape} — {tag}"
                )
            elif isinstance(spec, _ListOfTensorArg):
                lines.append(
                    f"`{aname}`: list[Tensor{spec.elem_shape}] x {spec.length} — {tag}. "
                    f"Iterate at host time and call `offchip_load` per element "
                    f"(e.g. `[offchip_load({aname}[i], ...) for i in range(len({aname}))]`); "
                    f"do NOT pass the list itself to a DSL consumer."
                )
            else:
                assert isinstance(spec, _ListOfIntArg)
                lines.append(
                    f"`{aname}`: list[int] x {spec.length} — {tag}. "
                    f"Convert to a tensor with `torch.tensor({aname})` before "
                    f"any DSL consumer (e.g. feed into `metadata_gen`)."
                )
        lines.append(
            f"Required output shapes (one per produced tensor): "
            f"`{list(parent_contract.out_shapes)}`"
        )
        lines.append(
            f"Output permutations (parallel to shapes; ``None`` = identity): "
            f"`{list(parent_contract.out_perms)}`"
        )
        contract_block = "\n".join(lines)

    # Build the user prompt
    # children_signatures expected as list of (child_name, arg_names,
    # arg_specs, out_shapes, out_is_tuple)
    children_sig_triples = [
        (child_path.rsplit("/", 1)[-1], child_sig.arg_names,
         child_sig.arg_specs, child_sig.out_shapes, child_sig.out_is_tuple)
        for child_path, child_sig in children_signatures
    ]
    # For internal nodes the planner already produced a decomposed parent
    # (``refactored_code``) whose ``forward`` calls each child by name; that
    # is the structural template the LLM should follow, not the original
    # monolithic reference. Leaves keep their own reference_code.
    agent_facing_reference = (
        node.refactored_code
        if node.refactored_code is not None
        else node.reference_code
    )
    user_prompt = build_pass1_user_prompt(
        node_name=node.name,
        is_root=is_root,
        reference_code=agent_facing_reference,
        dims=dims,
        tensors=node_tensors,
        contract=parent_contract,
        children_signatures=children_sig_triples,
        function_signature=function_signature,
    )

    node_dir = ckpt_root / "pass1" / f"iteration_{plan_iter}" / node.path
    node_dir.mkdir(parents=True, exist_ok=True)

    llm_config = getattr(agent_factory, "__llm_config__", None)
    assert llm_config is not None, (
        "agent_factory must expose __llm_config__ for make_pass1_agent; "
        "wrap your factory so it sets agent_factory.__llm_config__ = llm_config")
    is_leaf = not children_signatures
    few_shot_examples = getattr(agent_factory, "__few_shot_examples__", None)
    agent = make_pass1_agent(
        llm_config,
        is_leaf=is_leaf,
        child_blackbox_block=child_blackbox_block,
        contract_block=contract_block,
        few_shot_examples=few_shot_examples,
    )
    judge_agent = make_pass1_judge_agent(
        llm_config,
        is_leaf=is_leaf,
        child_blackbox_block=child_blackbox_block,
        contract_block=contract_block,
        function_signature=function_signature,
    )

    # extra_required_ops = child names (each stub must appear textually in the code)
    extra_required_ops = tuple(
        child_path.rsplit("/", 1)[-1] for child_path, _ in children_signatures
    )

    # Entry-point dispatch: root keeps the canonical
    # ``tiled_reference(dims, tensors)`` convention; non-root invokes the
    # node's own function with the parent-recorded tiled inputs and the
    # parent-declared out_shapes/out_perms.
    if is_root:
        entry_point = "tiled_reference"
        call_args: tuple | None = None
        call_kwargs: dict | None = None
        raw_arg_names_set: frozenset[str] = frozenset()
    else:
        entry_point = node.name
        call_args = tuple(parent_contract.tiled_values)
        call_kwargs = {
            "out_shapes": parent_contract.out_shapes,
            "out_perms": parent_contract.out_perms,
        }
        # Non-root nodes inherit per-arg rawness from the parent's recorded
        # call site. The pass-1 walk stamps ``arg_is_raw`` onto every child
        # contract before recursing, so its length must match arg_names here.
        assert len(parent_contract.arg_is_raw) == len(parent_contract.arg_names), (
            f"non-root node {node.path!r}: parent_contract.arg_is_raw "
            f"({parent_contract.arg_is_raw}) length must match arg_names "
            f"({parent_contract.arg_names}); the orchestrator should stamp "
            f"rawness from the parent's verified code before this point")
        raw_arg_names_set = frozenset(
            name for name, raw in zip(
                parent_contract.arg_names, parent_contract.arg_is_raw)
            if raw
        )

    async def _one_attempt(attempt_idx: int) -> dict:
        attempt_dir = (
            node_dir if node_attempts == 1
            else node_dir / f"attempt_{attempt_idx}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        attempt_log = _make_attempt_log(
            log, node.path, attempt_idx, node_attempts)
        # Per-attempt recorders + stubs, reset before every turn within the
        # attempt. Two leaks the harvester would otherwise see:
        #   1. Cross-attempt: a sibling attempt's failed run records a
        #      contract that the winning attempt then inlines (see
        #      checkpoints/2026-05-10-013023 outer_0 — fixed by
        #      instantiating recorders inside _one_attempt).
        #   2. Cross-turn: an earlier failed turn within this attempt calls
        #      a child stub, then the winning turn inlines it (see
        #      checkpoints/2026-05-10-032138 outer_1: turn_3 called
        #      expert_compute, turn_5 won by inlining; the recorder still
        #      held turn_3's contract → drift assert in _pass1_walk).
        # Resetting at the top of every turn — combined with the loop's
        # break-on-success — guarantees the surviving recorder state was
        # produced by the winning turn alone.
        child_recorders = {m[0]: ContractRecorder() for m in children_meta}
        extras: dict = {}
        for child_path, child_name, child_sig, child_ref in children_meta:
            extras[child_name] = make_stub(
                ref_module=child_ref,
                arg_names=child_sig.arg_names,
                arg_specs=child_sig.arg_specs,
                recorder=child_recorders[child_path],
            )

        def _reset_recorders():
            for r in child_recorders.values():
                r.contract = None

        result = await _run_pass_loop(
            agent, "refactor_final",
            kernel_name=synth_name, dims=dims, max_turns=max_turns,
            ckpt_dir=attempt_dir, executor="dsl", tensors=node_tensors,
            log=attempt_log,
            check_order="correctness-first",
            prebuilt_user_prompt=user_prompt,
            judge_agent=judge_agent,
            is_root=is_root,
            stateless=stateless,
            extra_globals=extras,
            extra_required_ops=extra_required_ops,
            entry_point=entry_point,
            call_args=call_args,
            call_kwargs=call_kwargs,
            raw_arg_names=raw_arg_names_set,
            pre_turn_hook=_reset_recorders,
        )
        # Attach this attempt's recorders so the harvester reads only the
        # winner's stub state, not a sibling's leftovers.
        result["child_recorders"] = child_recorders
        return result

    run_sequential = (node_attempts > 1) and non_root_sequential and not is_root

    if node_attempts == 1:
        result = await _one_attempt(0)
    elif run_sequential:
        log(f"[pass1] node {node.path!r}: running up to {node_attempts} sequential attempts (non-root)")
        result = {"success": False}
        for i in range(node_attempts):
            r = await _one_attempt(i)
            if r.get("success"):
                log(f"[pass1] node {node.path!r}: attempt {i} succeeded — skipping remaining {node_attempts - i - 1}")
                result = r
                break
            result = r
    else:
        log(f"[pass1] node {node.path!r}: spawning {node_attempts} parallel attempts")
        pending = {asyncio.create_task(_one_attempt(i)) for i in range(node_attempts)}
        last_failure: dict | None = None
        result = None
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                r = d.result()
                if r.get("success"):
                    log(f"[pass1] node {node.path!r}: attempt succeeded — cancelling {len(pending)} pending")
                    for t in pending:
                        t.cancel()
                    result = r
                    pending = set()
                    break
                last_failure = r
        if result is None:
            result = last_failure or {"success": False}

    if not result.get("success"):
        result.setdefault("failing_node", synth_name)
        result.setdefault("last_messages", [])
    return result


async def _pass1_walk(*, node, parent_contract, signatures, ref_modules,
                       dims, root_kernel, ckpt_root, agent_factory,
                       max_turns, log, node_attempts, non_root_sequential,
                       stateless, tensors, plan_iter: int = 0):
    """Pre-order Pass-1 walk.

    Refactors ``node`` first (using ``parent_contract`` as the call-site spec),
    then harvests each child's recorded contract from the stub calls the parent
    made during its correctness gate, and recurses into children in parallel.

    Returns a dict mapping node.path -> result dict.  Each result dict has at
    least ``{"success": bool}``.  On success, ``result["dsl"]`` holds the
    verified DSL string.

    Lenient child policy: a planned child whose stub never fired AND whose
    name doesn't appear in the parent's verified code is treated as inlined.
    The walk drops that subtree (no refactor, no contract, no pass2 entry)
    instead of asserting. See orchestrator.py docstring on _pass1_walk for
    the strict-mode alternative.
    """
    children_meta = [
        (c.path, c.name, signatures[c.path], ref_modules[c.path])
        for c in node.children
    ]

    result = await _refactor_one_node_pass1(
        node=node,
        parent_contract=parent_contract,
        children_meta=children_meta,
        dims=dims,
        root_kernel=root_kernel,
        ckpt_root=ckpt_root,
        agent_factory=agent_factory,
        max_turns=max_turns,
        log=log,
        node_attempts=node_attempts,
        non_root_sequential=non_root_sequential,
        stateless=stateless,
        tensors=tensors,
        plan_iter=plan_iter,
    )

    if not result["success"]:
        return {node.path: result}

    # Harvest per-child contracts from the stubs the parent called.
    # Recorders here are scoped to the winning attempt AND reset before every
    # turn within that attempt (see _refactor_one_node_pass1._one_attempt),
    # so contract presence reflects only the winning turn's verified code
    # path — neither failed sibling attempts nor earlier failed turns within
    # this attempt can leak in.
    #
    # Stamp ``arg_is_raw`` onto each contract by statically classifying the
    # parent's verified code at the child's first call site. The parent's own
    # raw arg names (from its own contract) propagate so that an arg forwarded
    # raw through multiple levels stays raw.
    winning_recorders = result.get("child_recorders", {})
    parent_raw_args = (
        frozenset()
        if parent_contract is None
        else frozenset(
            name for name, raw in zip(
                parent_contract.arg_names, parent_contract.arg_is_raw)
            if raw
        )
    )
    child_names = tuple(c.name for c in node.children)
    rawness_map = _extract_call_site_rawness(
        result["code"],
        child_names=child_names,
        blackbox_names=child_names,
        parent_raw_arg_names=parent_raw_args,
    )

    child_contracts: dict = {}
    inlined_children: set[str] = set()
    for child in node.children:
        c = winning_recorders.get(child.path)
        c = c.contract if c is not None else None
        in_ast = child.name in rawness_map
        if c is None and not in_ast:
            # Lenient case: the winner inlined this child entirely. Drop the
            # subtree from the refactor walk and from pass2 composition.
            log(f"[pass1] node {node.path!r}: child {child.name!r} was "
                f"inlined by the parent — pruning subtree from refactor walk")
            inlined_children.add(child.path)
            continue
        # Anything else is real drift — per-attempt + per-turn recorder
        # scoping should have eliminated both forms of leakage that produced
        # this case historically, so a half-set/half-AST state now indicates
        # a genuine bug worth surfacing loudly.
        assert c is not None and in_ast, (
            f"Parent {node.path!r}: drift for child {child.name!r} "
            f"(contract_recorded={c is not None}, in_ast={in_ast}) — "
            f"per-attempt + per-turn recorder scoping should make this "
            f"impossible; investigate")
        rawness = rawness_map[child.name]
        assert len(rawness) == len(c.arg_names), (
            f"Parent {node.path!r}: rawness tuple for child {child.name!r} "
            f"has {len(rawness)} entries but contract.arg_names has "
            f"{len(c.arg_names)} — call-site arity drift")
        child_contracts[child.path] = dataclasses.replace(c, arg_is_raw=rawness)

    # Persist the inlined-children list so resume reads can reproduce the
    # decision without re-running the rawness extractor.
    if inlined_children:
        node_pass1_dir = ckpt_root / "pass1" / f"iteration_{plan_iter}" / node.path
        node_pass1_dir.mkdir(parents=True, exist_ok=True)
        inlined_names = sorted(p.rsplit("/", 1)[-1] for p in inlined_children)
        (node_pass1_dir / "inlined_children.txt").write_text(
            "\n".join(inlined_names) + "\n")

    out: dict = {node.path: {"dsl": result["code"], "success": True}}

    live_children = [c for c in node.children if c.path not in inlined_children]
    if live_children:
        sub_results = await asyncio.gather(*[
            _pass1_walk(
                node=child,
                parent_contract=child_contracts[child.path],
                signatures=signatures,
                ref_modules=ref_modules,
                dims=dims,
                root_kernel=root_kernel,
                ckpt_root=ckpt_root,
                agent_factory=agent_factory,
                max_turns=max_turns,
                log=log,
                node_attempts=node_attempts,
                non_root_sequential=non_root_sequential,
                stateless=stateless,
                tensors=tensors,
                plan_iter=plan_iter,
            )
            for child in live_children
        ])
        for sr in sub_results:
            out.update(sr)

    return out


def _pass2_compose_namespace(*, parent_dsl: str,
                              child_dsls_in_order: list[str]) -> dict:
    """Build an exec namespace where each child name resolves to its verified DSL function.

    ``child_dsls_in_order`` MUST be in post-order (children before
    parents). The Pass-1 LLM is allowed to capture a same-named child via
    ``_<name>_child = <name>`` at module level *before* redefining the
    function — that capture reads ``<name>`` at exec time, so the child's
    ``def`` must already have run. Two same-name ``def``s shadow each
    other in ``ns``; the post-order contract preserves the parent's
    captured alias before its own redef overwrites the child binding.
    For function-call references (the common case), Python's late binding
    makes the order irrelevant — all functions are bound by call time.
    """
    from src.tools import _build_dsl_scaffold
    scaffold = _build_dsl_scaffold()
    ns: dict = {}
    exec(scaffold, ns)
    for child_dsl in child_dsls_in_order:
        exec(child_dsl, ns)
    exec(parent_dsl, ns)
    return ns


def _pass2_compose(*, tree, pass1_dsls: dict[str, str],
                    dims: dict, tensors: dict, root_kernel: str,
                    ckpt_root, log, plan_iter: int = 0) -> dict:
    """Run deterministic Pass 2 at the root level.

    Builds a shared namespace containing every descendant's Pass-1 DSL function,
    then runs ``tiled_reference(dims, tensors)`` and compares to gold.

    Writes per-iteration artifacts to
    ``<ckpt_root>/pass2/iteration_<plan_iter>/`` mirroring the pass1
    layout — ``composed_code.py`` (full descendants+root source),
    ``shape_trace.txt`` (captured step_dsl shape prints from the exec),
    ``correctness_result.txt`` (the comparison report), and
    ``status.txt`` (``PASS`` or ``MISMATCH: <first line>``). On exec
    crash these last three are absent; the outer-iteration wrapper
    logs the traceback to ``<ckpt_root>/log.txt``.

    Returns:
        {"success": True, "root_dsl": <composed string>}  on success
        {"success": False, "failing_node": <root path>, "exec_log": <str>} on failure
    """
    from src.tools import _exec_dsl_ref

    pass2_dir = ckpt_root / "pass2" / f"iteration_{plan_iter}"

    # Collect descendant DSLs in post-order. We keep a list (not a
    # name-keyed dict) because the planner can produce nested same-named
    # nodes — e.g. a parent ``moe_dispatch`` whose child is also named
    # ``moe_dispatch``. Keying by ``node.name`` would silently drop the
    # leaf when the parent overwrites it, so the parent's
    # ``_<name>_child = <name>`` capture would resolve against a namespace
    # where the child was never exec'd → NameError. Indexing by path also
    # has no functional role here (we never look entries up by key); the
    # only thing that matters is iteration order, which iter_topological
    # already guarantees is children-before-parents.
    descendants_in_order: list[str] = []
    for node in tree.iter_topological():   # post-order (children before parent)
        if node.path == tree.root.path:
            continue
        # Lenient mode: a planner-tree node may have been inlined by its
        # parent (no Pass-1 DSL on disk). Such a node has no caller in the
        # composed namespace, so skipping it leaves the rest of the
        # composition intact.
        dsl = pass1_dsls.get(node.path)
        if dsl is None:
            continue
        descendants_in_order.append(dsl)

    root_dsl = pass1_dsls[tree.root.path]
    composed_source = "\n\n".join(descendants_in_order + [root_dsl])
    _write(pass2_dir / "composed_code.py", composed_source)

    log(f"[pass2] composing root with all descendants name-rebound "
        f"(artifacts: {pass2_dir})")

    # Build a namespace containing scaffold + all descendants + root, then
    # re-execute the root through the standard executor with the composed
    # bindings as ``extra_globals``. We extract the extras by walking
    # ``composed_ns`` itself (all callables except the root entry point)
    # rather than indexing by descendant key — this is robust to the
    # same-name shadowing case: the parent's ``def`` overwrote the leaf in
    # ``composed_ns``, but the leaf survives via the parent's captured
    # ``_<name>_child`` alias, which is *also* a callable in ``composed_ns``.
    composed_ns = _pass2_compose_namespace(
        parent_dsl=root_dsl, child_dsls_in_order=descendants_in_order)
    extras = {k: v for k, v in composed_ns.items()
              if callable(v) and k != "tiled_reference"}

    # Capture step_dsl's shape-trace prints into pass2's folder rather
    # than letting them disappear into the parent process's stdout.
    _trace_buf = io.StringIO()
    with contextlib.redirect_stdout(_trace_buf):
        result = _exec_dsl_ref(root_dsl, dims, tensors, extra_globals=extras)
    shape_trace = _trace_buf.getvalue()
    if shape_trace:
        _write(pass2_dir / "shape_trace.txt", shape_trace)

    report = _compare_against_gold(result, root_kernel, dims)
    _write(pass2_dir / "correctness_result.txt", report)

    if _overall_match(report):
        _write(pass2_dir / "status.txt", "PASS")
        return {"success": True, "root_dsl": composed_source}

    first_line = report.splitlines()[0] if report else "(empty)"
    _write(pass2_dir / "status.txt", f"MISMATCH: {first_line}")
    log(f"[pass2] root mismatch: {report[:200]}")
    return {"success": False, "failing_node": tree.root.path,
            "exec_log": report}


async def refactor_tree(*, tree, dims, root_kernel, ckpt_root,
                        agent_factory, max_turns, log,
                        node_attempts: int = 1,
                        non_root_sequential: bool = True,
                        verified_cache: dict[str, str] | None = None,
                        translate_fn=None,
                        stateless: bool = False,
                        tensors: dict | None = None,
                        plan_iter: int = 0) -> dict:
    """Two-pass walk: pre-order Pass 1, then post-order Pass 2 at root.

    ``verified_cache`` and ``translate_fn`` are accepted for backward
    compatibility but are not wired through in v1 of the two-pass flow.

    Returns:
        {"success": True, "root_dsl": <str>}  on success
        {"success": False, "failing_node": <path>, "last_messages": [...],
         "phase": "pass1"|"pass2", "exec_log": <str>}  on failure
    """
    assert tensors is not None, (
        "tensors dict is now required by refactor_tree (used to instantiate "
        "Modules for signature extraction and to drive Pass-1 stubs)")

    # v1 limitation: dict-style modules (forward(self, dims, tensors)) — produced
    # by the planner when the original kernel is a function-based StepDB reference
    # (compute_gold(dims, tensors) with no class Model) — do not fit the per-arg
    # blackbox-stub contract, which assumes positional tensor args. Fail fast with
    # an actionable message rather than crashing later in _build_node_index.
    from src.planner import has_class_model
    if not has_class_model(tree.root.reference_code) and tree.root.children:
        raise AssertionError(
            "Two-pass refactor v1 does not support function-based StepDB kernels "
            "(forward(self, dims, tensors)) with non-leaf decompositions. "
            f"Root {tree.root.path!r} is function-based and has "
            f"{len(tree.root.children)} children. Workarounds: "
            "(a) reduce --max-plan-depth so the planner returns a leaf-only tree, "
            "(b) use a class-based kernel reference, or "
            "(c) extend the contract design to dict-style modules.")

    # Precompute per-node signatures and instantiate Modules once.
    signatures, ref_modules = _build_node_index(tree, tensors)

    # Pass 1: pre-order walk.
    pass1 = await _pass1_walk(
        node=tree.root, parent_contract=None,
        signatures=signatures, ref_modules=ref_modules,
        dims=dims, root_kernel=root_kernel, ckpt_root=ckpt_root,
        agent_factory=agent_factory, max_turns=max_turns, log=log,
        node_attempts=node_attempts,
        non_root_sequential=non_root_sequential,
        stateless=stateless, tensors=tensors,
        plan_iter=plan_iter,
    )
    failing = [(p, r) for p, r in pass1.items() if not r["success"]]
    if failing:
        path, r = failing[0]
        return {"success": False,
                "failing_node": path,
                "last_messages": r.get("last_messages", []),
                "phase": "pass1"}

    pass1_dsls = {p: r["dsl"] for p, r in pass1.items()}

    # Pass 2: deterministic composition + verification at root.
    pass2 = _pass2_compose(
        tree=tree, pass1_dsls=pass1_dsls,
        dims=dims, tensors=tensors,
        root_kernel=root_kernel, ckpt_root=ckpt_root, log=log,
        plan_iter=plan_iter,
    )
    if not pass2["success"]:
        return {"success": False,
                "failing_node": pass2["failing_node"],
                "last_messages": [],
                "phase": "pass2",
                "exec_log": pass2.get("exec_log", "")}

    return {"success": True, "root_dsl": pass2["root_dsl"]}


async def _initial_plan(*, root_reference, dims, agent, log,
                         turn_root: Path | None = None,
                         max_depth: int | None = None,
                         precompute_source: str | None = None,
                         tensors: dict | None = None) -> "PlanNode":
    """Thin wrapper around src.planner.plan for monkeypatching in tests."""
    from src.planner import plan
    return await plan(reference_code=root_reference, dims=dims,
                       agent=agent, path="root",
                       runner_fn=Runner.run, retry_budget=8, log=log,
                       turn_root=turn_root, max_depth=max_depth,
                       precompute_source=precompute_source,
                       tensors=tensors)


async def _replan(*, subtree_reference, dims, agent, node_path,
                  failing_node, last_messages, sibling_results,
                  replan_iteration, log,
                  turn_root: Path | None = None,
                  max_depth: int | None = None,
                  precompute_source: str | None = None,
                  tensors: dict | None = None) -> "PlanNode":
    """Re-invoke the planner on a failing subtree."""
    from src.planner import plan
    return await plan(
        reference_code=subtree_reference, dims=dims, agent=agent,
        path=node_path, runner_fn=Runner.run, retry_budget=8, log=log,
        turn_root=turn_root, max_depth=max_depth,
        precompute_source=precompute_source,
        tensors=tensors,
        replan_context={
            "replan_iteration": replan_iteration,
            "node_path": node_path,
            "failing_node": failing_node,
            "last_turn_messages": last_messages,
            "sibling_results": sibling_results,
        },
    )


async def _run_planner_phase(*, root_reference, dims, root_kernel, ckpt_root,
                             agent_factory, max_turns, log, max_replans,
                             node_attempts: int = 1,
                             non_root_sequential: bool = True,
                             max_plan_depth: int | None = None,
                             precompute_source: str | None = None,
                             tensors: dict | None = None,
                             resumed_tree=None,
                             verified_cache: dict[str, str] | None = None,
                             translate_fn=None,
                             stateless_refactor: bool = False):
    """Top-level Phase 0+1 loop with re-plan on failure.

    When ``resumed_tree`` is provided, the Phase 0 plan call is skipped and the
    given tree is used as iteration_0. ``verified_cache`` (when provided) is
    forwarded to ``refactor_tree`` so previously-verified per-node DSLs are
    reused without re-running the LLM.

    Returns: {"success": True, "root_dsl": str} or
             {"success": False, "failing_node": str, "last_messages": [...]}.
    """
    from src.planner import Tree

    planner_agent = getattr(agent_factory, "__planner_agent__", None)

    log(f"[planner] === Phase 0 starting for kernel {root_kernel!r} (max_replans={max_replans}) ===")

    plan_iter = 0
    iter_dir = ckpt_root / "plan" / f"iteration_{plan_iter}"
    if resumed_tree is not None:
        tree = resumed_tree
        log(f"[planner] iteration {plan_iter}: RESUMED tree from disk — skipping Phase 0 LLM plan")
    else:
        root_node = await _initial_plan(
            root_reference=root_reference, dims=dims,
            agent=planner_agent, log=log,
            turn_root=iter_dir / "turns",
            max_depth=max_plan_depth,
            precompute_source=precompute_source,
            tensors=tensors,
        )
        tree = Tree(root=root_node)
    # Globally disambiguate node names BEFORE pass-1 / pass-2 see the tree.
    # Idempotent — a tree loaded from disk that's already unique is unchanged.
    from src.planner import disambiguate_tree
    tree, renamed = disambiguate_tree(tree)
    if renamed:
        log(f"[planner] iteration {plan_iter}: disambiguated "
            f"{len(renamed)} name collision(s): {renamed}")
    _persist_tree(tree, iter_dir)
    n_nodes = sum(1 for _ in tree.iter_topological())
    cached_n = sum(1 for n in tree.iter_topological()
                   if (verified_cache or {}).get(n.path) is not None)
    log(f"[planner] iteration {plan_iter}: tree built ({n_nodes} node(s); {cached_n} cached) "
        f"— starting Phase 1 refactor walk")

    replans_used = 0
    while True:
        result = await refactor_tree(
            tree=tree, dims=dims, root_kernel=root_kernel,
            ckpt_root=ckpt_root, agent_factory=agent_factory,
            max_turns=max_turns, log=log,
            node_attempts=node_attempts,
            non_root_sequential=non_root_sequential,
            verified_cache=verified_cache,
            translate_fn=translate_fn,
            stateless=stateless_refactor,
            tensors=tensors,
            plan_iter=plan_iter,
        )
        if result["success"]:
            log(f"[planner] === Phase 0+1 SUCCEEDED (used {replans_used} replan(s)) ===")
            return result

        log(f"[planner] iteration {plan_iter}: refactor walk FAILED at node "
            f"{result.get('failing_node')!r}; replans_used={replans_used}/{max_replans}")
        if replans_used >= max_replans:
            log(f"[planner] === Phase 0+1 EXHAUSTED replan budget ({max_replans}) ===")
            return result

        failing = result["failing_node"]
        owner = tree.find_owner(failing) or tree.root
        sibling_results: list = []
        log(f"[planner] replanning subtree at {owner.path!r} (failing leaf: {failing!r})")
        next_iter_dir = ckpt_root / "plan" / f"iteration_{plan_iter + 1}"
        new_subtree = await _replan(
            subtree_reference=owner.reference_code, dims=dims,
            agent=planner_agent, node_path=owner.path,
            failing_node=failing,
            last_messages=result.get("last_messages", []),
            sibling_results=sibling_results,
            replan_iteration=replans_used + 1, log=log,
            turn_root=next_iter_dir / "turns",
            max_depth=max_plan_depth,
            precompute_source=precompute_source,
            tensors=tensors,
        )
        tree = _replace_subtree(tree, owner.path, new_subtree)
        # Re-run global disambiguation: the replanned subtree may have
        # introduced fresh name collisions with surviving parts of the
        # tree. Idempotent for the unchanged portion.
        tree, renamed = disambiguate_tree(tree)
        if renamed:
            log(f"[planner] iteration {plan_iter + 1}: disambiguated "
                f"{len(renamed)} name collision(s) post-replan: {renamed}")
        replans_used += 1
        plan_iter += 1
        _persist_tree(tree, next_iter_dir)
        n_nodes = sum(1 for _ in tree.iter_topological())
        log(f"[planner] iteration {plan_iter}: replanned tree built ({n_nodes} node(s)) — retrying refactor walk")


def _persist_tree(tree, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "tree.json").write_text(json.dumps(_tree_to_dict(tree.root), indent=2))
    _persist_node_files(tree.root, dest)


def _load_tree_from_dir(outer_dir: Path):
    """Inverse of ``_persist_tree``: reconstruct the latest planner tree saved
    under ``<outer_dir>/plan/iteration_*/``.

    Picks the highest iteration index (i.e. the tree state at the time the
    outer crashed, after any replans). Each node's reference.py / refactored.py
    is read back from disk. Used by ``--resume-planner``.
    """
    from src.planner import PlanNode, Tree
    plan_dir = outer_dir / "plan"
    iters = sorted(plan_dir.glob("iteration_*"))
    assert iters, f"no plan iterations found under {plan_dir}"
    latest = iters[-1]
    tree_dict = json.loads((latest / "tree.json").read_text())

    def _load_node(d: dict) -> PlanNode:
        node_dir = latest / d["path"].replace("/", "_")
        reference_code = (node_dir / "reference.py").read_text()
        refactored_code = None
        if d["has_refactored"]:
            refactored_code = (node_dir / "refactored.py").read_text()
        children = tuple(_load_node(c) for c in d["children"])
        return PlanNode(
            name=d["name"], path=d["path"],
            reference_code=reference_code,
            refactored_code=refactored_code,
            is_leaf=d["is_leaf"],
            children=children,
        )

    return Tree(root=_load_node(tree_dict))


def _required_nodes(tree, pass1_dsls: dict[str, str]) -> set[str]:
    """Compute the set of node paths whose Pass-1 DSL must exist for pass2.

    Lenient pass1 lets a parent inline a planned child (no DSL emitted for
    that subtree). Pass2 composition still works because the parent doesn't
    reference the child's name. To validate that a resumed checkpoint can be
    composed, we walk the tree top-down and demand a DSL only for nodes
    whose parent's DSL textually calls them — i.e. nodes that are still
    referenced from the surviving composition.

    Children of an inlined node are unreachable transitively and are not
    required either, since their only consumer was the now-collapsed
    parent node.
    """
    required: set[str] = {tree.root.path}
    queue = [tree.root]
    while queue:
        node = queue.pop()
        parent_dsl = pass1_dsls.get(node.path, "")
        for child in node.children:
            if re.search(rf"\b{re.escape(child.name)}\s*\(", parent_dsl):
                required.add(child.path)
                queue.append(child)
    return required


def _load_verified_dsls(outer_dir: Path) -> dict[str, str]:
    """Scan ``<outer_dir>/pass1/iteration_*/`` for nodes with a successful refactor turn.

    A node is verified iff it has a ``status.txt`` containing exactly ``PASS``
    under either ``refactor_final/turn_*/`` (single-attempt layout) or
    ``attempt_*/refactor_final/turn_*/`` (multi-attempt layout). Returns a
    ``{node_path: verified_dsl_code}`` mapping consumable by ``refactor_tree``.

    When the same node has a PASS in multiple plan iterations (because a
    later replan re-ran it under a new contract), the latest iteration's
    DSL wins — its parent contract matches the live tree.
    """
    pass1_root = outer_dir / "pass1"
    if not pass1_root.is_dir():
        return {}
    iteration_dirs = sorted(
        (d for d in pass1_root.iterdir()
         if d.is_dir() and d.name.startswith("iteration_")),
        key=lambda d: int(d.name.split("_", 1)[1]),
        reverse=True,
    )
    cache: dict[str, str] = {}
    for iter_dir in iteration_dirs:
        for status_path in iter_dir.rglob("status.txt"):
            if status_path.read_text().strip() != "PASS":
                continue
            if status_path.parent.parent.name != "refactor_final":
                continue
            code_path = status_path.parent / "extracted_code.py"
            if not code_path.exists():
                continue
            # Path layout (under iter_dir):
            #   <node_path>/refactor_final/turn_N/                  → 3 trailing parts
            #   <node_path>/attempt_K/refactor_final/turn_N/        → 4 trailing parts
            rel = status_path.parent.parent.parent.relative_to(iter_dir)
            if rel.name.startswith("attempt_"):
                rel = rel.parent
            node_path = str(rel).replace("\\", "/")
            # Latest iteration wins (iteration_dirs is sorted desc); within
            # an iteration, first PASS wins.
            cache.setdefault(node_path, code_path.read_text())
    return cache


def _tree_to_dict(node) -> dict:
    return {
        "name": node.name,
        "path": node.path,
        "is_leaf": node.is_leaf,
        "has_refactored": node.refactored_code is not None,
        "children": [_tree_to_dict(c) for c in node.children],
    }


def _persist_node_files(node, dest: Path) -> None:
    node_dir = dest / node.path.replace("/", "_")
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "reference.py").write_text(node.reference_code)
    if node.refactored_code is not None:
        (node_dir / "refactored.py").write_text(node.refactored_code)
    for c in node.children:
        _persist_node_files(c, dest)


def _replace_subtree(tree, owner_path: str, new_subtree) -> "Tree":
    """Return a new Tree with the subtree at owner_path replaced by new_subtree."""
    from src.planner import Tree
    if owner_path == tree.root.path:
        return Tree(root=new_subtree)
    return Tree(root=_replace_node(tree.root, owner_path, new_subtree))


def _replace_node(node, target_path: str, new_subtree):
    from src.planner import PlanNode
    new_children = tuple(
        new_subtree if c.path == target_path else _replace_node(c, target_path, new_subtree)
        for c in node.children
    )
    return PlanNode(
        name=node.name, path=node.path,
        reference_code=node.reference_code,
        refactored_code=node.refactored_code,
        is_leaf=node.is_leaf, children=new_children,
    )


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------

def _resolve_resume_dsl(resume_from: str, kernel_name: str) -> str:
    """Resolve a resume_from path to a DSL code string.

    Accepts:
        - Direct path to a .py file (e.g., .../outer_0/dsl_code.py)
        - Path to an outer_N directory containing dsl_code.py
        - Path to a checkpoint root (e.g., checkpoints/2026-04-14-035721)
          — searches for <kernel_name>/outer_*/dsl_code.py
    """
    p = Path(resume_from)

    # Case 1: direct path to a .py file
    if p.suffix == ".py" and p.is_file():
        return p.read_text()

    # Case 2: directory containing dsl_code.py
    if p.is_dir() and (p / "dsl_code.py").is_file():
        return (p / "dsl_code.py").read_text()

    # Case 3: checkpoint root — search under kernel_name/outer_*/
    if p.is_dir():
        candidates = sorted((p / kernel_name).glob("outer_*/dsl_code.py"))
        assert candidates, (
            f"No dsl_code.py found under {p / kernel_name}/outer_*/. "
            f"Ensure refactor_final succeeded in the checkpoint you're resuming from."
        )
        chosen = candidates[0]
        print(f"  Resolved resume path: {chosen}")
        return chosen.read_text()

    raise FileNotFoundError(
        f"Cannot resolve resume_from='{resume_from}'. "
        f"Expected a .py file, a directory with dsl_code.py, "
        f"or a checkpoint root directory."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_kernel(
    kernel_name: str,
    preset: str,
    llm_config: dict,
    max_outer: int = 3,
    max_turns: int = 5,
    results_dir: str = "results",
    experience_dir: str = "experience",
    checkpoint_dir: str = None,
    pipeline: str = "standard",
    resume_from: str = None,
    translator: str = "llm",
    few_shot_paths=None,
    bundle_dir: str | None = None,
    autotune_options: dict = None,
    check_order: str = "correctness-first",
    plan_enabled: bool = True,
    max_replans: int = 3,
    node_attempts: int = 1,
    non_root_sequential: bool = True,
    max_plan_depth: int | None = 3,
    resume_planner: str | None = None,
    resume_after_pass1: str | None = None,
    stateless_refactor: bool = False,
) -> dict:
    """Run the full pipeline for a single kernel + preset.

    Args:
        pipeline: "standard" (lowering + translate) or "direct" (PyTorch → STeP in one step).
        translator: "llm" (default) runs the LLM translate pass; "auto" runs the
            deterministic AST translator (``src.dsl_to_step.translate``). "auto"
            requires the standard pipeline since it consumes refactor_final's DSL output.
        resume_from: Path to resume from a previous checkpoint. Accepts:
            - Path to a dsl_code.py file directly
            - Path to an outer_N directory containing dsl_code.py
            - Path to a checkpoint root (e.g. checkpoints/2026-04-14-035721)
              — will search for <kernel_name>/outer_*/dsl_code.py
            When set, lowering is skipped and translation starts from the saved DSL code.
    """
    assert pipeline in PIPELINES, f"Unknown pipeline '{pipeline}'. Known: {sorted(PIPELINES.keys())}"
    assert translator in ("llm", "auto"), f"Unknown translator '{translator}'. Known: llm, auto"
    assert not (translator == "auto" and (pipeline == "direct" or pipeline == "direct_no_functional")), (
        "translator='auto' requires pipeline='standard' (it consumes refactor_final's DSL output)"
    )
    assert check_order in {"correctness-first", "compliance-first", "always-both"}, \
        f"Unknown check_order={check_order!r}. Known: correctness-first, compliance-first, always-both"

    if plan_enabled:
        assert bundle_dir is None, (
            "plan_enabled=True is incompatible with --bundle-dir"
        )
        assert pipeline == "standard", (
            "plan_enabled=True requires --pipeline=standard"
        )
        assert resume_from is None, (
            "plan_enabled=True is incompatible with --resume-from"
        )

    # --- Step 1: bundle-dir path resolution ---
    bundle_path = None
    bundle_compliance = None
    if bundle_dir is not None:
        bundle_path = Path(bundle_dir).resolve()
        assert bundle_path.exists(), f"bundle dir not found: {bundle_path}"
        if str(bundle_path) not in sys.path:
            sys.path.insert(0, str(bundle_path))
        manifest = json.loads((bundle_path / "manifest.json").read_text())
        assert "compliance" in manifest, (
            f"bundle {bundle_path} manifest.json is missing the `compliance` block; "
            "regenerate it with allowed_ops / banned_patterns / required_ops."
        )
        bundle_compliance = manifest["compliance"]

    # --- Step 2: resolve translate_fn and refactor_system_prompt ---
    if bundle_dir is not None:
        import importlib
        import importlib.util
        # Force reimport in case a previous bundle in the same process polluted sys.modules.
        if "transpiler" in sys.modules:
            del sys.modules["transpiler"]
        transpiler_mod = importlib.import_module("transpiler")
        translate_fn = transpiler_mod.translate
        refactor_system_prompt = (bundle_path / "refactor_system.txt").read_text()

        # Register the bundle's abstraction.py as `step_dsl` so exec'd kernel
        # code can `import step_dsl` (the name the system prompt uses).
        abstraction_path = bundle_path / "abstraction.py"
        assert abstraction_path.exists(), f"bundle missing abstraction.py: {abstraction_path}"
        if "step_dsl" in sys.modules:
            del sys.modules["step_dsl"]
        spec = importlib.util.spec_from_file_location("step_dsl", abstraction_path)
        step_dsl_mod = importlib.util.module_from_spec(spec)
        sys.modules["step_dsl"] = step_dsl_mod
        spec.loader.exec_module(step_dsl_mod)
    else:
        translate_fn = _dsl_to_step_translate
        refactor_system_prompt = None

    # --- Step 5: bundle-dir mode uses only refactor_final + deterministic translate ---
    if bundle_dir is not None:
        # Single-pass: PyTorch -> abstraction (refactor_final) -> transpiler.translate -> graph check.
        # The bundle's abstraction is required to be directly runnable, so the
        # ``dsl`` executor catches abstraction-level bugs; the post-validator
        # (transpiler + simulator) catches transpiler/IR-level bugs separately.
        lowering_passes = [{"name": "refactor_final", "executor": "dsl"}]
        translator_passes = []
        translator = "auto"
    else:
        pipeline_config = PIPELINES[pipeline]
        lowering_passes = pipeline_config["lowering"]
        translator_passes = pipeline_config["translation"]

    config = _load_stepdb_config()
    assert kernel_name in config, f"Kernel '{kernel_name}' not found"
    assert preset in config[kernel_name]["presets"], f"Preset '{preset}' not found"
    dims = config[kernel_name]["presets"][preset]

    # Resolve resume checkpoint — load saved DSL code if resuming
    resume_dsl_code = None
    if resume_from is not None:
        resume_dsl_code = _resolve_resume_dsl(resume_from, kernel_name)
        print(f"Resuming from checkpoint — loaded dsl_code ({len(resume_dsl_code)} chars)")
        print(f"Skipping lowering passes, starting from translation")

    # Pre-compute all tensors externally — functions receive these, can't create their own
    tensors = precompute_tensors(kernel_name, dims)
    print(f"Pre-computed tensors: {sorted(tensors.keys())}")
    print(f"Pipeline: {pipeline} ({len(lowering_passes)} lowering + {len(translator_passes)} translation passes)")

    # Create agents. When translator='auto' we don't need any LLM translator
    # agents -- only the lowering ones (or none, if resuming).
    if translator == "auto":
        agent_passes = [] if resume_dsl_code is not None else lowering_passes
    elif resume_dsl_code is not None:
        agent_passes = translator_passes
    else:
        agent_passes = lowering_passes + translator_passes
    few_shot_examples = resolve_few_shot_examples(few_shot_paths)
    if few_shot_examples:
        print(
            f"Few-shot examples: "
            f"{[ex['kernel_name'] for ex in few_shot_examples]}"
        )
    # --- Step 3: wire refactor_system_prompt into refactor_final agent ---
    pass_agents = {
        p["name"]: make_pass_agent(
            llm_config, p["name"], few_shot_examples=few_shot_examples,
            system_prompt_override=(
                refactor_system_prompt if p["name"] == "refactor_final" else None
            ))
        for p in agent_passes
    }

    # Create judge agents for passes that have one. In bundle_dir mode the
    # judge prompt is templated from the bundle's compliance config so it
    # speaks the abstraction's invented operator surface; non-bundle mode
    # uses the per-pass step_dsl-aware templates.
    if bundle_dir is not None:
        judge_agents = {"refactor_final": make_bundle_judge_agent(llm_config, bundle_compliance)}
    else:
        from src.prompts import _JUDGE_TEMPLATES
        judge_agents = {name: make_judge_agent(llm_config, name) for name in _JUDGE_TEMPLATES}

    # Set up checkpoint directory
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    if checkpoint_dir is None:
        checkpoint_dir = str(Path("checkpoints") / ts)
    else:
        checkpoint_dir = str(Path(checkpoint_dir) / ts)
    ckpt_root = Path(checkpoint_dir) / kernel_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Save run config
    _write(Path(checkpoint_dir) / "config.json", json.dumps({
        "llm_config": {k: v for k, v in llm_config.items() if k != "api_key"},
        "kernel": kernel_name,
        "preset": preset,
        "dims": dims,
        "max_outer": max_outer,
        "max_turns": max_turns,
        "resume_from": resume_from,
        "translator": translator,
        "check_order": check_order,
        "few_shot_paths": list(few_shot_paths) if few_shot_paths else [],
    }, indent=2))

    planner_agent = None
    root_reference = None
    if plan_enabled:
        ref_path = _STEPDB_DIR / config[kernel_name]["problem"]
        root_reference = ref_path.read_text()
        from src.agents import make_planner_agent
        planner_agent = make_planner_agent(llm_config)

    resumed_tree = None
    verified_cache: dict[str, str] = {}
    if resume_planner is not None:
        assert plan_enabled, "--resume-planner requires plan_enabled (no --no-plan)"
        resume_dir = Path(resume_planner)
        assert resume_dir.is_dir(), f"--resume-planner: not a directory: {resume_dir}"
        resumed_tree = _load_tree_from_dir(resume_dir)
        verified_cache = _load_verified_dsls(resume_dir)
        print(f"  Resumed planner state from {resume_dir} "
              f"({sum(1 for _ in resumed_tree.iter_topological())} nodes, "
              f"{len(verified_cache)} cached DSLs)")

    if resume_after_pass1 is not None:
        assert plan_enabled, "--resume-after-pass1 requires plan_enabled (no --no-plan)"
        resume_dir = Path(resume_after_pass1)
        assert resume_dir.is_dir(), (
            f"--resume-after-pass1: not a directory: {resume_dir}")
        # Load+report up front so a missing tree / empty pass1 surfaces here
        # rather than inside the per-outer body where the failure is harder
        # to relate to a CLI mistake.
        _resume_tree = _load_tree_from_dir(resume_dir)
        _resume_dsls = _load_verified_dsls(resume_dir)
        n_nodes = sum(1 for _ in _resume_tree.iter_topological())
        print(f"  Resume-after-pass1 source: {resume_dir} "
              f"({n_nodes} tree node(s), {len(_resume_dsls)} cached DSL(s))")

    # Run all outer iterations in parallel — they are independent attempts.
    # Each outer that has plan_enabled runs its own planner phase + refactor walk
    # (so we get max_outer parallel attempts at the lowering stage, restoring
    # the legacy non-planner behaviour at the right granularity).
    tasks = []
    for i in range(max_outer):
        outer_dir = ckpt_root / f"outer_{i}"
        tasks.append(_run_outer_iteration(
            i, max_outer, outer_dir, kernel_name, dims, tensors,
            pass_agents, judge_agents,
            max_turns, ckpt_root, preset, experience_dir,
            lowering_passes=lowering_passes,
            translator_passes=translator_passes,
            resume_dsl_code=resume_dsl_code,
            translator=translator,
            translate_fn=translate_fn,
            compliance_override=bundle_compliance,
            llm_config=llm_config,
            autotune_options=autotune_options,
            check_order=check_order,
            plan_enabled=plan_enabled,
            planner_agent=planner_agent,
            root_reference=root_reference,
            max_replans=max_replans,
            node_attempts=node_attempts,
            non_root_sequential=non_root_sequential,
            max_plan_depth=max_plan_depth,
            resumed_tree=resumed_tree,
            verified_cache=verified_cache,
            resume_planner_dir=Path(resume_planner) if resume_planner else None,
            resume_after_pass1_dir=Path(resume_after_pass1) if resume_after_pass1 else None,
            stateless_refactor=stateless_refactor,
            few_shot_examples=few_shot_examples,
        ))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Normalize exceptions into the same failure-result shape so one outer's
    # crash doesn't take down the rest of the run.
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            err_msg = f"{type(r).__name__}: {r}"
            print(f"[outer_{i}] CRASHED — {err_msg}")
            results[i] = {
                "success": False,
                "outer_iteration": i,
                "outer_iterations": max_outer,
                "total_tool_calls": 0,
                "total_tokens": 0,
                "cycle_count": None,
                "final_diagnosis": err_msg,
            }

    per_outer = [
        {
            "outer": i,
            "success": bool(r["success"]),
            "autotune": r.get("autotune"),
        }
        for i, r in enumerate(results)
    ]

    # Return first success, or the last failure
    chosen = next((r for r in results if r["success"]), results[-1])
    chosen["per_outer"] = per_outer
    chosen["total_tokens"] = sum(r.get("total_tokens", 0) for r in results)
    _write(ckpt_root / "result.json", json.dumps(chosen, indent=2, default=str))
    return chosen


async def _run_outer_iteration(
    i: int, max_outer: int, outer_dir: Path,
    kernel_name: str, dims: dict, tensors: dict,
    pass_agents: dict, judge_agents: dict,
    max_turns: int,
    ckpt_root: Path, preset: str, experience_dir: str,
    llm_config: dict,
    **kwargs,
) -> dict:
    """Run one outer iteration; surface crashes immediately, never raise.

    Wraps ``_run_outer_iteration_body`` so an unhandled exception in any
    phase is logged to ``outer_dir/log.txt`` and printed as ``[outer_N]
    CRASHED — <last line>`` *as soon as it happens*, instead of being
    silently held by ``asyncio.gather(return_exceptions=True)`` until all
    siblings finish. Sibling outers are unaffected (we still return a
    failure-shaped dict rather than raising).

    Also owns the per-outer log file and a write lock. Pass1 spawns many
    parallel attempts that all funnel ``log()`` calls into this single file;
    without serialization, the crash handler's multi-line traceback can
    interleave with sibling status lines and become unreadable (observed in
    checkpoints/2026-05-10-005659/.../outer_0/log.txt). The lock guards every
    write+flush so each line — including the full crash dump — is atomic
    against any concurrent writer.
    """
    outer_dir.mkdir(parents=True, exist_ok=True)
    log_path = outer_dir / "log.txt"
    log_file = open(log_path, "w")
    log_lock = threading.Lock()

    def log(msg: str) -> None:
        with log_lock:
            log_file.write(msg + "\n")
            log_file.flush()

    try:
        try:
            return await _run_outer_iteration_body(
                i, max_outer, outer_dir,
                kernel_name, dims, tensors,
                pass_agents, judge_agents, max_turns,
                ckpt_root, preset, experience_dir, llm_config,
                _log=log,
                **kwargs,
            )
        except Exception:
            err = traceback.format_exc()
            # Atomic multi-line dump: hold the lock for the whole traceback so
            # parallel pass1 writers can't interleave into it.
            with log_lock:
                log_file.write(f"\n--- CRASHED ---\n{err}\n")
                log_file.flush()
            last_line = err.strip().splitlines()[-1] if err.strip() else "<unknown>"
            print(f"[outer_{i}] CRASHED — {last_line}", flush=True)
            # Also persist a standalone crash file so the traceback survives
            # even if the log gets rotated/truncated by something else.
            (outer_dir / "crash.txt").write_text(err)
            return {
                "success": False,
                "outer_iteration": i,
                "outer_iterations": max_outer,
                "total_tool_calls": 0,
                "total_tokens": 0,
                "cycle_count": None,
                "final_diagnosis": last_line,
            }
    finally:
        log_file.close()


async def _run_outer_iteration_body(
    i: int, max_outer: int, outer_dir: Path,
    kernel_name: str, dims: dict, tensors: dict,
    pass_agents: dict, judge_agents: dict,
    max_turns: int,
    ckpt_root: Path, preset: str, experience_dir: str,
    llm_config: dict,
    lowering_passes: list = None, translator_passes: list = None,
    resume_dsl_code: str = None,
    translator: str = "llm",
    translate_fn=None,
    compliance_override: dict | None = None,
    autotune_options: dict = None,
    check_order: str = "correctness-first",
    plan_enabled: bool = False,
    planner_agent=None,
    root_reference: str | None = None,
    max_replans: int = 3,
    node_attempts: int = 1,
    non_root_sequential: bool = True,
    max_plan_depth: int | None = 3,
    resumed_tree=None,
    verified_cache: dict[str, str] | None = None,
    resume_planner_dir: Path | None = None,
    resume_after_pass1_dir: Path | None = None,
    stateless_refactor: bool = False,
    few_shot_examples: list | None = None,
    _log=None,
) -> dict:
    """Run a single outer iteration of the pipeline (lowering + translation).

    All detailed output goes to outer_dir/log.txt. Only summary lines go to terminal.

    Args:
        resume_dsl_code: If set, skip all lowering passes and use this as the
            DSL code for translation. Used when resuming from a checkpoint where
            refactor_final succeeded but translate failed.
        translate_fn: DSL→STeP translator callable. Defaults to
            ``_dsl_to_step_translate``; bundle-dir mode passes
            ``transpiler.translate`` from the bundle.
        _log: provided by ``_run_outer_iteration``. The wrapper owns the log
            file + write-lock so the crash handler can atomically dump
            tracebacks alongside the body's own writes. Asserted non-None so
            we never silently fall back to an unsynchronized writer.
    """
    assert _log is not None, (
        "_run_outer_iteration_body requires log callable from the wrapper"
    )
    if translate_fn is None:
        translate_fn = _dsl_to_step_translate
    log = _log
    log_path = outer_dir / "log.txt"

    tag = f"[outer_{i}]"
    print(f"{tag} Started — log: {log_path}")
    log(f"--- Outer iteration {i + 1}/{max_outer} ---")

    if lowering_passes is None:
        lowering_passes = LOWERING_PASSES
    if translator_passes is None:
        translator_passes = TRANSLATOR_PASSES

    dsl_code = None  # output of refactor_final, used as translation guide
    outer_total_tokens = 0

    # ============================================================
    # Phase 1: Lowering pass (refactor_final). Skipped on resume and on
    # direct pipelines (which have lowering_passes == []).
    # ============================================================
    if resume_after_pass1_dir is not None:
        # Skip planner + pass1 entirely. Copy plan/ and pass1/ from the source
        # outer dir so this run is self-contained, then load tree + DSLs and
        # run only pass2 composition. Failures here exit the iteration cleanly
        # — no replan loop (replan would force pass1 to re-run, contradicting
        # the resume contract).
        for sub in ("plan", "pass1"):
            src = resume_after_pass1_dir / sub
            if src.is_dir():
                shutil.copytree(src, outer_dir / sub, dirs_exist_ok=True)
        log(f"  Resume-after-pass1: copied plan/ + pass1/ from {resume_after_pass1_dir}")

        tree = _load_tree_from_dir(outer_dir)
        pass1_dsls = _load_verified_dsls(outer_dir)
        # Lenient: a planner-tree node may have been inlined by its parent
        # (no PASS DSL on disk by design). Only nodes whose parent's verified
        # DSL still calls them are required for composition.
        required = _required_nodes(tree, pass1_dsls)
        missing = [p for p in required if p not in pass1_dsls]
        if missing:
            msg = (f"resume-after-pass1: {len(missing)} node(s) without cached "
                   f"PASS DSL — cannot compose. Missing: {missing}")
            log(f"  -> FAILED: {msg}")
            print(f"{tag} FAILED — {msg}")
            return {
                "success": False,
                "outer_iteration": i,
                "outer_iterations": max_outer,
                "total_tool_calls": 0,
                "total_tokens": 0,
                "cycle_count": None,
                "final_diagnosis": msg,
            }

        log(f"  Running pass2 compose on {len(pass1_dsls)} cached DSL(s)")
        pass2 = _pass2_compose(
            tree=tree, pass1_dsls=pass1_dsls,
            dims=dims, tensors=tensors,
            root_kernel=kernel_name, ckpt_root=outer_dir, log=log,
        )
        if not pass2["success"]:
            msg = f"pass2 mismatch on resume: {pass2.get('exec_log', '')[:200]}"
            log(f"  -> FAILED: {msg}")
            print(f"{tag} FAILED — pass2 mismatch on resume")
            return {
                "success": False,
                "outer_iteration": i,
                "outer_iterations": max_outer,
                "total_tool_calls": 0,
                "total_tokens": 0,
                "cycle_count": None,
                "final_diagnosis": msg,
            }
        dsl_code = pass2["root_dsl"]
        _write(outer_dir / "dsl_code.py", dsl_code)
        log(f"  -> Pass2 OK on resume ({len(dsl_code)} chars)")
        print(f"{tag} Pass2 OK on resume — proceeding to translation")
    elif resume_dsl_code is not None:
        dsl_code = resume_dsl_code
        _write(outer_dir / "dsl_code.py", dsl_code)
        log(f"  Resumed from checkpoint — using saved dsl_code ({len(dsl_code)} chars)")
        print(f"{tag} Resumed — skipping lowering")
    elif plan_enabled and lowering_passes:
        assert planner_agent is not None and root_reference is not None, (
            "plan_enabled requires planner_agent and root_reference"
        )

        if resume_planner_dir is not None:
            for sub in ("plan", "pass1"):
                src = resume_planner_dir / sub
                if src.is_dir():
                    shutil.copytree(src, outer_dir / sub, dirs_exist_ok=True)
            log(f"  Copied resumed planner artifacts from {resume_planner_dir}")

        def _agent_factory(_few_shot):
            return pass_agents["refactor_final"]
        _agent_factory.__planner_agent__ = planner_agent
        _agent_factory.__refactor_judge_agent__ = judge_agents.get("refactor_final")
        _agent_factory.__llm_config__ = llm_config
        _agent_factory.__few_shot_examples__ = few_shot_examples

        log(f"  Planner phase (plan + per-node refactor)")
        print(f"{tag} Planner phase starting")
        from src.prompts import _get_precompute_source
        plan_result = await _run_planner_phase(
            root_reference=root_reference, dims=dims,
            root_kernel=kernel_name, ckpt_root=outer_dir,
            agent_factory=_agent_factory, max_turns=max_turns,
            log=log, max_replans=max_replans,
            node_attempts=node_attempts,
            non_root_sequential=non_root_sequential,
            max_plan_depth=max_plan_depth,
            precompute_source=_get_precompute_source(kernel_name),
            tensors=precompute_tensors(kernel_name, dims),
            resumed_tree=resumed_tree,
            verified_cache=verified_cache,
            translate_fn=(translate_fn if translator == "auto" else None),
            stateless_refactor=stateless_refactor,
        )
        if not plan_result["success"]:
            log(f"  -> Planner phase FAILED at {plan_result.get('failing_node')}")
            print(f"{tag} Planner phase FAILED")
            return {
                "success": False,
                "outer_iteration": i,
                "outer_iterations": max_outer,
                "total_tool_calls": 0,
                "total_tokens": outer_total_tokens,
                "cycle_count": None,
                "final_diagnosis": (
                    f"Planner phase failed at {plan_result.get('failing_node')}"
                ),
            }
        dsl_code = plan_result["root_dsl"]
        _write(outer_dir / "dsl_code.py", dsl_code)
        log(f"  -> Planner phase OK")
        print(f"{tag} Planner phase OK")
    elif lowering_passes:
        assert len(lowering_passes) == 1 and lowering_passes[0]["name"] == "refactor_final", (
            f"phase 1 expects exactly one refactor_final pass, got {[p['name'] for p in lowering_passes]}"
        )
        pass_info = lowering_passes[0]
        pass_name = pass_info["name"]
        executor = pass_info["executor"]

        # When using deterministic translation, gate refactor_final on the
        # translator: if translation fails, treat it as a refactor error so
        # the model fixes the DSL until it lowers cleanly into STeP IR.
        post_validator = None
        if translator == "auto":
            post_validator = _make_translation_post_validator(
                kernel_name, dims, tensors, log,
                translate_fn=translate_fn,
            )

        log(f"  Lowering pass: {pass_name}")
        pass_result = await _run_pass_loop(
            pass_agents[pass_name], pass_name, kernel_name, dims, max_turns,
            ckpt_dir=outer_dir,
            executor=executor,
            tensors=tensors,
            log=log,
            judge_agent=judge_agents.get(pass_name),
            post_validator=post_validator,
            compliance_override=compliance_override,
            check_order=check_order,
            stateless=stateless_refactor,
        )
        outer_total_tokens += pass_result.get("total_tokens", 0)
        if pass_result["success"]:
            dsl_code = pass_result["code"]
            _write(outer_dir / "dsl_code.py", dsl_code)
            log(f"  -> {pass_name} OK")
            log(f"Lowering pipeline succeeded")
            print(f"{tag} Lowering OK")
        else:
            log(f"  -> {pass_name} FAILED")
            log(f"Lowering pipeline failed")
            print(f"{tag} Lowering FAILED")
            return {
                "success": False,
                "outer_iteration": i,
                "outer_iterations": max_outer,
                "total_tool_calls": 0,
                "total_tokens": outer_total_tokens,
                "cycle_count": None,
            }
    else:
        log(f"No lowering passes (direct pipeline)")
        print(f"{tag} Direct pipeline — skipping lowering")

    # ============================================================
    # Phase 2: STeP translation passes
    # The DSL code from the refactor pass is passed as a translation guide.
    # Each DSL call maps 1:1 to a STeP node, making translation mechanical.
    # ============================================================
    translated_code = dsl_code
    translation_ok = True

    if translator == "auto":
        assert dsl_code is not None, (
            "translator='auto' requires DSL code from refactor_final (or --resume); "
            "got None — lowering must run before deterministic translation."
        )
        det_result = _run_deterministic_translate(
            dsl_code, kernel_name, dims, tensors, outer_dir, log,
            translate_fn=translate_fn,
        )
        if det_result["success"]:
            translated_code = det_result["code"]
        else:
            translation_ok = False
            print(f"{tag} Translation FAILED (deterministic)")
        # Skip the LLM translation loop entirely.
        translator_passes = []

    for pass_info in translator_passes:
        pass_name = pass_info["name"]
        executor = pass_info["executor"]

        log(f"  Translation pass: {pass_name} (executor={executor})")
        pass_result = await _run_pass_loop(
            pass_agents[pass_name], pass_name, kernel_name, dims, max_turns,
            ckpt_dir=outer_dir,
            prev_code=translated_code,
            executor=executor,
            tensors=tensors,
            log=log,
            judge_agent=judge_agents.get(pass_name),
            dsl_code=dsl_code,
            check_order=check_order,
        )
        outer_total_tokens += pass_result.get("total_tokens", 0)
        if pass_result["success"]:
            translated_code = pass_result["code"]
            log(f"  -> {pass_name} OK")
        else:
            log(f"  -> {pass_name} FAILED")
            print(f"{tag} Translation FAILED at {pass_name}")
            translation_ok = False
            break

    if translation_ok:
        # Verify the final code actually works as a build_graph. The
        # try/except is scoped tightly around _run_graph_correctness — that
        # call can legitimately raise on user code, but everything past it
        # is harness logic; surfacing harness bugs loud is worth more than
        # smoothing them into a kernel "fail".
        final_code = translated_code
        log(f"Translation pipeline completed — verifying final graph...")
        try:
            graph_result = _run_graph_correctness(final_code, kernel_name, dims, tensors)
        except Exception:
            err_msg = traceback.format_exc().splitlines()[-1]
            log(f"-> ERROR: {err_msg}")
            translation_ok = False
            graph_result = None

        if graph_result is not None and _overall_match(graph_result):
            log(f"-> PASS (graph verified)")
            print(f"{tag} SUCCESS")
            result = _build_success_result(i, 0,
                                           {"code": final_code, "tool_outputs": []},
                                           [], dsl_code)
            result["total_tokens"] = outer_total_tokens
            if autotune_options is not None:
                log(f"{tag} starting autotune...")
                print(f"{tag} starting autotune...")
                result["autotune"] = await _run_outer_autotune(
                    outer_dir=outer_dir,
                    kernel_name=kernel_name,
                    preset=preset,
                    llm_config=llm_config,
                    autotune_options=autotune_options,
                    log=log,
                    tag=tag,
                )
            return result
        elif graph_result is not None:
            log(f"-> FAIL: {graph_result.splitlines()[0]}")
            translation_ok = False

    if not translation_ok:
        log(f"Translation pipeline failed")
        print(f"{tag} Translation FAILED")

    return {
        "success": False,
        "outer_iteration": i,
        "outer_iterations": max_outer,
        "total_tool_calls": 0,
        "total_tokens": outer_total_tokens,
        "cycle_count": None,
        "tiled_code": dsl_code,
    }
