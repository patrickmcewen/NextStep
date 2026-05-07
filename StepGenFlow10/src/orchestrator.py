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

import asyncio
import contextlib
import io
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

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
                         build_subdivide_user_prompt,
                         build_pass_system_prompt,
                         build_subdivide_results_block,
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
from src import subdivide as _sub_mod


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


def _compare_single(result, gold, label):
    """Compare a single result tensor against a single gold tensor. Returns formatted string."""
    import torch

    if gold.shape != result.shape:
        return f"SHAPE MISMATCH: gold {tuple(gold.shape)} vs {label} {tuple(result.shape)}\nmatch=False"

    max_err = (gold - result).abs().max().item()
    rel_err = max_err / (gold.abs().max().item() + 1e-12)
    match = rel_err < 1e-5

    out = f"match={match}\nmax_abs_err={max_err:.2e}\nrel_err={rel_err:.2e}\noutput_shape={tuple(result.shape)}"
    if not match:
        diff = (gold - result).abs()
        worst_multi = torch.unravel_index(diff.argmax(), diff.shape)
        out += (
            f"\nworst_error_index={tuple(i.item() for i in worst_multi)}"
            f"\ngold_value={gold[worst_multi].item():.6e}"
            f"\n{label}_value={result[worst_multi].item():.6e}"
        )
    return out


def _compare_against_gold(result, kernel_name, dims, label="result"):
    """Compare a result (tensor or tuple of tensors) against gold reference. Returns formatted string."""
    gold = _get_gold(kernel_name, dims)
    if isinstance(gold, tuple) or isinstance(result, tuple):
        if not (isinstance(gold, tuple) and isinstance(result, tuple)):
            return (
                f"match=False\n"
                f"arity mismatch: gold is {'tuple' if isinstance(gold, tuple) else 'single'}, "
                f"candidate is {'tuple' if isinstance(result, tuple) else 'single'}"
            )
        if len(gold) != len(result):
            return (
                f"match=False\n"
                f"arity mismatch: gold has {len(gold)} elements, "
                f"candidate has {len(result)} elements"
            )
        elem_reports = []
        all_match = True
        for i, (g, r) in enumerate(zip(gold, result)):
            elem_label = f"{label}[{i}]"
            elem_report = _compare_single(r, g, elem_label)
            if "match=False" in elem_report:
                all_match = False
                elem_reports.append(f"## element[{i}]\n{elem_report}")
        if all_match:
            return f"match=True\nall {len(gold)} tuple elements matched"
        return "match=False\n" + "\n\n".join(elem_reports)
    return _compare_single(result, gold, label)


def _run_dsl_correctness(code, kernel_name, dims, tensors):
    """Run a refactor-pass candidate against gold via the DSL executor.

    The DSL surface is directly runnable (the standalone ``step_dsl`` module
    or the bundle's mounted abstraction), so we exec the candidate as
    ``tiled_reference(dims, tensors)`` and compare its output to gold.
    """
    result = _exec_dsl_ref(code, dims, tensors)
    return _compare_against_gold(result, kernel_name, dims, "dsl")


def _run_graph_correctness(code, kernel_name, dims, tensors):
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
        "allowed_torch": set(),
        "allowed_F": set(),
        "banned_patterns": [
            (".unsqueeze(", "use promote(x, rank) or promote_outer(x)"),
            (".squeeze(",   "use flatten(x, rank, rank) or accum_retile_row/col"),
            (".expand(",    "use expand_ref(x, ref) or repeat_static(x, factor)"),
            (".sum(",       "use accum_add(x, rank=1) or unary_rowwise_sum(x)"),
            (".prod(",      "use accum_mul(x, rank=1)"),
            ("torch.matmul", "use binary_matmul(a, b)"),
            ("torch.exp",    "use unary_exp(x)"),
            ("torch.rsqrt",  "use unary_rsqrt(x)"),
            ("F.silu",       "use unary_silu(x)"),
            ("out_shape_tiled=(1,)",
             "NEVER load as one giant tile — use proper streaming: out_shape_tiled=(B//tile_n,) or similar"),
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


def _check_banned_ops(code: str, pass_name: str) -> list[str]:
    """Check whether ``code`` complies with this pass's output constraints.

    Returns a list of violation messages; empty means compliant. Each pass's
    rules are independent — there is no cumulative inheritance. For translation
    passes, only the ``build_graph`` body is checked so scaffold code (DSL
    functions, functional.py) may use torch.* internally without false
    positives.
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
    for pattern, fix in rules["banned_patterns"]:
        if re.search(r'\b' + re.escape(pattern), code):
            violations.append(f"- `{pattern}` still present — {fix}")

    for required in rules["required_ops"]:
        if required not in code:
            violations.append(f"- `{required}` missing — this pass must introduce {required} nodes")

    # Deduplicate while preserving order
    return list(dict.fromkeys(violations))


def _check_bundle_compliance(code: str, compliance: dict) -> list[str]:
    """Bundle-mode compliance checker driven by the bundle's manifest config.

    Mirrors ``_check_banned_ops`` but pulls allowed/banned/required from the
    bundle's compliance dict rather than the hard-coded step_dsl tables.
    Empty allowlist = allowlist disabled.
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

    for name in required:
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

        if "match=True" not in result:
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

    success = "match=True" in result
    _write(turn_dir / "status.txt", "PASS" if success else "MISMATCH")
    log(f"  -> deterministic translate {'OK' if success else 'FAILED'}")
    return {"success": success, "code": step_code}


async def _gate_correctness(code, kernel_name, dims, tensors, executor,
                            turn_dir: Path, log) -> tuple[_GateResult, str]:
    """Run check_correctness with stdout captured for shape trace.

    Returns (gate_result, shape_trace). The shape trace is captured even on
    failure so the caller can append it to feedback for the LLM.
    """
    check_correctness = _CORRECTNESS_CHECKERS[executor]
    log(f"      Running correctness check ({executor})...")
    _trace_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(_trace_buf):
            result = check_correctness(code, kernel_name, dims, tensors)
        shape_trace = _trace_buf.getvalue()
        if shape_trace:
            _write(turn_dir / "shape_trace.txt", shape_trace)
        _write(turn_dir / "correctness_result.txt", result)

        if "match=True" in result:
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
                           *, correctness_verified: bool) -> _GateResult:
    """Regex compliance check.

    On failure, on `pass_name == "refactor_final"` with a non-None judge_agent,
    also runs the LLM judge for richer line-specific feedback (parity with the
    pre-refactor inline carve-out).
    """
    if compliance_override is not None:
        violations = _check_bundle_compliance(code, compliance_override)
    else:
        violations = _check_banned_ops(code, pass_name)

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


async def _run_pass_loop(agent, pass_name, kernel_name, dims, max_turns,
                         ckpt_dir: Path, *, executor: str, tensors: dict,
                         prev_code=None, log=print,
                         judge_agent=None, dsl_code=None,
                         post_validator=None,
                         compliance_override: dict | None = None,
                         check_order: str,
                         prebuilt_user_prompt: str | None = None,
                         subdivide_options=None,
                         registry: list | None = None,
                         depth: int = 0,
                         counter=None,
                         llm_config: dict | None = None):
    """Run a single pass agent (lowering or translator).

    ``post_validator`` is an optional ``(code, turn_dir) -> str | None`` callable
    that runs after correctness + regex compliance + judge all pass. Returning a
    string treats the turn as failed and feeds that string back into the next
    user prompt — this is how deterministic translation surfaces errors back to
    the refactor pass.

    Returns dict with success, code.
    """
    if prebuilt_user_prompt is not None:
        user_prompt = prebuilt_user_prompt
    else:
        user_prompt = build_pass_user_prompt(
            pass_name, kernel_name, dims,
            prev_code=prev_code,
            tensors=tensors,
            dsl_code=dsl_code,
            subdivide_results_block=(
                build_subdivide_results_block(registry) if registry else ""
            ),
        )
    conversation = [{"role": "user", "content": user_prompt}]

    pass_dir = ckpt_dir / pass_name
    # Save the system prompt actually used by the agent (agent.instructions
    # already contains any {few_shot_examples} substitutions from make_pass_agent).
    _write(pass_dir / "system_prompt.txt", agent.instructions)

    last_code = None
    success = False
    total_tokens = 0

    for turn in range(max_turns):
        turn_dir = pass_dir / f"turn_{turn}"
        log(f"    [{pass_name}] Turn {turn + 1}/{max_turns}...")

        last_user_msg = conversation[-1]["content"] if conversation[-1]["role"] == "user" else ""
        _write(turn_dir / "user_prompt.txt", last_user_msg)

        run_result = await Runner.run(agent, conversation)
        if run_result.context_wrapper.usage is not None:
            total_tokens += run_result.context_wrapper.usage.total_tokens
        assistant_text = run_result.final_output or ""
        conversation.append({"role": "assistant", "content": assistant_text})
        _write(turn_dir / "response.txt", assistant_text)
        reasoning = _reasoning_text(run_result)
        if reasoning:
            _write(turn_dir / "reasoning.txt", reasoning)

        code = _extract_code(assistant_text)
        if not code:
            log(f"      No code block found ({len(assistant_text)} chars). Retrying.")
            _write(turn_dir / "status.txt", "NO_CODE_EXTRACTED")
            conversation.append({"role": "user", "content":
                "Your response did not contain extractable Python. Either wrap "
                "the implementation in a ```python ... ``` fence, OR make the "
                "entire response valid Python source with no surrounding prose "
                "(comments are fine). Your previous response failed both checks."
            })
            continue

        last_code = code
        _write(turn_dir / "extracted_code.py", code)
        log(f"      Extracted code: {len(code)} chars")

        # ----- Subdivide directive branch -----
        if subdivide_options is not None and registry is not None and counter is not None:
            assert pass_name == "refactor_final", (
                "subdivide is only supported for refactor_final"
            )
            from src.subdivide import parse_directive, NotADirective
            try:
                parsed_sub_tasks = parse_directive(
                    code, registry=registry, depth=depth,
                    counter=counter, options=subdivide_options,
                )
            except NotADirective:
                parsed_sub_tasks = None
            except AssertionError as exc:
                log(f"      Directive validation failed: {exc}")
                _write(turn_dir / "status.txt", "DIRECTIVE_INVALID")
                feedback = (
                    f"## Subdivide directive: validation failed\n{exc}\n\n"
                    f"Fix the directive or write tiled_reference instead."
                )
                conversation.append({"role": "user", "content": feedback})
                continue

            if parsed_sub_tasks is not None:
                from src import subdivide as _sub_mod
                log(f"      Directive with {len(parsed_sub_tasks)} sub-task(s); dispatching...")
                directive_outcome = await _sub_mod.dispatch_directive(
                    parsed_sub_tasks, dims=dims, parent_tensors=tensors,
                    registry=registry, depth=depth, counter=counter,
                    options=subdivide_options, ckpt_dir=turn_dir,
                    llm_config=llm_config or {},
                    log=log,
                )
                if directive_outcome["success"]:
                    _write(turn_dir / "status.txt", "DIRECTIVE_SUCCESS")
                    names = [v.name for v in registry[-len(parsed_sub_tasks):]]
                    feedback = (
                        f"## Subdivide: sub-task(s) {names} verified\n\n"
                        f"The verified DSL forms are now shown above as "
                        f"reference material under '## Verified sub-task results'. "
                        f"Adapt them as needed and emit `tiled_reference` next."
                    )
                else:
                    _write(turn_dir / "status.txt", "DIRECTIVE_FAILED")
                    feedback = directive_outcome["feedback"]

                rendered = build_subdivide_results_block(registry)
                if rendered:
                    feedback = rendered + "\n\n" + feedback
                conversation.append({"role": "user", "content": feedback})
                continue
        # ----- End subdivide branch -----

        # ----- Per-turn gate cascade -----
        assert check_order in ("correctness-first", "compliance-first"), \
            f"Unknown check_order={check_order!r}"

        if check_order == "correctness-first":
            gate_order = ["correctness", "compliance", "judge", "post_validator"]
            correctness_verified = True
        else:  # "compliance-first"
            gate_order = ["compliance", "judge", "correctness", "post_validator"]
            correctness_verified = False

        shape_trace = ""
        turn_feedback = None
        turn_status = None
        success_this_turn = False

        try:
            for gate_name in gate_order:
                if gate_name == "correctness":
                    res, shape_trace = await _gate_correctness(
                        code, kernel_name, dims, tensors, executor,
                        turn_dir, log)
                    correctness_verified = (res.feedback is None)
                elif gate_name == "compliance":
                    res = await _gate_compliance(
                        code, pass_name, compliance_override, judge_agent,
                        tensors, turn_dir, log,
                        correctness_verified=correctness_verified)
                elif gate_name == "judge":
                    res = await _gate_judge(
                        judge_agent, code, tensors, turn_dir, log,
                        correctness_verified=correctness_verified)
                else:  # "post_validator"
                    res = _gate_post_validator(post_validator, code, turn_dir, log)
                total_tokens += res.tokens
                if res.feedback is not None:
                    turn_feedback = res.feedback
                    turn_status = res.status
                    break
            else:
                turn_status = "PASS"
                success_this_turn = True
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
        conversation.append({"role": "user", "content": feedback})

    return {"success": success, "code": last_code, "total_tokens": total_tokens}


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
    max_subdivide_turns: int = 8,
    max_subdivides_per_outer: int = 5,
    max_subdivide_depth: int = 2,
    subdivide_enabled: bool = True,
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
    if subdivide_enabled:
        assert bundle_dir is None, (
            "subdivide is not supported in bundle mode in v1; pass --no-subdivide"
        )
        assert pipeline == "standard", (
            f"subdivide is only supported with --pipeline=standard in v1; "
            f"got --pipeline={pipeline}. Pass --no-subdivide to disable."
        )
    assert pipeline in PIPELINES, f"Unknown pipeline '{pipeline}'. Known: {sorted(PIPELINES.keys())}"
    assert translator in ("llm", "auto"), f"Unknown translator '{translator}'. Known: llm, auto"
    assert not (translator == "auto" and (pipeline == "direct" or pipeline == "direct_no_functional")), (
        "translator='auto' requires pipeline='standard' (it consumes refactor_final's DSL output)"
    )
    assert check_order in {"correctness-first", "compliance-first"}, \
        f"Unknown check_order={check_order!r}. Known: correctness-first, compliance-first"

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
    if checkpoint_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        checkpoint_dir = str(Path("checkpoints") / ts)
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

    # Run all outer iterations in parallel — they are independent attempts
    subdivide_options = (
        _sub_mod.SubdivideOptions(
            max_subdivide_turns=max_subdivide_turns,
            max_subdivides_per_outer=max_subdivides_per_outer,
            max_subdivide_depth=max_subdivide_depth,
        ) if subdivide_enabled else None
    )

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
            subdivide_options=subdivide_options,
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
    lowering_passes: list = None, translator_passes: list = None,
    resume_dsl_code: str = None,
    translator: str = "llm",
    translate_fn=None,
    compliance_override: dict | None = None,
    autotune_options: dict = None,
    check_order: str = "correctness-first",
    subdivide_options=None,
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
    """
    if translate_fn is None:
        translate_fn = _dsl_to_step_translate
    log_path = outer_dir / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    def log(msg):
        log_file.write(msg + "\n")
        log_file.flush()

    tag = f"[outer_{i}]"
    print(f"{tag} Started — log: {log_path}")
    log(f"--- Outer iteration {i + 1}/{max_outer} ---")

    if lowering_passes is None:
        lowering_passes = LOWERING_PASSES
    if translator_passes is None:
        translator_passes = TRANSLATOR_PASSES

    dsl_code = None  # output of refactor_final, used as translation guide
    outer_total_tokens = 0

    sub_registry = []
    sub_counter = _sub_mod.SubdivideCounter() if subdivide_options is not None else None

    # ============================================================
    # Phase 1: Lowering pass (refactor_final). Skipped on resume and on
    # direct pipelines (which have lowering_passes == []).
    # ============================================================
    if resume_dsl_code is not None:
        dsl_code = resume_dsl_code
        _write(outer_dir / "dsl_code.py", dsl_code)
        log(f"  Resumed from checkpoint — using saved dsl_code ({len(dsl_code)} chars)")
        print(f"{tag} Resumed — skipping lowering")
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
            llm_config=llm_config,
            subdivide_options=subdivide_options,
            registry=sub_registry if subdivide_options is not None else None,
            depth=0,
            counter=sub_counter,
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
            log_file.close()
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

        if graph_result is not None and "match=True" in graph_result:
            log(f"-> PASS (graph verified)")
            print(f"{tag} SUCCESS")
            result = _build_success_result(i, 0,
                                           {"code": final_code, "tool_outputs": []},
                                           [], dsl_code)
            result["total_tokens"] = outer_total_tokens
            result["subdivides"] = [v.name for v in sub_registry] if subdivide_options is not None else []
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
            log_file.close()
            return result
        elif graph_result is not None:
            log(f"-> FAIL: {graph_result.splitlines()[0]}")
            translation_ok = False

    if not translation_ok:
        log(f"Translation pipeline failed")
        print(f"{tag} Translation FAILED")

    log_file.close()
    return {
        "success": False,
        "outer_iteration": i,
        "outer_iterations": max_outer,
        "total_tool_calls": 0,
        "total_tokens": outer_total_tokens,
        "cycle_count": None,
        "tiled_code": dsl_code,
    }


# ---------------------------------------------------------------------------
# Subdivide runner wiring (module-load-time)
# ---------------------------------------------------------------------------


async def _subdivide_pass_loop_runner(*, agent, name, kernel_name, dims,
                                       sub_tensors, sub_dir, options,
                                       sub_reference_source, preamble_source,
                                       registry, depth, counter,
                                       llm_config, log):
    """Bridge from subdivide module → _run_pass_loop with sub-task context."""
    user_prompt = build_subdivide_user_prompt(
        name=name,
        sub_reference_source=sub_reference_source,
        preamble_source=preamble_source,
        dims=dims,
        sub_tensors=sub_tensors,
    )
    return await _run_pass_loop(
        agent, "refactor_final", kernel_name, dims,
        max_turns=options.max_subdivide_turns,
        ckpt_dir=sub_dir,
        executor="dsl",
        tensors=sub_tensors,
        log=log,
        judge_agent=None,
        post_validator=None,
        compliance_override=None,
        check_order="correctness-first",
        prebuilt_user_prompt=user_prompt,
        subdivide_options=options,
        registry=registry,
        depth=depth,
        counter=counter,
        llm_config=llm_config,
    )


def _make_subdivide_pass_agent(llm_config: dict, options: _sub_mod.SubdivideOptions):
    """Build a refactor_final agent with a system prompt formatted for the
    given subdivide options (so the agent's prompt mentions the actual depth/cap).
    """
    system_prompt = build_pass_system_prompt(
        "refactor_final",
        few_shot_examples=None,
        subdivide_options={
            "max_subdivide_depth": options.max_subdivide_depth,
            "max_subdivides_per_outer": options.max_subdivides_per_outer,
        },
    )
    return make_pass_agent(
        llm_config, "refactor_final",
        few_shot_examples=None,
        system_prompt_override=system_prompt,
    )


_sub_mod.set_runners(
    pass_loop_runner=_subdivide_pass_loop_runner,
    pass_agent_factory=_make_subdivide_pass_agent,
)
