"""Orchestrator for StepGenFlow5 — progressive translation pipeline.

Two-phase pipeline:
  Phase 1 (lowering): tiler -> router -> retiler -> canonicalize
    Each pass outputs tiled_reference(dims) -> torch.Tensor, validated against gold.
    Canonicalize enforces single-assignment form with only canonical ops.
  Phase 2 (translation): single translate pass (DSL -> STeP graph)
    Takes DSL-refactored code and translates all DSL calls 1:1 into STeP graph nodes, outputs
    build_graph(dims) -> (graph, output_op), validated via emulator against gold.

Checkpoint structure:
  checkpoints/<timestamp>/<kernel>/outer_<N>/<pass_name>/turn_<M>/...
  checkpoints/<timestamp>/<kernel>/outer_<N>/analyst/...
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

import yaml

# Enable shape-trace logging in step_dsl ops before tools.py exec's the scaffold.
# Each op prints its input/output shapes; the orchestrator captures the trace
# and feeds it to the LLM alongside any error.
os.environ.setdefault("STEP_DSL_TRACE", "1")

from src.dsl_to_step import translate as _dsl_to_step_translate
from agents import Runner

from src.agents import make_diagnostician_agent, make_judge_agent, make_pass_agent
from src.prompts import (LOWERING_PASSES, TRANSLATOR_PASSES,
                         DIRECT_TRANSLATOR_PASSES, PIPELINES,
                         build_pass_user_prompt,
                         _format_tensors_description,
                         resolve_few_shot_examples)
from src.tools import (_exec_build_graph, _exec_tiled_ref, _exec_hybrid_ref,
                       _exec_dsl_ref,
                       _validate_functional_mod, enhance_emulator_error)

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
    """Extract the last python code block from LLM output."""
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else ""


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

_GOLD_CACHE: dict = {}


def _get_gold(kernel_name, dims):
    """Return the cached gold tensor for (kernel_name, dims), computing once.

    `compute_gold(dims)` is deterministic (fixed seeds inside the reference) and
    only depends on `dims`, so the result is safe to memoize for the lifetime of
    the process.  This avoids re-allocating multi-GiB reference tensors on every
    correctness check, which otherwise OOMs the cgroup on heavy kernels (e.g.
    end_to_end Mixtral).
    """
    key = (kernel_name, json.dumps(dims, sort_keys=True, default=str))
    cached = _GOLD_CACHE.get(key)
    if cached is not None:
        return cached
    config = _validate_functional_mod.load_config()
    gold = _validate_functional_mod.run_reference(kernel_name, dims, config)
    _GOLD_CACHE[key] = gold
    return gold


def _compare_against_gold(result, kernel_name, dims, label="result"):
    """Compare a tensor result against gold reference. Returns formatted string."""
    import torch

    gold = _get_gold(kernel_name, dims)

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


def _run_tiled_correctness(code, kernel_name, dims, tensors=None):
    result = _exec_tiled_ref(code, dims, tensors)
    return _compare_against_gold(result, kernel_name, dims, "tiled")


def _run_hybrid_correctness(code, kernel_name, dims, tensors=None):
    result = _exec_hybrid_ref(code, dims, tensors)
    return _compare_against_gold(result, kernel_name, dims, "hybrid")


def _run_dsl_correctness(code, kernel_name, dims, tensors=None):
    result = _exec_dsl_ref(code, dims, tensors)
    return _compare_against_gold(result, kernel_name, dims, "dsl")


def _run_graph_correctness(code, kernel_name, dims, tensors=None):
    from step_py.functional import execute
    graph, output_op = _exec_build_graph(code, dims, tensors)
    try:
        sim = execute(graph, output_op)
    except Exception as exc:
        # Enhance emulator errors with node + user code context
        stripped = code.replace("import ", "# import ")  # strip imports for line matching
        enhanced = enhance_emulator_error(exc, stripped)
        raise type(exc)(enhanced) from exc
    return _compare_against_gold(sim, kernel_name, dims, "sim")


# Map executor type to correctness checker
_CORRECTNESS_CHECKERS = {
    "tiled": _run_tiled_correctness,
    "dsl": _run_dsl_correctness,
    "hybrid": _run_hybrid_correctness,
    "graph": _run_graph_correctness,
    # bundle-dir mode: DSL correctness is skipped; only the post-validator runs
    "passthrough": lambda code, kernel_name, dims, tensors=None: "match=True",
}


# ---------------------------------------------------------------------------
# Compliance checking — each pass progressively constrains allowed operations
# ---------------------------------------------------------------------------

# Cumulative pass orders — each pass inherits all prior bans within its group.
_REFACTOR_ORDER = ["refactor_load", "refactor_compute", "refactor_shape", "refactor_final"]
_TRANSLATION_ORDER = ["translate", "translate_full", "translate_full_no_functional"]

# Allowed torch.XXX() calls in the OUTPUT of each pass.
# None = unrestricted.  set() = nothing allowed.
# Within a cumulative group: effective allowlist = last non-None up to that point.
# Standalone passes (canonicalize, tiler, etc.): checked independently.
_PASS_ALLOWED_TORCH = {
    "refactor_final": set(),  # everything must be DSL — no torch at all
    # --- Translation passes (cumulative) ---
    # Input is DSL code (no torch at all), so torch is banned from the start.
    # The build_graph body should only contain STeP graph construction.
    "translate": set(),  # no torch in build_graph body — all ops are STeP nodes
    "translate_full": set(),
    "translate_full_no_functional": set(),
}

# Allowed F.XXX() calls per pass.
_PASS_ALLOWED_F = {
    "canonicalize": {"F.silu", "F.pad"},
    "refactor_load": {"F.silu", "F.pad"},  # compute still PyTorch
    "refactor_compute": {"F.pad"},           # F.silu replaced by unary_silu; F.pad kept for routing stream padding
    "refactor_shape": {"F.pad"},            # shape ops are DSL; F.pad kept for routing
    "refactor_final": set(),
    "translate": set(),
    "translate_full": set(),
    "translate_full_no_functional": set(),
}

# Extra string patterns banned at each stage.
# For cumulative groups (refactor, translation), bans accumulate across passes.
_PASS_EXTRA_BANS = {
    "tiler": [
        ("torch.einsum",   "use torch.matmul for matrix multiplication"),
        ("torch.bmm",      "use torch.matmul for matrix multiplication"),
        ("F.softmax",      "decompose into exp, row-wise sum, div"),
        ("F.layer_norm",   "decompose into mean-subtract, variance, rsqrt, scale"),
        ("F.gelu",         "decompose into primitives or use F.silu"),
    ],
    "refactor_load": [
        ("out_shape_tiled=(1,)", "NEVER load as one giant tile — use proper streaming: out_shape_tiled=(B//tile_n,) or similar"),
    ],
    "refactor_compute": [
        ("torch.matmul",   "use binary_matmul(a, b)"),
        ("torch.exp",      "use unary_exp(x)"),
        ("torch.rsqrt",    "use unary_rsqrt(x)"),
        ("F.silu",         "use unary_silu(x)"),
    ],
    "refactor_shape": [
        (".unsqueeze(",    "use promote(x, rank) or promote_outer(x)"),
        (".squeeze(",      "use flatten(x, rank, rank) or accum_retile_row/col"),
        (".expand(",       "use expand_ref(x, ref) or repeat_static(x, factor)"),
    ],
    "refactor_final": [
        (".sum(",          "use accum_add(x, rank=1) or unary_rowwise_sum(x)"),
        (".prod(",         "use accum_mul(x, rank=1)"),
    ],
    # Translation: all DSL calls must become STeP nodes in a single pass.
    "translate": [
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
    # Direct pipeline: same bans as translate (no DSL, no PyTorch, only STeP nodes)
    "translate_full": [
        ("execute_values",   "remove mid-function execution; return (graph, output_op)"),
    ],
    "translate_full_no_functional": [
        ("execute_values",   "remove mid-function execution; return (graph, output_op)"),
    ],
}

# Passes where the judge should also run on NONCOMPLIANT code (not just after
# regex passes). Gives the LLM richer line-specific feedback for shape/routing
# conversions where the regex message alone ("`.unsqueeze(` still present") is
# not enough to guide the fix.
_JUDGE_ON_NONCOMPLIANT = {"refactor_shape", "refactor_final"}

# Ops that MUST appear in code after this pass.
_PASS_REQUIRES = {
    "refactor_load":     ["offchip_load", "offchip_store"],
    "translate":         ["LinearOffChipLoad", "OffChipStore"],
    "translate_full":    ["LinearOffChipLoad", "OffChipStore"],
    "translate_full_no_functional": ["LinearOffChipLoad", "OffChipStore"],
}

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


def _check_banned_ops(code: str, pass_name: str) -> list[str]:
    """Check if code complies with this pass's output constraints.

    Returns list of violation messages. Empty = compliant.

    Refactor and translation passes use cumulative rules within their group.
    Standalone passes (canonicalize, tiler) are checked independently.

    For translation passes, only the build_graph body is checked — scaffold
    code (DSL functions, functional.py) may legitimately use torch.* internally.
    """
    # Strip annotation comments before checking — they contain STeP node names
    # that would falsely satisfy _PASS_REQUIRES checks.
    code = _strip_annotations(code)

    # For translation passes, scope checking to the function body only
    if pass_name in _TRANSLATION_ORDER:
        code = _extract_func_body(code)

    has_rules = (pass_name in _PASS_ALLOWED_TORCH or pass_name in _PASS_ALLOWED_F
                 or pass_name in _PASS_EXTRA_BANS or pass_name in _PASS_REQUIRES)
    if not has_rules:
        return []

    # Determine which passes to check cumulatively
    if pass_name in _REFACTOR_ORDER:
        idx = _REFACTOR_ORDER.index(pass_name)
        check_passes = _REFACTOR_ORDER[:idx + 1]
    elif pass_name in _TRANSLATION_ORDER:
        idx = _TRANSLATION_ORDER.index(pass_name)
        check_passes = _TRANSLATION_ORDER[:idx + 1]
    else:
        check_passes = [pass_name]

    violations = []

    # 1. Effective torch allowlist (last non-None among checked passes)
    allowed_torch = None
    for p in check_passes:
        stage = _PASS_ALLOWED_TORCH.get(p)
        if stage is not None:
            allowed_torch = stage

    if allowed_torch is not None:
        for match in _TORCH_CALL_RE.finditer(code):
            call = f"torch.{match.group(1)}"
            if call not in allowed_torch:
                violations.append(f"- `{call}()` is not allowed — replace with a canonical pattern")

    # 2. Effective F allowlist (last non-None among checked passes)
    allowed_f = None
    for p in check_passes:
        stage = _PASS_ALLOWED_F.get(p)
        if stage is not None:
            allowed_f = stage

    if allowed_f is not None:
        for match in _F_CALL_RE.finditer(code):
            call = f"F.{match.group(1)}"
            if call not in allowed_f:
                violations.append(f"- `{call}()` is not allowed — decompose into primitives")

    # 3. Extra banned patterns (cumulative across checked passes)
    # Use word-boundary regex to avoid false positives like "broadcast(" matching "infer_broadcast("
    for p in check_passes:
        for pattern, fix in _PASS_EXTRA_BANS.get(p, []):
            regex = r'\b' + re.escape(pattern)
            if re.search(regex, code):
                violations.append(f"- `{pattern}` still present — {fix}")

    # 4. Required ops (cumulative across checked passes)
    for p in check_passes:
        for required in _PASS_REQUIRES.get(p, []):
            if required not in code:
                violations.append(f"- `{required}` missing — this pass must introduce {required} nodes")

    # Deduplicate
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


async def _run_pass_loop(agent, pass_name, kernel_name, dims, max_turns,
                         ckpt_dir: Path, prev_code=None,
                         executor="tiled", tensors=None, log=print,
                         judge_agent=None, dsl_code=None,
                         post_validator=None):
    """Run a single pass agent (lowering or translator).

    ``post_validator`` is an optional ``(code, turn_dir) -> str | None`` callable
    that runs after correctness + regex compliance + judge all pass. Returning a
    string treats the turn as failed and feeds that string back into the next
    user prompt — this is how deterministic translation surfaces errors back to
    the refactor pass.

    Returns dict with success, code.
    """
    user_prompt = build_pass_user_prompt(pass_name, kernel_name, dims,
                                         prev_code=prev_code,
                                         tensors=tensors,
                                         dsl_code=dsl_code)
    conversation = [{"role": "user", "content": user_prompt}]

    pass_dir = ckpt_dir / pass_name
    # Save the system prompt actually used by the agent (agent.instructions
    # already contains any {few_shot_examples} substitutions from make_pass_agent).
    _write(pass_dir / "system_prompt.txt", agent.instructions)

    check_correctness = _CORRECTNESS_CHECKERS[executor]

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
                "Your response did not contain a code block. "
                "Please provide your implementation inside a ```python code fence."
            })
            continue

        last_code = code
        _write(turn_dir / "extracted_code.py", code)
        log(f"      Extracted code: {len(code)} chars")

        # Run correctness check. Capture stdout so the per-op shape trace
        # printed by step_dsl ops can be fed back to the model on failure.
        log(f"      Running correctness check ({executor})...")
        feedback = ""
        shape_trace = ""
        _trace_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(_trace_buf):
                result = check_correctness(code, kernel_name, dims, tensors)
            shape_trace = _trace_buf.getvalue()
            if shape_trace:
                _write(turn_dir / "shape_trace.txt", shape_trace)
            _write(turn_dir / "correctness_result.txt", result)

            if "match=True" in result:
                violations = _check_banned_ops(code, pass_name)
                if violations:
                    _write(turn_dir / "status.txt", "CORRECT_BUT_NONCOMPLIANT")
                    log(f"      -> CORRECT but {len(violations)} violation(s) remain")
                    if pass_name in _TRANSLATION_ORDER:
                        fix_hint = "Replace these with the corresponding STeP operations."
                    elif pass_name in _REFACTOR_ORDER:
                        fix_hint = "Replace these with the corresponding DSL function calls listed in the instructions."
                    else:
                        fix_hint = "Refactor these into the canonical primitives listed in the instructions."

                    # For some passes, also run the judge on noncompliant code
                    # to give richer line-specific guidance alongside regex violations.
                    judge_feedback = ""
                    if judge_agent is not None and pass_name in _JUDGE_ON_NONCOMPLIANT:
                        log(f"      Also running judge for richer feedback...")
                        judge_ctx = ""
                        if tensors is not None:
                            judge_ctx = "## Input tensors\n" + _format_tensors_description(tensors) + "\n\n"
                        judge_violations, judge_tokens = await _run_judge(
                            judge_agent, code, turn_dir, log, context=judge_ctx)
                        total_tokens += judge_tokens
                        if judge_violations is not None:
                            judge_feedback = "\n\n## Judge feedback (line-specific):\n\n" + judge_violations

                    feedback = (
                        "## Correctness: PASS\n\n"
                        "Your code produces the correct output, but still contains "
                        "disallowed operations:\n\n"
                        + "\n".join(violations)
                        + f"\n\n{fix_hint}"
                        + judge_feedback
                    )
                else:
                    # Regex compliance passed. Run the LLM judge first (its
                    # structural feedback is the most actionable signal we have
                    # at this stage), then the deterministic post_validator —
                    # the translator's errors are last because they're often
                    # downstream symptoms of the same canonical-form issues the
                    # judge catches.
                    judge_violations = None
                    if judge_agent is not None:
                        log(f"      Running judge...")
                        judge_ctx = (
                            "## Correctness status\n"
                            "This code has ALREADY been executed and its output matches the reference "
                            "to within floating-point tolerance. Numerical correctness is verified. "
                            "Only evaluate the three structural checks.\n\n"
                        )
                        if tensors is not None:
                            judge_ctx += "## Input tensors\n" + _format_tensors_description(tensors) + "\n\n"
                        judge_violations, judge_tokens = await _run_judge(
                            judge_agent, code, turn_dir, log, context=judge_ctx)
                        total_tokens += judge_tokens

                    if judge_violations is not None:
                        _write(turn_dir / "status.txt", "CORRECT_BUT_JUDGE_REJECTED")
                        log(f"      -> CORRECT but judge rejected")
                        feedback = (
                            "## Correctness: PASS\n\n"
                            "Your code produces the correct output and uses allowed operations, "
                            "but does not follow canonical form:\n\n"
                            + judge_violations
                            + "\n\nFix these structural issues while keeping the output correct."
                        )
                    else:
                        post_feedback = None
                        if post_validator is not None:
                            log(f"      Running post-validator...")
                            post_feedback = post_validator(code, turn_dir)

                        if post_feedback is not None:
                            _write(turn_dir / "status.txt", "CORRECT_BUT_POST_VALIDATOR_REJECTED")
                            log(f"      -> CORRECT but post-validator rejected")
                            feedback = post_feedback
                        else:
                            success = True
                            _write(turn_dir / "status.txt", "PASS")
                            log(f"      -> PASS{' (judge approved)' if judge_agent is not None else ''}")
                            break
            else:
                _write(turn_dir / "status.txt", f"FAIL: {result.splitlines()[0]}")
                log(f"      -> FAIL: {result.splitlines()[0]}")
                feedback = f"## Correctness check result\n{result}"
        except Exception:
            shape_trace = _trace_buf.getvalue()
            if shape_trace:
                _write(turn_dir / "shape_trace.txt", shape_trace)
            err = traceback.format_exc()
            _write(turn_dir / "correctness_result.txt", f"ERROR:\n{err}")
            log(f"      -> ERROR: {_error_summary(err)}")
            feedback = f"## Error running code\n{err}"
            if "ModuleNotFoundError" in err or "ImportError" in err:
                feedback += (
                    "\n\n**IMPORTANT: Do NOT include any import statements in your code.** "
                    "All imports are injected automatically. Remove ALL import/from lines."
                )
            if "missing 1 required positional argument" in err:
                feedback += (
                    "\n\n**IMPORTANT: Most STeP ops require `graph` as the FIRST positional arg.** "
                    "Source ops (LinearOffChipLoad, SelectGen, MetadataGen) do NOT take graph. "
                    "ALL other ops take `graph` as their first argument: "
                    "`Promote(graph, input, promote_rank=2)` not `Promote(input, promote_rank=2)`."
                )
            if "FlatPartition" in err and ("not subscriptable" in err or "not iterable" in err):
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

        if shape_trace:
            # Cap at the last MAX_LINES so the trace nearest the failure point
            # is preserved without blowing up the context window on long runs.
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

    # --- Step 1: bundle-dir path resolution ---
    bundle_path = None
    if bundle_dir is not None:
        bundle_path = Path(bundle_dir).resolve()
        assert bundle_path.exists(), f"bundle dir not found: {bundle_path}"
        if str(bundle_path) not in sys.path:
            sys.path.insert(0, str(bundle_path))

    # --- Step 2: resolve translate_fn and refactor_system_prompt ---
    if bundle_dir is not None:
        import importlib
        # Force reimport in case a previous bundle in the same process polluted sys.modules.
        if "transpiler" in sys.modules:
            del sys.modules["transpiler"]
        transpiler_mod = importlib.import_module("transpiler")
        translate_fn = transpiler_mod.translate
        refactor_system_prompt = (bundle_path / "refactor_system.txt").read_text()
    else:
        translate_fn = _dsl_to_step_translate
        refactor_system_prompt = None

    # --- Step 5: bundle-dir mode uses only refactor_final + deterministic translate ---
    if bundle_dir is not None:
        # Single-pass: input abstraction DSL → refactor_final → translate_fn → graph check.
        # DSL correctness is skipped (passthrough); the post-validator is the only gate.
        lowering_passes = [{"name": "refactor_final", "executor": "passthrough"}]
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

    # Create judge agents for passes that have one
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
        "few_shot_paths": list(few_shot_paths) if few_shot_paths else [],
    }, indent=2))

    # Run all outer iterations in parallel — they are independent attempts
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

    per_outer = [{"outer": i, "success": bool(r["success"])} for i, r in enumerate(results)]

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
    lowering_passes: list = None, translator_passes: list = None,
    resume_dsl_code: str = None,
    translator: str = "llm",
    translate_fn=None,
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

    lowered_code = None
    dsl_code = None  # output of refactor_final, used as translation guide
    pipeline_ok = True
    outer_total_tokens = 0

    # ============================================================
    # Resume mode: skip lowering, use provided DSL code directly
    # ============================================================
    if resume_dsl_code is not None:
        dsl_code = resume_dsl_code
        _write(outer_dir / "dsl_code.py", dsl_code)
        log(f"  Resumed from checkpoint — using saved dsl_code ({len(dsl_code)} chars)")
        print(f"{tag} Resumed — skipping lowering")

    # ============================================================
    # Phase 1: Lowering passes (skipped when resuming)
    #   tiler -> router -> retiler -> canonicalize ->
    #   refactor_load -> refactor_compute -> refactor_final
    # The refactor passes progressively rewrite canonical PyTorch into
    # DSL function calls that map 1:1 to STeP graph nodes.
    # ============================================================
    else:
        for pass_info in lowering_passes:
            pass_name = pass_info["name"]
            executor = pass_info.get("executor", "tiled")

            # Skip refactor passes if code already complies AND no judge needs to
            # verify. For refactor_final under translator='auto', also require that
            # deterministic translation already succeeds — otherwise the skip
            # would mask a translator-side failure that the loop is meant to fix.
            if pass_name in _REFACTOR_ORDER and lowered_code is not None:
                violations = _check_banned_ops(lowered_code, pass_name)
                has_judge = pass_name in judge_agents
                needs_translate_check = (
                    pass_name == "refactor_final" and translator == "auto"
                )
                if not violations and not has_judge and not needs_translate_check:
                    log(f"  Lowering pass: {pass_name} -> SKIP (already compliant)")
                    if pass_name == "refactor_final":
                        dsl_code = lowered_code
                        _write(outer_dir / "dsl_code.py", dsl_code)
                    continue

            # When using deterministic translation, gate refactor_final on the
            # translator: if translation fails, treat it as a refactor error so
            # the model fixes the DSL until it lowers cleanly into STeP IR.
            post_validator = None
            if pass_name == "refactor_final" and translator == "auto":
                post_validator = _make_translation_post_validator(
                    kernel_name, dims, tensors, log,
                    translate_fn=translate_fn,
                )

            log(f"  Lowering pass: {pass_name}")
            pass_result = await _run_pass_loop(
                pass_agents[pass_name], pass_name, kernel_name, dims, max_turns,
                ckpt_dir=outer_dir,
                prev_code=lowered_code,
                executor=executor,
                tensors=tensors,
                log=log,
                judge_agent=judge_agents.get(pass_name),
                post_validator=post_validator,
            )
            outer_total_tokens += pass_result.get("total_tokens", 0)
            if pass_result["success"]:
                lowered_code = pass_result["code"]
                log(f"  -> {pass_name} OK")
                # Save refactor_final output as DSL code for translation guidance
                if pass_name == "refactor_final":
                    dsl_code = lowered_code
                    _write(outer_dir / "dsl_code.py", dsl_code)
            else:
                log(f"  -> {pass_name} FAILED, stopping lowering pipeline")
                pipeline_ok = False
                break

        if lowering_passes:
            if not pipeline_ok or lowered_code is None:
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

            log(f"Lowering pipeline succeeded")
            print(f"{tag} Lowering OK")
        else:
            log(f"No lowering passes (direct pipeline)")
            print(f"{tag} Direct pipeline — skipping lowering")

    # ============================================================
    # Phase 2: STeP translation passes
    # The DSL code from the refactor pass is passed as a translation guide.
    # Each DSL call maps 1:1 to a STeP node, making translation mechanical.
    # ============================================================
    translated_code = dsl_code if dsl_code is not None else lowered_code
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

        # Skip if input already complies with this pass's requirements
        if translated_code is not None:
            violations = _check_banned_ops(translated_code, pass_name)
            if not violations:
                log(f"  Translation pass: {pass_name} -> SKIP (already compliant)")
                continue

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
        # Verify the final code actually works as a build_graph
        final_code = translated_code
        log(f"Translation pipeline completed — verifying final graph...")
        try:
            graph_result = _run_graph_correctness(final_code, kernel_name, dims, tensors)
            if "match=True" in graph_result:
                log(f"-> PASS (graph verified)")
                print(f"{tag} SUCCESS")
                log_file.close()
                result = _build_success_result(i, 0,
                                               {"code": final_code, "tool_outputs": []},
                                               [], lowered_code)
                result["total_tokens"] = outer_total_tokens
                return result
            else:
                log(f"-> FAIL: {graph_result.splitlines()[0]}")
                translation_ok = False
        except Exception:
            err_msg = traceback.format_exc().splitlines()[-1]
            log(f"-> ERROR: {err_msg}")
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
        "tiled_code": lowered_code,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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


