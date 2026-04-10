"""Orchestrator for StepGenFlow5 — progressive translation pipeline.

Two-phase pipeline:
  Phase 1 (lowering): tiler -> router -> retiler -> canonicalize
    Each pass outputs tiled_reference(dims) -> torch.Tensor, validated against gold.
    Canonicalize enforces single-assignment form with only canonical ops.
  Phase 2 (translation): translate_load -> translate_compute -> translate_final
    First 2 output hybrid_reference(dims) -> torch.Tensor (mix of STeP + PyTorch), validated against gold.
    Last one (translate_final) handles routing + reductions + graph finalization, outputs
    build_graph(dims) -> (graph, output_op), validated via emulator against gold.

Checkpoint structure:
  checkpoints/<timestamp>/<kernel>/outer_<N>/<pass_name>/turn_<M>/...
  checkpoints/<timestamp>/<kernel>/outer_<N>/analyst/...
  checkpoints/<timestamp>/<kernel>/result.json
"""

import asyncio
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml
from agents import Runner

from src.agents import make_judge_agent, make_pass_agent
from src.precompute import precompute_tensors
from src.prompts import (LOWERING_PASSES, TRANSLATOR_PASSES,
                         build_pass_system_prompt, build_pass_user_prompt,
                         _format_tensors_description)
from src.tools import (_exec_build_graph, _exec_tiled_ref, _exec_hybrid_ref,
                       _exec_dsl_ref, validate_tiling, _format_node_values,
                       _validate_functional_mod, enhance_emulator_error)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_DEIO_ROOT = Path(__file__).resolve().parent.parent.parent  # DEIOpt/
_STEPDB_DIR = _DEIO_ROOT / "StepDB"
_STEP_TL_SRC = _DEIO_ROOT / "step_tl" / "src"
_STEP_TL_PROTO = _STEP_TL_SRC / "proto"

for p in (_STEP_TL_SRC, _STEP_TL_PROTO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """Extract the last python code block from LLM output."""
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else ""


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


def _save_experience(kernel_name: str, code: str, metadata: dict, experience_dir: str) -> None:
    out_dir = Path(experience_dir) / kernel_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "solution.py").write_text(code)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


# ---------------------------------------------------------------------------
# Correctness checkers for each executor type
# ---------------------------------------------------------------------------

def _compare_against_gold(result, kernel_name, dims, label="result"):
    """Compare a tensor result against gold reference. Returns formatted string."""
    import torch

    config = _validate_functional_mod.load_config()
    gold = _validate_functional_mod.run_reference(kernel_name, dims, config)

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
}


# ---------------------------------------------------------------------------
# Compliance checking — each pass progressively constrains allowed operations
# ---------------------------------------------------------------------------

# Cumulative pass orders — each pass inherits all prior bans within its group.
_REFACTOR_ORDER = ["refactor_load", "refactor_compute", "refactor_shape", "refactor_final"]
_TRANSLATION_ORDER = ["translate_load", "translate_compute", "translate_final"]

# Allowed torch.XXX() calls in the OUTPUT of each pass.
# None = unrestricted.  set() = nothing allowed.
# Within a cumulative group: effective allowlist = last non-None up to that point.
# Standalone passes (canonicalize, tiler, etc.): checked independently.
_PASS_ALLOWED_TORCH = {
    # --- Lowering: canonicalize (standalone) ---
    "canonicalize": {
        "torch.matmul", "torch.exp", "torch.rsqrt", "torch.pow",
        "torch.zeros", "torch.zeros_like", "torch.ones", "torch.ones_like",
        "torch.arange", "torch.tensor",
        "torch.stack", "torch.cat",
        "torch.where", "torch.nonzero",
        "torch.no_grad",
    },
    # --- Refactor passes (cumulative) ---
    "refactor_load": {
        # Loads are DSL, but compute + routing still PyTorch
        "torch.matmul", "torch.exp", "torch.rsqrt", "torch.pow",
        "torch.zeros", "torch.zeros_like", "torch.ones", "torch.ones_like",
        "torch.arange", "torch.tensor",
        "torch.stack", "torch.cat",
        "torch.where", "torch.nonzero",
        "torch.no_grad",
    },
    "refactor_compute": {
        # Compute ops removed — only routing/accum/utility remains
        "torch.zeros", "torch.zeros_like", "torch.ones", "torch.ones_like",
        "torch.arange", "torch.tensor",
        "torch.stack", "torch.cat",
        "torch.where", "torch.nonzero",
        "torch.no_grad",
    },
    "refactor_shape": {
        # Shape ops now DSL — same torch allowed as refactor_compute (routing still PyTorch)
        "torch.zeros", "torch.zeros_like", "torch.ones", "torch.ones_like",
        "torch.arange", "torch.tensor",
        "torch.stack", "torch.cat",
        "torch.where", "torch.nonzero",
        "torch.no_grad",
    },
    "refactor_final": set(),  # everything must be DSL — no torch at all
    # --- Translation passes (cumulative) ---
    # Input is DSL code (no torch at all), so torch is banned from the start.
    # The build_graph body should only contain STeP graph construction.
    "translate_load": set(),      # no torch in build_graph body
    "translate_compute": set(),
    "translate_final": set(),
}

# Allowed F.XXX() calls per pass.
_PASS_ALLOWED_F = {
    "canonicalize": {"F.silu", "F.pad"},
    "refactor_load": {"F.silu", "F.pad"},  # compute still PyTorch
    "refactor_compute": {"F.pad"},           # F.silu replaced by unary_silu; F.pad kept for routing stream padding
    "refactor_shape": {"F.pad"},            # shape ops are DSL; F.pad kept for routing
    "refactor_final": set(),
    "translate_load": set(),
    "translate_compute": set(),
    "translate_final": set(),
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
    # Translation: progressively ban DSL calls as they become STeP nodes.
    # These are cumulative — translate_final inherits all bans from load + compute.
    "translate_load": [
        ("offchip_load(",    "replace with LinearOffChipLoad(underlying, stride, out_shape_tiled, tile_row, tile_col, par_dispatch, transposed)"),
        ("offchip_store(",   "replace with OffChipStore(graph, input, par_dispatch=1)"),
        ("select_gen(",      "replace with SelectGen(is_multihot, tensor, n)"),
    ],
    "translate_compute": [
        ("binary_matmul(",   "replace with BinaryMap(graph, a, b, map_fn.Matmul(), False, 1024)"),
        ("binary_mul(",      "replace with BinaryMap(graph, a, b, map_fn.Mul(), False, 1024)"),
        ("binary_add(",      "replace with BinaryMap(graph, a, b, map_fn.Add(), False, 1024)"),
        ("binary_div(",      "replace with BinaryMap(graph, a, b, map_fn.Div(), False, 1024)"),
        ("binary_is_equal(", "replace with BinaryMap(graph, a, b, map_fn.IsEqual(), False, 1024)"),
        ("unary_silu(",      "replace with UnaryMap(graph, x, map_fn.Silu(), False, 1024)"),
        ("unary_square(",    "replace with UnaryMap(graph, x, map_fn.Square(), False, 1024)"),
        ("unary_exp(",       "replace with UnaryMap(graph, x, map_fn.Exp(), False, 1024)"),
        ("unary_rsqrt(",     "replace with UnaryMap(graph, x, map_fn.Rsqrt(), False, 1024)"),
        ("unary_pow2(",      "replace with UnaryMap(graph, x, map_fn.Pow2(), False, 1024)"),
        ("unary_mul_imm(",   "replace with UnaryMap(graph, x, map_fn.MulImmediate(c), False, 1024)"),
        ("unary_add_imm(",   "replace with UnaryMap(graph, x, map_fn.AddImmediate(c), False, 1024)"),
        ("unary_sub_imm(",   "replace with UnaryMap(graph, x, map_fn.SubImmediate(c), False, 1024)"),
        ("unary_rowwise_sum(","replace with UnaryMap(graph, x, map_fn.RowWiseSum(), False, 1024)"),
    ],
    "translate_final": [
        ("accum_add(",       "replace with Accum(graph, x, ..., accum_fn.Add(), ..., accum_rank=rank)"),
        ("accum_mul(",       "replace with Accum(graph, x, ..., accum_fn.Mul(), ..., accum_rank=rank)"),
        ("accum_retile_row(","replace with Accum(graph, x, ..., accum_fn.RetileRow(), ...)"),
        ("accum_retile_col(","replace with Accum(graph, x, ..., accum_fn.RetileCol(), ...)"),
        ("promote(",         "replace with Promote(graph, input, promote_rank=rank)"),
        ("promote_outer(",   "replace with PromoteOuter(graph, input)"),
        ("expand_ref(",      "replace with ExpandRef(graph, input, ref)"),
        ("repeat_static(",   "replace with RepeatStatic(graph, input, repeat_factor)"),
        ("flatten(",         "replace with Flatten(graph, input, min_rank, max_rank)"),
        ("flat_partition(",  "replace with FlatPartition(graph, input, control, ...)"),
        ("flat_reassemble(", "replace with FlatReassemble(graph, inputs, control, ...)"),
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
    "translate_load":    ["LinearOffChipLoad", "OffChipStore"],
    "translate_compute": ["BinaryMap"],
    "translate_final":   ["OffChipStore"],
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
    for p in check_passes:
        for pattern, fix in _PASS_EXTRA_BANS.get(p, []):
            if pattern in code:
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
                     log=print, context: str = "") -> str | None:
    """Run the LLM judge on code. Returns None if PASS, or violation feedback if REJECT."""
    judge_prompt = f"Review this code:\n\n{context}\n```python\n{code}\n```" if context else \
                   f"Review this code for compliance:\n\n```python\n{code}\n```"
    result = await Runner.run(judge_agent, [{"role": "user", "content": judge_prompt}])
    judge_text = result.final_output or ""
    _write(turn_dir / "judge_response.txt", judge_text)

    if "VERDICT: PASS" in judge_text:
        return None

    # Extract violations from judge response
    if "VERDICT: REJECT" in judge_text:
        # Everything after VIOLATIONS: is the feedback
        idx = judge_text.find("VIOLATIONS:")
        if idx != -1:
            violations_text = judge_text[idx:]
        else:
            violations_text = judge_text[judge_text.find("VERDICT: REJECT"):]
        return violations_text

    # Ambiguous response — treat as reject
    log(f"      Judge gave ambiguous verdict, treating as reject")
    return judge_text


async def _run_pass_loop(agent, pass_name, kernel_name, dims, max_turns,
                         ckpt_dir: Path, prev_code=None,
                         executor="tiled", tensors=None, log=print,
                         judge_agent=None, dsl_code=None):
    """Run a single pass agent (lowering or translator).

    Returns dict with success, code.
    """
    system_prompt = build_pass_system_prompt(pass_name)
    user_prompt = build_pass_user_prompt(pass_name, kernel_name, dims,
                                         prev_code=prev_code,
                                         tensors=tensors,
                                         dsl_code=dsl_code)
    conversation = [{"role": "user", "content": user_prompt}]

    pass_dir = ckpt_dir / pass_name
    _write(pass_dir / "system_prompt.txt", system_prompt)

    check_correctness = _CORRECTNESS_CHECKERS[executor]

    last_code = None
    success = False

    for turn in range(max_turns):
        turn_dir = pass_dir / f"turn_{turn}"
        log(f"    [{pass_name}] Turn {turn + 1}/{max_turns}...")

        last_user_msg = conversation[-1]["content"] if conversation[-1]["role"] == "user" else ""
        _write(turn_dir / "user_prompt.txt", last_user_msg)

        run_result = await Runner.run(agent, conversation)
        assistant_text = run_result.final_output or ""
        conversation.append({"role": "assistant", "content": assistant_text})
        _write(turn_dir / "response.txt", assistant_text)

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

        # Run correctness check
        log(f"      Running correctness check ({executor})...")
        feedback = ""
        try:
            result = check_correctness(code, kernel_name, dims, tensors)
            _write(turn_dir / "correctness_result.txt", result)

            if "match=True" in result:
                # Correctness passed — run tiling structure validation
                # Runs after every pass to ensure tiled form is maintained.
                if tensors is not None:
                    tiling_violations = validate_tiling(code, dims, tensors)
                    if tiling_violations:
                        _write(turn_dir / "status.txt", "CORRECT_BUT_BAD_TILING")
                        log(f"      -> CORRECT but {len(tiling_violations)} tiling violation(s)")
                        _write(turn_dir / "tiling_violations.txt",
                               "\n".join(tiling_violations))
                        feedback = (
                            "## Correctness: PASS\n\n"
                            "Your code produces the correct output, but the tensor tiling "
                            "structure is wrong:\n\n"
                            + "\n".join(f"- {v}" for v in tiling_violations)
                            + "\n\nEvery data tensor must be in tiled form "
                            "(*stream_dims, tile_r, tile_c) with at least one streaming "
                            "dimension > 1. Use tile sizes from dims (like tile_n), "
                            "not full dimensions (like B)."
                        )
                        conversation.append({"role": "user", "content": feedback})
                        continue

                # Tiling OK — now check banned ops compliance
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
                        judge_violations = await _run_judge(
                            judge_agent, code, turn_dir, log, context=judge_ctx)
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
                    # Regex compliance passed — run LLM judge if configured
                    if judge_agent is not None:
                        log(f"      Running judge...")
                        judge_ctx = ""
                        if tensors is not None:
                            judge_ctx = "## Input tensors\n" + _format_tensors_description(tensors) + "\n\n"
                        judge_violations = await _run_judge(
                            judge_agent, code, turn_dir, log, context=judge_ctx)
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
                            success = True
                            _write(turn_dir / "status.txt", "PASS")
                            log(f"      -> PASS (judge approved)")
                            break
                    else:
                        success = True
                        _write(turn_dir / "status.txt", "PASS")
                        log(f"      -> PASS")
                        break
            else:
                _write(turn_dir / "status.txt", f"FAIL: {result.splitlines()[0]}")
                log(f"      -> FAIL: {result.splitlines()[0]}")
                feedback = f"## Correctness check result\n{result}"
        except Exception:
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

        conversation.append({"role": "user", "content": feedback})

    return {"success": success, "code": last_code}


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
) -> dict:
    """Run the full pipeline for a single kernel + preset."""
    config = _load_stepdb_config()
    assert kernel_name in config, f"Kernel '{kernel_name}' not found"
    assert preset in config[kernel_name]["presets"], f"Preset '{preset}' not found"
    dims = config[kernel_name]["presets"][preset]

    # Pre-compute all tensors externally — functions receive these, can't create their own
    tensors = precompute_tensors(kernel_name, dims)
    print(f"Pre-computed tensors: {sorted(tensors.keys())}")

    # Create agents for all passes
    all_passes = LOWERING_PASSES + TRANSLATOR_PASSES
    pass_agents = {p["name"]: make_pass_agent(llm_config, p["name"]) for p in all_passes}

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
    }, indent=2))

    # Run all outer iterations in parallel — they are independent attempts
    tasks = []
    for i in range(max_outer):
        outer_dir = ckpt_root / f"outer_{i}"
        tasks.append(_run_outer_iteration(
            i, max_outer, outer_dir, kernel_name, dims, tensors,
            pass_agents, judge_agents,
            max_turns, ckpt_root, preset, experience_dir,
        ))

    results = await asyncio.gather(*tasks)

    # Return first success, or the last failure
    for result in results:
        if result["success"]:
            _write(ckpt_root / "result.json", json.dumps(result, indent=2, default=str))
            return result

    # All failed — return last result
    result = results[-1]
    _write(ckpt_root / "result.json", json.dumps(result, indent=2, default=str))
    return result


async def _run_outer_iteration(
    i: int, max_outer: int, outer_dir: Path,
    kernel_name: str, dims: dict, tensors: dict,
    pass_agents: dict, judge_agents: dict,
    max_turns: int,
    ckpt_root: Path, preset: str, experience_dir: str,
) -> dict:
    """Run a single outer iteration of the pipeline (lowering + translation).

    All detailed output goes to outer_dir/log.txt. Only summary lines go to terminal.
    """
    log_path = outer_dir / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")

    def log(msg):
        log_file.write(msg + "\n")
        log_file.flush()

    tag = f"[outer_{i}]"
    print(f"{tag} Started — log: {log_path}")
    log(f"--- Outer iteration {i + 1}/{max_outer} ---")

    # ============================================================
    # Phase 1: Lowering passes
    #   tiler -> router -> retiler -> canonicalize ->
    #   refactor_load -> refactor_compute -> refactor_final
    # The refactor passes progressively rewrite canonical PyTorch into
    # DSL function calls that map 1:1 to STeP graph nodes.
    # ============================================================
    lowered_code = None
    dsl_code = None  # output of refactor_final, used as translation guide
    pipeline_ok = True
    for pass_info in LOWERING_PASSES:
        pass_name = pass_info["name"]
        executor = pass_info.get("executor", "tiled")

        # Skip refactor passes if code already complies AND no judge needs to verify
        if pass_name in _REFACTOR_ORDER and lowered_code is not None:
            violations = _check_banned_ops(lowered_code, pass_name)
            has_judge = pass_name in judge_agents
            if not violations and not has_judge:
                log(f"  Lowering pass: {pass_name} -> SKIP (already compliant)")
                if pass_name == "refactor_final":
                    dsl_code = lowered_code
                    _write(outer_dir / "dsl_code.py", dsl_code)
                continue

        log(f"  Lowering pass: {pass_name}")
        pass_result = await _run_pass_loop(
            pass_agents[pass_name], pass_name, kernel_name, dims, max_turns,
            ckpt_dir=outer_dir,
            prev_code=lowered_code,
            executor=executor,
            tensors=tensors,
            log=log,
            judge_agent=judge_agents.get(pass_name),
        )
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

    if not pipeline_ok or lowered_code is None:
        log(f"Lowering pipeline failed")
        print(f"{tag} Lowering FAILED")
        log_file.close()
        return {
            "success": False,
            "outer_iteration": i,
            "outer_iterations": max_outer,
            "total_tool_calls": 0,
            "cycle_count": None,
        }

    log(f"Lowering pipeline succeeded")
    print(f"{tag} Lowering OK")

    # ============================================================
    # Phase 2: STeP translation passes
    # The DSL code from the refactor pass is passed as a translation guide.
    # Each DSL call maps 1:1 to a STeP node, making translation mechanical.
    # ============================================================
    translated_code = dsl_code if dsl_code is not None else lowered_code
    translation_ok = True
    for pass_info in TRANSLATOR_PASSES:
        pass_name = pass_info["name"]
        executor = pass_info["executor"]

        # Skip if input already complies with this pass's requirements
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
                _save_experience(kernel_name, final_code,
                                 {"kernel": kernel_name, "preset": preset}, experience_dir)
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


