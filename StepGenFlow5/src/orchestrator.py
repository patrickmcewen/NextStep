"""Orchestrator for StepGenFlow4 — progressive translation pipeline.

Two-phase pipeline:
  Phase 1 (lowering): tiler -> router -> retiler
    Each pass outputs tiled_reference(dims) -> torch.Tensor, validated against gold.
  Phase 2 (translation): translate_load -> translate_compute -> translate_routing -> translate_accum
    First 3 output hybrid_reference(dims) -> torch.Tensor (mix of STeP + PyTorch), validated against gold.
    Last one outputs build_graph(dims) -> (graph, output_op), validated via emulator against gold.

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

from src.agents import make_pass_agent
from src.precompute import precompute_tensors
from src.prompts import (LOWERING_PASSES, TRANSLATOR_PASSES,
                         build_pass_system_prompt, build_pass_user_prompt)
from src.tools import (_exec_build_graph, _exec_tiled_ref, _exec_hybrid_ref,
                       _format_node_values, _validate_functional_mod,
                       enhance_emulator_error)

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


def _run_graph_correctness(code, kernel_name, dims, tensors=None):
    from step_py.functional import execute
    graph, output_op = _exec_build_graph(code, dims, tensors)
    sim = execute(graph, output_op)
    return _compare_against_gold(sim, kernel_name, dims, "sim")


# Map executor type to correctness checker
_CORRECTNESS_CHECKERS = {
    "tiled": _run_tiled_correctness,
    "hybrid": _run_hybrid_correctness,
    "graph": _run_graph_correctness,
}


# ---------------------------------------------------------------------------
# Banned-ops checking — each translation pass progressively bans more PyTorch
# ---------------------------------------------------------------------------

# What each pass eliminates. After a pass, its patterns (and all prior) are banned.
_PASS_BANS = {
    "translate_load": [
        ("torch.randn",        "use tensors dict + LinearOffChipLoad"),
        ("torch.manual_seed",  "tensors are pre-computed"),
        ("torch.rand(",        "use tensors dict + LinearOffChipLoad"),
    ],
    "translate_compute": [
        ("torch.matmul",  "use BinaryMap(map_fn.Matmul) or BinaryMapAccum(map_accum_fn.Matmul)"),
        ("F.silu(",       "use UnaryMap(map_fn.Silu)"),
        ("torch.exp(",    "use UnaryMap(map_fn.Exp)"),
        ("torch.rsqrt(",  "use UnaryMap(map_fn.Rsqrt)"),
    ],
    "translate_routing": [
        # Routing patterns are structurally complex — hard to detect with string matching.
        # Correctness check handles this; compliance is best-effort.
    ],
    "translate_accum": [
        (".sum(dim=",       "use Accum(accum_fn.Add)"),
        ("execute_values",  "remove mid-function execution; return (graph, output_op)"),
    ],
}

_TRANSLATION_ORDER = ["translate_load", "translate_compute", "translate_routing", "translate_accum"]


def _check_banned_ops(code: str, pass_name: str) -> list[str]:
    """Check if code contains ops that should have been translated by this pass.

    Returns list of violation messages. Empty = compliant.
    Cumulative: bans everything from this pass and all prior translation passes.
    """
    if pass_name not in _TRANSLATION_ORDER:
        return []

    pass_idx = _TRANSLATION_ORDER.index(pass_name)
    violations = []
    for i in range(pass_idx + 1):
        for pattern, fix in _PASS_BANS.get(_TRANSLATION_ORDER[i], []):
            if pattern in code:
                violations.append(f"- `{pattern}` still present — {fix}")
    return violations


# ---------------------------------------------------------------------------
# Generic pass loop — works for both lowering and translator passes
# ---------------------------------------------------------------------------

async def _run_pass_loop(agent, pass_name, kernel_name, dims, max_turns,
                         ckpt_dir: Path, prev_code=None,
                         executor="tiled", tensors=None, log=print):
    """Run a single pass agent (lowering or translator).

    Returns dict with success, code.
    """
    system_prompt = build_pass_system_prompt(pass_name)
    user_prompt = build_pass_user_prompt(pass_name, kernel_name, dims,
                                         prev_code=prev_code,
                                         tensors=tensors)
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
                # Correctness passed — now check banned ops compliance
                violations = _check_banned_ops(code, pass_name)
                if violations:
                    _write(turn_dir / "status.txt", "CORRECT_BUT_NONCOMPLIANT")
                    log(f"      -> CORRECT but {len(violations)} banned op(s) remain")
                    feedback = (
                        "## Correctness: PASS\n\n"
                        "Your code produces the correct output, but still contains PyTorch "
                        "operations that must be replaced with STeP graph nodes:\n\n"
                        + "\n".join(violations)
                        + "\n\nReplace these with the corresponding STeP operations."
                    )
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
            pass_agents, max_turns, ckpt_root, preset, experience_dir,
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
    pass_agents: dict, max_turns: int,
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
    # Phase 1: PyTorch lowering passes (tiler -> router -> retiler)
    # ============================================================
    lowered_code = None
    pipeline_ok = True
    for pass_info in LOWERING_PASSES:
        pass_name = pass_info["name"]

        log(f"  Lowering pass: {pass_name}")
        pass_result = await _run_pass_loop(
            pass_agents[pass_name], pass_name, kernel_name, dims, max_turns,
            ckpt_dir=outer_dir,
            prev_code=lowered_code,
            executor="tiled",
            tensors=tensors,
            log=log,
        )
        if pass_result["success"]:
            lowered_code = pass_result["code"]
            log(f"  -> {pass_name} OK")
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
    # ============================================================
    translated_code = lowered_code
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


