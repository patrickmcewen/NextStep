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

import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml
from agents import Runner

from src.agents import make_pass_agent, make_analyst_agent
from src.precompute import precompute_tensors
from src.prompts import (LOWERING_PASSES, TRANSLATOR_PASSES,
                         build_pass_system_prompt, build_pass_user_prompt,
                         build_analyst_prompt)
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
# Generic pass loop — works for both lowering and translator passes
# ---------------------------------------------------------------------------

async def _run_pass_loop(agent, pass_name, kernel_name, dims, max_turns,
                         ckpt_dir: Path, prev_code=None, diagnosis=None,
                         executor="tiled", allow_nop=True, tensors=None):
    """Run a single pass agent (lowering or translator).

    Args:
        executor: "tiled" for tiled_reference, "hybrid" for hybrid_reference,
                  "graph" for build_graph.
        allow_nop: if False, NOP responses are rejected and the LLM is told to produce output.
        tensors: pre-computed tensors dict to pass to the function. If None, functions
                 use old (dims)-only signature.
    Returns dict with success, code, nop.
    """
    system_prompt = build_pass_system_prompt(pass_name)
    user_prompt = build_pass_user_prompt(pass_name, kernel_name, dims,
                                         prev_code=prev_code, diagnosis=diagnosis,
                                         tensors=tensors)
    conversation = [{"role": "user", "content": user_prompt}]

    pass_dir = ckpt_dir / pass_name
    _write(pass_dir / "system_prompt.txt", system_prompt)

    check_correctness = _CORRECTNESS_CHECKERS[executor]

    last_code = None
    success = False

    for turn in range(max_turns):
        turn_dir = pass_dir / f"turn_{turn}"
        print(f"    [{pass_name}] Turn {turn + 1}/{max_turns}...")

        last_user_msg = conversation[-1]["content"] if conversation[-1]["role"] == "user" else ""
        _write(turn_dir / "user_prompt.txt", last_user_msg)

        run_result = await Runner.run(agent, conversation)
        assistant_text = run_result.final_output or ""
        conversation.append({"role": "assistant", "content": assistant_text})
        _write(turn_dir / "response.txt", assistant_text)

        # Check for NOP — LLM may return it as plain text or in a code block
        is_nop = False
        stripped_text = assistant_text.strip()
        if stripped_text == "NOP" or stripped_text.endswith("\nNOP"):
            is_nop = True

        code = _extract_code(assistant_text)
        if not is_nop and not code:
            print(f"      No code block found ({len(assistant_text)} chars). Stopping.")
            _write(turn_dir / "status.txt", "NO_CODE_EXTRACTED")
            break

        if not is_nop and code and code.strip() == "NOP":
            is_nop = True

        if is_nop:
            if allow_nop:
                print(f"      -> NOP (no transformation needed)")
                _write(turn_dir / "status.txt", "NOP")
                return {"success": True, "code": prev_code, "nop": True}
            else:
                print(f"      -> NOP rejected (this pass must produce output)")
                _write(turn_dir / "status.txt", "NOP_REJECTED")
                conversation.append({"role": "user", "content":
                    "NOP is not allowed for this pass. You MUST transform the code into a "
                    "`build_graph(dims)` function that returns `(graph, output_op)`. "
                    "Convert all remaining PyTorch operations to STeP graph nodes, "
                    "add OffChipStore, and remove any execute_values calls."
                })
                continue

        last_code = code
        _write(turn_dir / "extracted_code.py", code)
        print(f"      Extracted code: {len(code)} chars")

        # Run correctness check
        print(f"      Running correctness check ({executor})...")
        feedback = ""
        try:
            result = check_correctness(code, kernel_name, dims, tensors)
            _write(turn_dir / "correctness_result.txt", result)

            if "match=True" in result:
                success = True
                _write(turn_dir / "status.txt", "PASS")
                print(f"      -> PASS")
                break
            else:
                _write(turn_dir / "status.txt", f"FAIL: {result.splitlines()[0]}")
                print(f"      -> FAIL: {result.splitlines()[0]}")
                feedback = f"## Correctness check result\n{result}"
        except Exception:
            err = traceback.format_exc()
            _write(turn_dir / "correctness_result.txt", f"ERROR:\n{err}")
            print(f"      -> ERROR: {_error_summary(err)}")
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

        # Include the failing code so the LLM can apply targeted fixes
        #feedback += (
        #    f"\n\n## Your code (fix the error and resubmit)\n\n```python\n{code}\n```"
        #)

        conversation.append({"role": "user", "content": feedback})

    return {"success": success, "code": last_code, "nop": False}


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
    analyst_agent = make_analyst_agent(llm_config)

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

    all_inner_results = []
    total_tool_calls = 0
    diagnosis = None

    # Track NOP decisions across outer iterations — once a pass attempts
    # translation (non-NOP), it must not NOP on retries
    nop_locked = {}  # pass_name -> True (must NOP) or False (must not NOP)

    for i in range(max_outer):
        outer_dir = ckpt_root / f"outer_{i}"
        print(f"\n--- Outer iteration {i + 1}/{max_outer} ---")

        # ============================================================
        # Phase 1: PyTorch lowering passes (tiler -> router -> retiler)
        # ============================================================
        lowered_code = None
        pipeline_ok = True
        for pass_info in LOWERING_PASSES:
            pass_name = pass_info["name"]

            # If locked to NOP, skip entirely
            if nop_locked.get(pass_name) is True:
                print(f"  Lowering pass: {pass_name} -> NOP (locked)")
                continue

            print(f"  Lowering pass: {pass_name}")
            pass_result = await _run_pass_loop(
                pass_agents[pass_name], pass_name, kernel_name, dims, max_turns,
                ckpt_dir=outer_dir,
                prev_code=lowered_code,
                diagnosis=diagnosis,
                executor="tiled",
                tensors=tensors,
                allow_nop=nop_locked.get(pass_name) is not False,
            )
            if pass_result["nop"]:
                nop_locked.setdefault(pass_name, True)
                print(f"    -> NOP")
            elif pass_result["success"]:
                nop_locked[pass_name] = False  # lock: must not NOP in future
                lowered_code = pass_result["code"]
                print(f"    -> OK")
            else:
                nop_locked[pass_name] = False  # it tried, so lock non-NOP
                print(f"    -> FAILED, stopping lowering pipeline")
                pipeline_ok = False
                break

        if not pipeline_ok or lowered_code is None:
            print(f"  Lowering pipeline failed")
            # Run analyst for next iteration
            if i < max_outer - 1:
                diagnosis = await _run_analyst(analyst_agent, outer_dir, kernel_name, dims, all_inner_results)
            continue

        print(f"  Lowering pipeline succeeded")

        # ============================================================
        # Phase 2: STeP translation passes
        # ============================================================
        translated_code = lowered_code
        translation_ok = True
        for pass_info in TRANSLATOR_PASSES:
            pass_name = pass_info["name"]
            executor = pass_info["executor"]
            config_allow_nop = pass_info.get("allow_nop", True)

            # If locked to NOP, skip entirely
            if nop_locked.get(pass_name) is True:
                print(f"  Translation pass: {pass_name} -> NOP (locked)")
                continue

            # Determine allow_nop: config can forbid it, or prior non-NOP locks it
            allow_nop = config_allow_nop and (nop_locked.get(pass_name) is not False)

            print(f"  Translation pass: {pass_name} (executor={executor})")
            pass_result = await _run_pass_loop(
                pass_agents[pass_name], pass_name, kernel_name, dims, max_turns,
                ckpt_dir=outer_dir,
                prev_code=translated_code,
                diagnosis=diagnosis,
                executor=executor,
                allow_nop=allow_nop,
                tensors=tensors,
            )
            if pass_result["nop"]:
                nop_locked.setdefault(pass_name, True)
                print(f"    -> NOP")
            elif pass_result["success"]:
                nop_locked[pass_name] = False
                translated_code = pass_result["code"]
                print(f"    -> OK")
            else:
                nop_locked[pass_name] = False
                print(f"    -> FAILED at {pass_name}")
                translation_ok = False
                break

        if translation_ok:
            # Verify the final code actually works as a build_graph
            final_code = translated_code
            print(f"  Translation pipeline completed — verifying final graph...")
            try:
                graph_result = _run_graph_correctness(final_code, kernel_name, dims, tensors)
                if "match=True" in graph_result:
                    print(f"  -> PASS (graph verified)")
                    result = _build_success_result(i, total_tool_calls,
                                                   {"code": final_code, "tool_outputs": []},
                                                   all_inner_results, lowered_code)
                    _save_experience(kernel_name, final_code,
                                     {"kernel": kernel_name, "preset": preset}, experience_dir)
                    _write(ckpt_root / "result.json", json.dumps(result, indent=2, default=str))
                    return result
                else:
                    print(f"  -> FAIL: final code doesn't pass graph correctness ({graph_result.splitlines()[0]})")
                    translation_ok = False
            except Exception:
                err_msg = traceback.format_exc().splitlines()[-1]
                print(f"  -> ERROR: final code isn't a valid build_graph ({err_msg})")
                translation_ok = False

        # Translation failed — retry on next outer iteration
        if not translation_ok:
            print(f"  Translation pipeline failed")

    # All outer iterations exhausted
    result = {
        "success": False,
        "outer_iterations": max_outer,
        "total_tool_calls": total_tool_calls,
        "cycle_count": None,
        "final_diagnosis": diagnosis,
        "tiled_code": lowered_code,
        "traces": [{"code": r.get("code"), "tool_outputs": r.get("tool_outputs", [])} for r in all_inner_results],
    }
    _write(ckpt_root / "result.json", json.dumps(result, indent=2, default=str))
    return result


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


async def _run_analyst(analyst_agent, outer_dir, kernel_name, dims, all_inner_results):
    from src.prompts import build_analyst_prompt

    print("  Running analyst...")
    analyst_dir = outer_dir / "analyst"

    traces = [{"code": r.get("code", ""), "tool_outputs": r.get("tool_outputs", [])} for r in all_inner_results]
    analyst_prompt = build_analyst_prompt(kernel_name, dims, traces)
    _write(analyst_dir / "prompt.txt", analyst_prompt)

    analyst_result = await Runner.run(analyst_agent, analyst_prompt)
    diagnosis = analyst_result.final_output
    _write(analyst_dir / "diagnosis.txt", diagnosis or "(empty)")
    print(f"  Diagnosis: {(diagnosis or '')[:200]}...")
    return diagnosis
