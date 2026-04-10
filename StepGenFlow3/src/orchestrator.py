"""Orchestrator: manages the outer loop, inner tool-execution loop, and experience library.

The inner loop extracts code from the LLM's text response, runs the functional
emulator and correctness checker automatically, and feeds results back as the
next conversation message.

Checkpoint structure:
  checkpoints/<timestamp>/<kernel>/outer_<N>/writer/turn_<M>/...
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

from src.agents import make_pass_agent, make_writer_agent, make_analyst_agent
from src.prompts import (LOWERING_PASSES, build_pass_system_prompt, build_pass_user_prompt,
                         build_writer_system_prompt, build_writer_user_prompt, build_analyst_prompt)
from src.tools import (_exec_build_graph, _exec_tiled_ref, _format_node_values,
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


# ---------------------------------------------------------------------------
# Code validation — allowlist approach: only permitted torch/F calls allowed
# ---------------------------------------------------------------------------

# Allowed torch.* calls — initialization, dtypes, and routing metadata only.
_ALLOWED_TORCH_CALLS = {
    # Tensor creation / initialization
    "torch.randn", "torch.zeros", "torch.ones", "torch.arange",
    "torch.tensor", "torch.empty", "torch.randint", "torch.full", "torch.eye",
    "torch.manual_seed",
    # Dtypes
    "torch.float32", "torch.float16", "torch.int64", "torch.int32", "torch.bool",
    # Routing metadata (control signals, not data-path computation)
    "torch.topk", "torch.softmax", "torch.stack", "torch.cat",
    # Stacking weight tensors for loading
    "torch.no_grad",
}


def _validate_no_compute(code: str) -> str:
    """Check that code only uses allowed torch/F calls.

    Uses an allowlist — any torch.* or F.* call not on the list is rejected.
    Returns empty string if OK, or an error message listing violations.
    """
    # Find all torch.xxx(...) calls (not attributes like torch.float32 used as dtype)
    torch_calls = set(re.findall(r'(torch\.[a-zA-Z_]+)\s*\(', code))
    # Also catch torch.nn.functional.xxx and torch.nn.xxx
    torch_nn_calls = set(re.findall(r'(torch\.nn\.[a-zA-Z_.]+)\s*\(', code))
    # Catch F.xxx calls
    f_calls = set(re.findall(r'(F\.[a-zA-Z_]+)\s*\(', code))

    violations = []

    for call in sorted(torch_calls):
        if call not in _ALLOWED_TORCH_CALLS:
            violations.append(f"  - {call}")

    for call in sorted(torch_nn_calls):
        violations.append(f"  - {call}")

    for call in sorted(f_calls):
        violations.append(f"  - {call}")

    if not violations:
        return ""

    return (
        "REJECTED: Your code uses torch/F calls that are not allowed in build_graph.\n"
        "All computation must happen inside STeP graph operators (BinaryMap, UnaryMap, Accum, etc.).\n"
        "Only these torch calls are permitted (for initialization and routing metadata):\n"
        f"  {', '.join(sorted(_ALLOWED_TORCH_CALLS))}\n\n"
        "Disallowed calls found:\n" + "\n".join(violations)
    )


# Minimum number of compute/shape/sink nodes for a real graph.
# A "cheat" graph that just loads a precomputed result has only 1-2 nodes.
_MIN_GRAPH_NODES = 4


def _validate_graph_complexity(graph) -> str:
    """Reject trivial graphs that are likely precomputed results.

    Returns empty string if OK, or an error message.
    """
    n_nodes = graph.number_of_nodes()
    if n_nodes < _MIN_GRAPH_NODES:
        return (
            f"REJECTED: Your graph has only {n_nodes} node(s). A real implementation "
            f"needs at least {_MIN_GRAPH_NODES} nodes (loads + compute ops + store). "
            f"It looks like you precomputed the result in PyTorch and just stored it. "
            f"All computation must happen inside STeP graph operators."
        )
    return ""


# ---------------------------------------------------------------------------
# Decomposer tool execution
# ---------------------------------------------------------------------------

def _run_tiled_correctness(code: str, kernel_name: str, dims: dict) -> str:
    """Run tiled_reference and compare against the PyTorch gold reference."""
    import torch

    result = _exec_tiled_ref(code, dims)

    config = _validate_functional_mod.load_config()
    gold = _validate_functional_mod.run_reference(kernel_name, dims, config)

    if gold.shape != result.shape:
        return f"SHAPE MISMATCH: gold {tuple(gold.shape)} vs tiled {tuple(result.shape)}\nmatch=False"

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
            f"\ntiled_value={result[worst_multi].item():.6e}"
        )
    return out


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _run_inspect(code: str, dims: dict) -> str:
    from step_py.functional import execute_values
    graph, _output_op = _exec_build_graph(code, dims)
    try:
        values = execute_values(graph)
    except Exception as exc:
        raise RuntimeError(enhance_emulator_error(exc, code)) from None
    return _format_node_values(values, graph)


def _run_correctness(code: str, kernel_name: str, dims: dict) -> str:
    import torch
    from step_py.functional import execute

    graph, output_op = _exec_build_graph(code, dims)
    try:
        sim = execute(graph, output_op)
    except Exception as exc:
        raise RuntimeError(enhance_emulator_error(exc, code)) from None

    config = _validate_functional_mod.load_config()
    gold = _validate_functional_mod.run_reference(kernel_name, dims, config)

    if gold.shape != sim.shape:
        return f"SHAPE MISMATCH: gold {tuple(gold.shape)} vs sim {tuple(sim.shape)}\nmatch=False"

    max_err = (gold - sim).abs().max().item()
    rel_err = max_err / (gold.abs().max().item() + 1e-12)
    match = rel_err < 1e-5

    result = f"match={match}\nmax_abs_err={max_err:.2e}\nrel_err={rel_err:.2e}\noutput_shape={tuple(sim.shape)}"
    if not match:
        diff = (gold - sim).abs()
        worst_multi = torch.unravel_index(diff.argmax(), diff.shape)
        result += (
            f"\nworst_error_index={tuple(i.item() for i in worst_multi)}"
            f"\ngold_value={gold[worst_multi].item():.6e}"
            f"\nsim_value={sim[worst_multi].item():.6e}"
        )
    return result


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
# Generic lowering pass loop
# ---------------------------------------------------------------------------

async def _run_pass_loop(agent, pass_name, kernel_name, dims, max_turns,
                         ckpt_dir: Path, prev_code=None, diagnosis=None):
    """Run a single lowering pass agent.

    Returns dict with success, code, nop.
    """
    system_prompt = build_pass_system_prompt(pass_name)
    user_prompt = build_pass_user_prompt(pass_name, kernel_name, dims,
                                         prev_code=prev_code, diagnosis=diagnosis)
    conversation = [{"role": "user", "content": user_prompt}]

    pass_dir = ckpt_dir / pass_name
    _write(pass_dir / "system_prompt.txt", system_prompt)

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

        code = _extract_code(assistant_text)
        if not code:
            print(f"      No code block found ({len(assistant_text)} chars). Stopping.")
            _write(turn_dir / "status.txt", "NO_CODE_EXTRACTED")
            break

        # Check for NOP
        if code.strip() == "NOP":
            print(f"      -> NOP (no transformation needed)")
            _write(turn_dir / "status.txt", "NOP")
            return {"success": True, "code": prev_code, "nop": True}

        last_code = code
        _write(turn_dir / "extracted_code.py", code)
        print(f"      Extracted code: {len(code)} chars")

        # Run tiled correctness check
        print(f"      Running correctness check...")
        feedback = ""
        try:
            result = _run_tiled_correctness(code, kernel_name, dims)
            _write(turn_dir / "correctness_result.txt", result)

            if "match=True" in result:
                success = True
                _write(turn_dir / "status.txt", "PASS")
                print(f"      -> PASS")
                break
            else:
                _write(turn_dir / "status.txt", f"FAIL: {result.splitlines()[0]}")
                print(f"      -> FAIL: {result.splitlines()[0]}")
                feedback = f"## Correctness check result\n{result}\n\nFix the issues and try again."
        except Exception:
            err = traceback.format_exc()
            _write(turn_dir / "correctness_result.txt", f"ERROR:\n{err}")
            print(f"      -> ERROR: {err.splitlines()[-1]}")
            feedback = f"## Error running tiled_reference\n{err}\n\nFix the error and try again."

        conversation.append({"role": "user", "content": feedback})

    return {"success": success, "code": last_code, "nop": False}


# ---------------------------------------------------------------------------
# Writer inner loop
# ---------------------------------------------------------------------------

async def _run_inner_loop(writer_agent, kernel_name, dims, max_turns,
                          ckpt_dir: Path, writer_system_prompt: str,
                          diagnosis=None, tiled_code=None):
    """Run the Writer agent inner loop with verbose checkpointing.

    Returns dict with success, code, cycle_count, tool_calls, tool_outputs.
    """
    user_prompt = build_writer_user_prompt(kernel_name, dims, diagnosis=diagnosis,
                                           tiled_code=tiled_code)
    conversation = [{"role": "user", "content": user_prompt}]

    # Save writer system prompt (same for all turns)
    _write(ckpt_dir / "writer" / "system_prompt.txt", writer_system_prompt)

    tool_outputs = []
    last_code = None
    success = False

    for turn in range(max_turns):
        turn_dir = ckpt_dir / "writer" / f"turn_{turn}"
        print(f"    Turn {turn + 1}/{max_turns}...")

        # Save the user prompt for this turn
        # For turn 0 it's the initial prompt; for later turns it's the feedback
        last_user_msg = conversation[-1]["content"] if conversation[-1]["role"] == "user" else ""
        _write(turn_dir / "user_prompt.txt", last_user_msg)

        # Call LLM
        run_result = await Runner.run(writer_agent, conversation)
        assistant_text = run_result.final_output or ""
        conversation.append({"role": "assistant", "content": assistant_text})

        # Save raw response
        _write(turn_dir / "response.txt", assistant_text)

        # Extract code
        code = _extract_code(assistant_text)
        if not code:
            print(f"      No code block found ({len(assistant_text)} chars). Stopping.")
            _write(turn_dir / "status.txt", "NO_CODE_EXTRACTED")
            break
        last_code = code
        _write(turn_dir / "extracted_code.py", code)
        print(f"      Extracted code: {len(code)} chars")

        # Validate: no computation outside STeP ops
        validation_err = _validate_no_compute(code)
        if validation_err:
            print(f"      REJECTED: computation outside graph")
            _write(turn_dir / "status.txt", "REJECTED")
            _write(turn_dir / "validation_error.txt", validation_err)
            tool_outputs.append(validation_err)
            conversation.append({"role": "user", "content": validation_err})
            continue

        # Run tools automatically
        feedback_parts = []

        # 1. execute_and_inspect
        print(f"      Running execute_and_inspect...")
        try:
            inspect_result = _run_inspect(code, dims)
            feedback_parts.append(f"## execute_and_inspect result\n{inspect_result}")
            tool_outputs.append(inspect_result)
            _write(turn_dir / "inspect_result.txt", inspect_result)
            print(f"      -> OK ({inspect_result.count(chr(10))} nodes)")
        except Exception:
            err = traceback.format_exc()
            feedback_parts.append(f"## execute_and_inspect ERROR\n{err}")
            tool_outputs.append(f"ERROR: {err}")
            _write(turn_dir / "inspect_result.txt", f"ERROR:\n{err}")
            print(f"      -> ERROR: {err.splitlines()[-1]}")
            conversation.append({"role": "user", "content": "\n\n".join(feedback_parts) + "\n\nFix the error and try again."})
            continue

        # 1b. Reject trivial precomputed graphs
        try:
            graph, _ = _exec_build_graph(code, dims)
            complexity_err = _validate_graph_complexity(graph)
            if complexity_err:
                print(f"      REJECTED: trivial graph ({graph.number_of_nodes()} nodes)")
                _write(turn_dir / "status.txt", "REJECTED_TRIVIAL")
                tool_outputs.append(complexity_err)
                conversation.append({"role": "user", "content": complexity_err})
                continue
        except Exception:
            pass  # graph build failed but inspect passed — shouldn't happen, skip check

        # 2. check_correctness
        print(f"      Running check_correctness...")
        try:
            correctness_result = _run_correctness(code, kernel_name, dims)
            feedback_parts.append(f"## check_correctness result\n{correctness_result}")
            tool_outputs.append(correctness_result)
            _write(turn_dir / "correctness_result.txt", correctness_result)

            if "match=True" in correctness_result:
                success = True
                _write(turn_dir / "status.txt", "PASS")
                print(f"      -> PASS")
                break
            else:
                _write(turn_dir / "status.txt", f"FAIL: {correctness_result.splitlines()[0]}")
                print(f"      -> FAIL: {correctness_result.splitlines()[0]}")
        except Exception:
            err = traceback.format_exc()
            feedback_parts.append(f"## check_correctness ERROR\n{err}")
            tool_outputs.append(f"ERROR: {err}")
            _write(turn_dir / "correctness_result.txt", f"ERROR:\n{err}")
            print(f"      -> ERROR: {err.splitlines()[-1]}")

        # Feed results back
        conversation.append({
            "role": "user",
            "content": "\n\n".join(feedback_parts) + "\n\nFix the issues and try again. Output the corrected build_graph function in a python code block."
        })

    return {
        "success": success,
        "code": last_code,
        "cycle_count": None,
        "tool_calls": len(tool_outputs),
        "tool_outputs": tool_outputs,
    }


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
    """Run the full outer loop for a single kernel + preset."""
    config = _load_stepdb_config()
    assert kernel_name in config, f"Kernel '{kernel_name}' not found"
    assert preset in config[kernel_name]["presets"], f"Preset '{preset}' not found"
    dims = config[kernel_name]["presets"][preset]

    pass_agents = {p["name"]: make_pass_agent(llm_config, p["name"]) for p in LOWERING_PASSES}
    writer_agent = make_writer_agent(llm_config)
    analyst_agent = make_analyst_agent(llm_config)
    writer_system_prompt = build_writer_system_prompt()

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

    for i in range(max_outer):
        outer_dir = ckpt_root / f"outer_{i}"
        print(f"\n--- Outer iteration {i + 1}/{max_outer} ---")

        # Phase 1: Lowering pass pipeline
        lowered_code = None
        pipeline_ok = True
        for pass_info in LOWERING_PASSES:
            pass_name = pass_info["name"]
            print(f"  Pass: {pass_name}")
            pass_result = await _run_pass_loop(
                pass_agents[pass_name], pass_name, kernel_name, dims, max_turns,
                ckpt_dir=outer_dir,
                prev_code=lowered_code,
                diagnosis=diagnosis,
            )
            if pass_result["nop"]:
                print(f"  {pass_name} -> NOP (skipped)")
            elif pass_result["success"]:
                lowered_code = pass_result["code"]
                print(f"  {pass_name} -> OK")
            else:
                print(f"  {pass_name} -> FAILED, stopping pipeline")
                pipeline_ok = False
                break

        tiled_code = lowered_code if pipeline_ok and lowered_code else None
        if tiled_code:
            print(f"  Pipeline succeeded — passing lowered blueprint to Writer")
        else:
            print(f"  Pipeline failed — Writer will proceed without blueprint")

        # Phase 2: Writer — construct STeP graph (with optional tiled blueprint)
        print("  Phase 2: Writer")
        inner = await _run_inner_loop(
            writer_agent, kernel_name, dims, max_turns,
            ckpt_dir=outer_dir,
            writer_system_prompt=writer_system_prompt,
            diagnosis=diagnosis,
            tiled_code=tiled_code,
        )
        all_inner_results.append(inner)
        total_tool_calls += inner["tool_calls"]

        if inner["success"]:
            result = {
                "success": True,
                "outer_iterations": i + 1,
                "total_tool_calls": total_tool_calls,
                "cycle_count": inner["cycle_count"],
                "final_diagnosis": None,
                "tiled_code": tiled_code,
                "traces": [{"code": r["code"], "tool_outputs": r["tool_outputs"]} for r in all_inner_results],
            }
            _save_experience(
                kernel_name, inner["code"],
                {"kernel": kernel_name, "preset": preset, "cycles": inner["cycle_count"]},
                experience_dir,
            )
            _write(ckpt_root / "result.json", json.dumps(result, indent=2, default=str))
            return result

        # Analyst diagnosis (if not last iteration)
        if i < max_outer - 1:
            print("  Running analyst...")
            analyst_dir = outer_dir / "analyst"

            traces = [{"code": r["code"], "tool_outputs": r["tool_outputs"]} for r in all_inner_results]
            analyst_prompt = build_analyst_prompt(kernel_name, dims, traces)
            _write(analyst_dir / "prompt.txt", analyst_prompt)

            analyst_result = await Runner.run(analyst_agent, analyst_prompt)
            diagnosis = analyst_result.final_output
            _write(analyst_dir / "diagnosis.txt", diagnosis or "(empty)")
            print(f"  Diagnosis: {(diagnosis or '')[:200]}...")

    result = {
        "success": False,
        "outer_iterations": max_outer,
        "total_tool_calls": total_tool_calls,
        "cycle_count": None,
        "final_diagnosis": diagnosis,
        "tiled_code": tiled_code,
        "traces": [{"code": r["code"], "tool_outputs": r["tool_outputs"]} for r in all_inner_results],
    }
    _write(ckpt_root / "result.json", json.dumps(result, indent=2, default=str))
    return result
