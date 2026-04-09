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

from src.agents import make_writer_agent, make_analyst_agent
from src.prompts import build_writer_system_prompt, build_writer_user_prompt, build_analyst_prompt
from src.tools import _exec_build_graph, _format_node_values, _validate_functional_mod, enhance_emulator_error

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
# Code validation — reject computation outside STeP ops
# ---------------------------------------------------------------------------

# Allowed torch calls (initialization / metadata only)
_ALLOWED_TORCH = {
    "torch.randn", "torch.zeros", "torch.ones", "torch.arange",
    "torch.manual_seed", "torch.tensor", "torch.empty",
    "torch.randint", "torch.full", "torch.eye",
    "torch.float32", "torch.float16", "torch.int64", "torch.bool",
}

# Forbidden patterns — computation that must live inside STeP ops.
# Routing metadata (topk, softmax for expert weights, @/matmul for router logits)
# is allowed since it produces control signals, not final outputs.
_FORBIDDEN_PATTERNS = [
    (r'F\.silu\b', "F.silu (use UnaryMap with map_fn.Silu())"),
    (r'F\.relu\b', "F.relu"),
    (r'F\.gelu\b', "F.gelu"),
    (r'F\.softmax\b', "F.softmax (decompose into Exp/RowWiseSum/Div in STeP)"),
    (r'torch\.sigmoid\b', "torch.sigmoid"),
    (r'torch\.sqrt\b', "torch.sqrt (use UnaryMap with map_fn.Rsqrt())"),
    (r'torch\.rsqrt\b', "torch.rsqrt (use UnaryMap with map_fn.Rsqrt())"),
    (r'torch\.log\b', "torch.log"),
    (r'torch\.where\b', "torch.where"),
    (r'torch\.gather\b', "torch.gather (use FlatPartition)"),
    (r'torch\.scatter\b', "torch.scatter (use FlatReassemble)"),
]


def _validate_no_compute(code: str) -> str:
    """Check that code doesn't do computation outside STeP ops.

    Returns empty string if OK, or an error message listing violations.
    """
    violations = []
    for pattern, description in _FORBIDDEN_PATTERNS:
        matches = re.findall(pattern, code)
        if matches:
            violations.append(f"  - {description} (found {len(matches)} occurrence(s))")

    if not violations:
        return ""

    return (
        "REJECTED: Your code performs computation outside of STeP graph operators.\n"
        "All computation must happen inside STeP ops (BinaryMap, UnaryMap, Accum, etc.).\n"
        "You may only use PyTorch for tensor initialization (torch.randn, torch.zeros, etc.).\n\n"
        "Violations found:\n" + "\n".join(violations)
    )


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
# Inner loop
# ---------------------------------------------------------------------------

async def _run_inner_loop(writer_agent, kernel_name, dims, max_turns,
                          ckpt_dir: Path, writer_system_prompt: str,
                          diagnosis=None):
    """Run the Writer agent inner loop with verbose checkpointing.

    Returns dict with success, code, cycle_count, tool_calls, tool_outputs.
    """
    user_prompt = build_writer_user_prompt(kernel_name, dims, diagnosis=diagnosis)
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

        inner = await _run_inner_loop(
            writer_agent, kernel_name, dims, max_turns,
            ckpt_dir=outer_dir,
            writer_system_prompt=writer_system_prompt,
            diagnosis=diagnosis,
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
        "traces": [{"code": r["code"], "tool_outputs": r["tool_outputs"]} for r in all_inner_results],
    }
    _write(ckpt_root / "result.json", json.dumps(result, indent=2, default=str))
    return result
