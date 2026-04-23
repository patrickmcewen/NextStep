"""Prompt construction for StepGenFlow7 — progressive translation pipeline.

Phase 1: PyTorch lowering passes (tiler, refactor_final)
Phase 2: STeP translation (single pass — DSL calls map 1:1 to STeP graph nodes)

Each phase produces verifiable intermediate code that matches gold.
"""
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROMPTS_DIR = _PROJECT_ROOT / "prompts"
_DEIO_ROOT = _PROJECT_ROOT.parent  # DEIOpt/
_STEPDB_DIR = _DEIO_ROOT / "StepDB"
_STEP_TL_SRC = _DEIO_ROOT / "step_tl" / "src"
_STEP_TL_PROTO = _STEP_TL_SRC / "proto"
_FUNCTIONAL_PY = _STEP_TL_SRC / "step_py" / "functional.py"
_TIMING_PY = _STEP_TL_SRC / "step_py" / "timing.py"

for p in (_STEP_TL_SRC, _STEP_TL_PROTO):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# ---------------------------------------------------------------------------
# Import IMPORT_SCAFFOLD from validate_functional via importlib
# ---------------------------------------------------------------------------
_vf_spec = importlib.util.spec_from_file_location(
    "validate_functional", str(_STEPDB_DIR / "validate_functional.py"))
_validate_functional_mod = importlib.util.module_from_spec(_vf_spec)
_vf_spec.loader.exec_module(_validate_functional_mod)

IMPORT_SCAFFOLD = _validate_functional_mod.IMPORT_SCAFFOLD

# ---------------------------------------------------------------------------
# Helper: load StepDB config
# ---------------------------------------------------------------------------
def _load_stepdb_config() -> dict:
    config_path = _STEPDB_DIR / "bench_config.yaml"
    assert config_path.exists(), f"bench_config.yaml not found at {config_path}"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Phase 1: PyTorch lowering passes
# ---------------------------------------------------------------------------
LOWERING_PASSES = [
    {"name": "refactor_final", "template": "refactor_final_system.txt", "executor": "dsl"},
]

# ---------------------------------------------------------------------------
# Phase 2: STeP translation (single pass — DSL calls map 1:1 to STeP nodes)
# ---------------------------------------------------------------------------
TRANSLATOR_PASSES = [
    {"name": "translate", "template": "translate_system.txt",
     "func_name": "build_graph", "executor": "graph"},
]

# ---------------------------------------------------------------------------
# Direct pipeline: skip lowering, go straight from PyTorch to STeP graph
# ---------------------------------------------------------------------------
DIRECT_TRANSLATOR_PASSES = [
    {"name": "translate_full", "template": "translate_system_full.txt",
     "func_name": "build_graph", "executor": "graph"},
]

# Pipeline configurations
PIPELINES = {
    "standard": {
        "lowering": LOWERING_PASSES,
        "translation": TRANSLATOR_PASSES,
    },
    "direct": {
        "lowering": [],
        "translation": DIRECT_TRANSLATOR_PASSES,
    },
}


# ---------------------------------------------------------------------------
# Public API — pass prompt builders
# ---------------------------------------------------------------------------

def build_pass_system_prompt(pass_name: str) -> str:
    """Build a lowering/translator pass agent's system prompt from its template.

    Templates contain {placeholder} tokens that are filled from source files:
      {ops_code}        — step_tl/src/step_py/ops.py
      {functional_code} — step_tl/src/step_py/functional.py
      {dsl_code}        — StepGenFlow7/src/step_dsl.py
    This keeps the prompts in sync with the actual source code automatically.
    """
    all_passes = LOWERING_PASSES + TRANSLATOR_PASSES + DIRECT_TRANSLATOR_PASSES
    pass_info = None
    for p in all_passes:
        if p["name"] == pass_name:
            pass_info = p
            break
    assert pass_info is not None, f"Unknown pass: {pass_name}. Known: {[p['name'] for p in all_passes]}"

    template_path = _PROMPTS_DIR / pass_info["template"]
    assert template_path.exists(), f"Template not found: {template_path}"
    template = template_path.read_text()

    # Inject source code from canonical files into placeholders
    replacements = {}
    if "{ops_code}" in template:
        ops_path = _STEP_TL_SRC / "step_py" / "ops.py"
        assert ops_path.exists(), f"ops.py not found: {ops_path}"
        replacements["ops_code"] = ops_path.read_text()#_strip_ops_boilerplate(ops_path.read_text())
    if "{utility_ops_code}" in template:
        utility_ops_path = _STEP_TL_SRC / "step_py" / "utility_ops.py"
        assert utility_ops_path.exists(), f"utility_ops.py not found: {utility_ops_path}"
        replacements["utility_ops_code"] = utility_ops_path.read_text()#_strip_ops_boilerplate(utility_ops_path.read_text())
    if "{functional_code}" in template:
        assert _FUNCTIONAL_PY.exists(), f"functional.py not found: {_FUNCTIONAL_PY}"
        replacements["functional_code"] = _FUNCTIONAL_PY.read_text()
    if "{dsl_code}" in template:
        dsl_path = _PROJECT_ROOT / "src" / "step_dsl.py"
        assert dsl_path.exists(), f"step_dsl.py not found: {dsl_path}"
        replacements["dsl_code"] = dsl_path.read_text()

    if replacements:
        template = template.format(**replacements)

    return template


# Judge prompt templates — keyed by pass name
_JUDGE_TEMPLATES = {
    "refactor_final": "refactor_final_judge_system.txt",
    "translate": "translate_judge_system.txt",
    "translate_full": "translate_judge_system.txt",
}


def build_autotune_system_prompt(hw_constraints: dict,
                                  template_name: str = "autotune_system.txt") -> str:
    """Build the autotuner agent's system prompt.

    Injects the full source of step_py/timing.py at {timing_code} so the LLM
    knows exactly how each knob maps to total_cycles, and the caller-supplied
    `hw_constraints` dict (rendered as JSON) at {hw_constraints}.

    `template_name` selects which autotune prompt to use (e.g.,
    "autotune_system.txt" for the generalist, "autotune_parallel_system.txt"
    for the parallelism specialist). All templates share the same placeholder
    set so injection logic is identical.
    """
    template_path = _PROMPTS_DIR / template_name
    assert template_path.exists(), f"autotune prompt not found: {template_path}"
    assert _TIMING_PY.exists(), f"timing.py not found: {_TIMING_PY}"
    template = template_path.read_text()
    replacements = {}
    if "{ops_code}" in template:
        ops_path = _STEP_TL_SRC / "step_py" / "ops.py"
        assert ops_path.exists(), f"ops.py not found: {ops_path}"
        replacements["ops_code"] = ops_path.read_text()
    if "{utility_ops_code}" in template:
        utility_ops_path = _STEP_TL_SRC / "step_py" / "utility_ops.py"
        assert utility_ops_path.exists(), f"utility_ops.py not found: {utility_ops_path}"
        replacements["utility_ops_code"] = utility_ops_path.read_text()
    if "{functional_code}" in template:
        assert _FUNCTIONAL_PY.exists(), f"functional.py not found: {_FUNCTIONAL_PY}"
        replacements["functional_code"] = _FUNCTIONAL_PY.read_text()
    if "{timing_code}" in template:
        assert _TIMING_PY.exists(), f"timing.py not found: {_TIMING_PY}"
        replacements["timing_code"] = _TIMING_PY.read_text()
    if "{hw_constraints}" in template:
        replacements["hw_constraints"] = json.dumps(hw_constraints, indent=2)
    return template.format(**replacements)


def build_autotune_user_prompt(kernel_name: str, dims: dict, build_graph_code: str,
                                timing_report: str, baseline_cycles: int,
                                best_cycles: int) -> str:
    """Build the autotuner user prompt for a single turn.

    `timing_report` is the pretty-printed analyze_timing() output for the
    current build_graph. `baseline_cycles` is the cycle count at the start
    of the tuning run; `best_cycles` is the best we've seen so far.
    """
    return "\n".join([
        f"## Kernel: {kernel_name}",
        "",
        "### Dimensions",
        "",
        "```json",
        json.dumps(dims, indent=2),
        "```",
        "",
        f"### Baseline total_cycles: {baseline_cycles}",
        f"### Best so far:          {best_cycles}",
        "",
        "### Current build_graph (correctness verified)",
        "",
        "```python",
        build_graph_code.rstrip(),
        "```",
        "",
        "### Current timing report",
        "",
        "```",
        timing_report.rstrip(),
        "```",
        "",
        "Propose a change that reduces total_cycles. Output the full updated "
        "`build_graph(dims, tensors)` in a single ```python block.",
    ])


def build_judge_system_prompt(pass_name: str) -> str:
    """Build a judge agent's system prompt from its template."""
    assert pass_name in _JUDGE_TEMPLATES, f"No judge template for pass '{pass_name}'"
    template_path = _PROMPTS_DIR / _JUDGE_TEMPLATES[pass_name]
    assert template_path.exists(), f"Judge template not found: {template_path}"
    return template_path.read_text()


def _format_tensors_description(tensors: dict) -> str:
    """Format a human-readable description of the tensors dict for the prompt."""
    lines = []
    for key, val in tensors.items():
        if isinstance(val, list):
            if len(val) > 0 and hasattr(val[0], "shape"):
                lines.append(f'  "{key}": list of {len(val)} tensors, each shape {tuple(val[0].shape)}')
            else:
                lines.append(f'  "{key}": list of {len(val)} items')
        elif hasattr(val, "shape"):
            lines.append(f'  "{key}": tensor shape {tuple(val.shape)}, dtype {val.dtype}')
        else:
            lines.append(f'  "{key}": {type(val).__name__} = {val}')
    return "\n".join(lines)


def build_pass_user_prompt(pass_name: str, kernel_name: str, dims: dict,
                           prev_code: str = None,
                           tensors: dict = None,
                           dsl_code: str = None) -> str:
    """Build a pass's user prompt with reference code, dims, and previous output.

    Args:
        dsl_code: DSL-refactored code (from refactor pass). Passed to translator passes
                  as a translation guide — each DSL call maps 1:1 to a STeP node.
    """
    config = _load_stepdb_config()
    assert kernel_name in config, f"Kernel '{kernel_name}' not found in bench_config.yaml"

    ref_path = _STEPDB_DIR / config[kernel_name]["problem"]
    assert ref_path.exists(), f"Reference file not found: {ref_path}"
    reference_code = ref_path.read_text()

    dims_json = json.dumps(dims, indent=2)

    # Determine which function name this pass expects
    pass_info = None
    for p in LOWERING_PASSES + TRANSLATOR_PASSES + DIRECT_TRANSLATOR_PASSES:
        if p["name"] == pass_name:
            pass_info = p
            break
    assert pass_info is not None

    is_translator = pass_info in TRANSLATOR_PASSES
    func_name = pass_info.get("func_name", "tiled_reference")

    lines = [
        f"## Kernel: {kernel_name}",
        "",
        "### Original PyTorch Reference",
        "",
        "```python",
        reference_code.rstrip(),
        "```",
        "",
        "### Dimensions",
        "",
        "```json",
        dims_json,
        "```",
    ]

    # Add tensors dict description
    if tensors is not None:
        lines.extend([
            "",
            "### Pre-computed Tensors",
            "",
            "Your function receives a `tensors` dict as its second argument with these entries:",
            "",
            "```",
            _format_tensors_description(tensors),
            "```",
            "",
            "**You MUST NOT call `torch.manual_seed`, `torch.randn`, `torch.rand`, or use the `@` operator.**",
            "All tensors are pre-created. Access them via `tensors[\"key\"]`.",
        ])

    if prev_code is not None:
        label = "Current Code (from previous pass \u2014 verified correct)"
        lines.extend([
            "",
            f"### {label}",
            "",
            "```python",
            prev_code.rstrip(),
            "```",
            "",
        ])

        sig = f"{func_name}(dims, tensors)" if tensors is not None else f"{func_name}(dims)"
        lines.append(
            f"Transform this code according to your instructions. "
            f"Output a `{sig}` function."
        )
    else:
        sig = f"{func_name}(dims, tensors)" if tensors is not None else f"{func_name}(dims)"
        lines.extend([
            "",
            f"Rewrite this as a `{sig}` function that computes the same result.",
        ])

    lines.append("I will automatically run your code and compare the output against the reference. Above the function definition, include a comment detailing your thought process for your implementation or fix.")

    return "\n".join(lines)