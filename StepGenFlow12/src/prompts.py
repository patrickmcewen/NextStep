"""Prompt construction for StepGenFlow.

Phase 1: PyTorch -> DSL form via the ``refactor_final`` pass (gated by the
``dsl`` executor; under ``--translator=auto`` also gated by deterministic
translation as a post-validator).

Phase 2: DSL -> STeP graph. Either the deterministic AST translator
(``--translator=auto``, no LLM call) or an LLM ``translate`` pass
(``--translator=llm``). The ``direct`` pipelines collapse phases 1+2 into a
single ``translate_full`` LLM pass.
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
_FUNCTIONAL_PY = _STEP_TL_SRC / "timing_and_emulator" / "functional.py"
_TIMING_PY = _STEP_TL_SRC / "timing_and_emulator" / "timing.py"
_STEP_DSL_PY = _PROJECT_ROOT / "src" / "step_dsl.py"
_STEP_DSL_1X1_PY = _PROJECT_ROOT / "src" / "step_dsl_1x1.py"
_STEP_DSL_MEM_PY = _PROJECT_ROOT / "src" / "step_dsl_memory.py"

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

DIRECT_TRANSLATOR_PASSES_NO_FUNCTIONAL = [
    {"name": "translate_full_no_functional", "template": "translate_system_full_no_functional.txt",
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
    "direct_no_functional": {
        "lowering": [],
        "translation": DIRECT_TRANSLATOR_PASSES_NO_FUNCTIONAL,
    },
}


# ---------------------------------------------------------------------------
# Public API — pass prompt builders
# ---------------------------------------------------------------------------

def _resolve_few_shot_example(path: str) -> dict:
    """Resolve a path to a single PyTorch→DSL few-shot example.

    Accepts:
      - Path to a dsl_code.py file (e.g. .../outer_N/dsl_code.py)
      - Path to an outer_N directory containing dsl_code.py
      - Path to a checkpoint root containing config.json and
        <kernel_name>/outer_*/dsl_code.py — picks the latest outer_N with a
        dsl_code.py file.

    Returns a dict ``{kernel_name, pytorch_ref, dsl_code}``. The PyTorch
    reference is loaded from StepDB via the kernel's bench_config entry.
    """
    p = Path(path)
    assert p.exists(), f"Few-shot path does not exist: {path}"

    if p.suffix == ".py" and p.is_file():
        dsl_path = p
    elif p.is_dir() and (p / "dsl_code.py").is_file():
        dsl_path = p / "dsl_code.py"
    elif p.is_dir():
        candidates = sorted(p.glob("*/outer_*/dsl_code.py"))
        assert candidates, (
            f"No <kernel>/outer_*/dsl_code.py found under {p}. "
            f"Pass either a dsl_code.py file, an outer_N directory, "
            f"or a checkpoint root directory."
        )
        dsl_path = candidates[-1]
    else:
        assert False, f"Cannot resolve few-shot path: {path}"

    # Walk up from dsl_path to find a config.json with the kernel name.
    kernel_name = None
    cur = dsl_path.parent
    for _ in range(5):
        cfg = cur / "config.json"
        if cfg.is_file():
            cfg_data = json.loads(cfg.read_text())
            if "kernel" in cfg_data:
                kernel_name = cfg_data["kernel"]
                break
        cur = cur.parent
    if kernel_name is None:
        # Fallback: outer_N's parent dir is the kernel name.
        kernel_name = dsl_path.parent.parent.name

    config = _load_stepdb_config()
    assert kernel_name in config, (
        f"Kernel '{kernel_name}' (resolved from {path}) not in StepDB bench_config.yaml"
    )
    ref_path = _STEPDB_DIR / config[kernel_name]["problem"]
    assert ref_path.exists(), f"PyTorch reference not found: {ref_path}"

    return {
        "kernel_name": kernel_name,
        "pytorch_ref": ref_path.read_text(),
        "dsl_code": dsl_path.read_text(),
    }


def resolve_few_shot_examples(paths) -> list:
    """Resolve a list of path strings into few-shot example dicts.

    Pass ``None`` or an empty list when no examples are configured.
    """
    if not paths:
        return []
    return [_resolve_few_shot_example(p) for p in paths]


def _format_few_shot_examples(examples: list) -> str:
    """Render few-shot example dicts as a markdown section for the prompt.

    Returns "" when ``examples`` is empty so the placeholder collapses cleanly.
    """
    if not examples:
        return ""
    lines = [
        "",
        "## Few-shot examples",
        "",
        "Below are previously completed PyTorch → DSL translations for other "
        "kernels. Use them as reference for how operations should be lowered "
        "into DSL form. Each example shows the original PyTorch reference and "
        "the resulting DSL code.",
        "",
    ]
    for ex in examples:
        lines.extend([
            f"### Example: {ex['kernel_name']}",
            "",
            "PyTorch reference:",
            "```python",
            ex["pytorch_ref"].rstrip(),
            "```",
            "",
            "DSL form:",
            "```python",
            ex["dsl_code"].rstrip(),
            "```",
            "",
            "---",
            "",
        ])
    return "\n".join(lines)


def build_pass_system_prompt(pass_name: str, few_shot_examples=None) -> str:
    """Build a lowering/translator pass agent's system prompt from its template.

    Templates contain {placeholder} tokens that are filled from source files:
      {ops_code}           — step_tl/src/step_py/ops.py
      {functional_code}    — step_tl/src/timing_and_emulator/functional.py
      {dsl_code}           — StepGenFlow8/src/step_dsl.py
      {few_shot_examples}  — optional PyTorch→DSL example pairs (refactor_final)
    This keeps the prompts in sync with the actual source code automatically.

    ``few_shot_examples`` is an optional list of dicts as returned by
    ``resolve_few_shot_examples`` — each containing ``kernel_name``,
    ``pytorch_ref``, and ``dsl_code``. Templates without the
    ``{few_shot_examples}`` placeholder ignore this argument.
    """
    all_passes = LOWERING_PASSES + TRANSLATOR_PASSES + DIRECT_TRANSLATOR_PASSES + DIRECT_TRANSLATOR_PASSES_NO_FUNCTIONAL
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
    if "{import_scaffold}" in template:
        replacements["import_scaffold"] = IMPORT_SCAFFOLD
    if "{ops_code}" in template:
        ops_path = _STEP_TL_SRC / "step_py" / "ops.py"
        assert ops_path.exists(), f"ops.py not found: {ops_path}"
        replacements["ops_code"] = ops_path.read_text()#_strip_ops_boilerplate(ops_path.read_text())
    if "{utility_ops_code}" in template:
        utility_ops_path = _STEP_TL_SRC / "step_py" / "utility_ops.py"
        assert utility_ops_path.exists(), f"utility_ops.py not found: {utility_ops_path}"
        replacements["utility_ops_code"] = utility_ops_path.read_text()#_strip_ops_boilerplate(utility_ops_path.read_text())
    if "{accum_code}" in template:
        accum_path = _STEP_TL_SRC / "step_py" / "functions" / "accum_fn.py"
        assert accum_path.exists(), f"accum_fn.py not found: {accum_path}"
        replacements["accum_code"] = accum_path.read_text()
    if "{init_code}" in template:
        init_path = _STEP_TL_SRC / "step_py" / "functions" / "init_fn.py"
        assert init_path.exists(), f"init_fn.py not found: {init_path}"
        replacements["init_code"] = init_path.read_text()
    if "{map_accum_code}" in template:
        map_accum_path = _STEP_TL_SRC / "step_py" / "functions" / "map_accum_fn.py"
        assert map_accum_path.exists(), f"map_accum_fn.py not found: {map_accum_path}"
        replacements["map_accum_code"] = map_accum_path.read_text()
    if "{map_code}" in template:
        map_path = _STEP_TL_SRC / "step_py" / "functions" / "map_fn.py"
        assert map_path.exists(), f"map_fn.py not found: {map_path}"
        replacements["map_code"] = map_path.read_text()
    for fn_name in ("init_fn", "map_fn", "accum_fn", "map_accum_fn"):
        placeholder = "{" + fn_name + "_code}"
        if placeholder in template:
            fn_path = _STEP_TL_SRC / "step_py" / "functions" / f"{fn_name}.py"
            assert fn_path.exists(), f"{fn_name}.py not found: {fn_path}"
            replacements[f"{fn_name}_code"] = fn_path.read_text()
    if "{functional_code}" in template:
        assert _FUNCTIONAL_PY.exists(), f"functional.py not found: {_FUNCTIONAL_PY}"
        replacements["functional_code"] = _FUNCTIONAL_PY.read_text()
    if "{dsl_code}" in template:
        dsl_path = _PROJECT_ROOT / "src" / "step_dsl.py"
        assert dsl_path.exists(), f"step_dsl.py not found: {dsl_path}"
        replacements["dsl_code"] = dsl_path.read_text()
    if "{few_shot_examples}" in template:
        replacements["few_shot_examples"] = _format_few_shot_examples(
            few_shot_examples or [])

    if replacements:
        template = template.format(**replacements)

    return template


# Judge prompt templates — keyed by pass name
_JUDGE_TEMPLATES = {
    "refactor_final": "refactor_final_judge_system.txt",
    "translate": "translate_judge_system.txt",
    "translate_full": "translate_judge_system.txt",
    "translate_full_no_functional": "translate_judge_system.txt",
}


def build_autotune_system_prompt(hw_constraints: dict,
                                  template_name: str = "autotune_system.txt") -> str:
    """Build the autotuner agent's system prompt.

    Embeds the DSL surface (step_dsl.py) at {step_dsl_code} and the timing
    model source (timing.py) at {timing_code}, with caller-supplied
    `hw_constraints` (rendered as JSON) at {hw_constraints}. The autotuner
    edits ``tiled_reference(dims, tensors)`` (DSL form) — the deterministic
    translator runs inside its per-turn loop to produce the build_graph the
    timing model scores.

    `template_name` selects which autotune prompt to use (e.g.,
    "autotune_system.txt" for the generalist, "autotune_parallel_system.txt"
    for the parallelism specialist). All templates share the same placeholder
    set so injection logic is identical.
    """
    template_path = _PROMPTS_DIR / template_name
    assert template_path.exists(), f"autotune prompt not found: {template_path}"
    assert _TIMING_PY.exists(), f"timing.py not found: {_TIMING_PY}"
    assert _STEP_DSL_PY.exists(), f"step_dsl.py not found: {_STEP_DSL_PY}"
    template = template_path.read_text()
    replacements = {}
    if "{step_dsl_code}" in template:
        replacements["step_dsl_code"] = _STEP_DSL_PY.read_text()
    if "{step_dsl_memory_code}" in template:
        replacements["step_dsl_memory_code"] = _STEP_DSL_MEM_PY.read_text()
    if "{timing_code}" in template:
        replacements["timing_code"] = _TIMING_PY.read_text()
    if "{hw_constraints}" in template:
        replacements["hw_constraints"] = json.dumps(hw_constraints, indent=2)
    return template.format(**replacements)


def build_autotune_user_prompt(kernel_name: str, dims: dict, build_graph_code: str,
                                timing_report: str, baseline_cycles: int,
                                best_cycles: int,
                                feasibility: dict | None = None) -> str:
    """Build the autotuner user prompt for a single turn.

    `build_graph_code` parameter name is kept for back-compat at the call site,
    but the body is now the DSL-form ``tiled_reference`` source.
    `timing_report` is the pretty-printed analyze_timing() output for the
    *translated* build_graph. `baseline_cycles` is the cycle count at the start
    of the tuning run; `best_cycles` is the best we've seen so far.
    `feasibility`, if provided, is a dict of mem_info-field -> upper bound that
    the proposal must satisfy for the pass to be considered feasible.
    """
    sections = [
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
        "### Current tiled_reference (correctness verified)",
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
        "If the current best is not feasible, propose a change that restores feasibility or reduces infeasibility. Otherwise, propose a change that reduces total_cycles", 
        " while maintaining feasibility. Output the full updated ",
        "`tiled_reference(dims, tensors)` in a single ```python block.",
    ]
    if feasibility:
        bullets = "\n".join(f"  - {k} <= {v}" for k, v in feasibility.items())
        sections.extend([
            "",
            "## Hard feasibility constraints",
            "Your final proposal MUST satisfy:",
            bullets,
            "Proposals that violate these may still be tried, but the pass "
            "will be marked infeasible if no proposal satisfies all bounds.",
        ])
    return "\n".join(sections)


def build_judge_system_prompt(pass_name: str) -> str:
    """Build a judge agent's system prompt from its template."""
    assert pass_name in _JUDGE_TEMPLATES, f"No judge template for pass '{pass_name}'"
    template_path = _PROMPTS_DIR / _JUDGE_TEMPLATES[pass_name]
    assert template_path.exists(), f"Judge template not found: {template_path}"
    return template_path.read_text()


_BUNDLE_JUDGE_TEMPLATE = "bundle_refactor_judge_system.txt"


def build_bundle_judge_system_prompt(compliance: dict) -> str:
    """Build the bundle-mode judge system prompt from a compliance config.

    `compliance` is the dict persisted in bundle manifest.json
    (see ``flowv1/src/bundle._normalize_compliance`` for the schema).
    The template carries placeholders {allowed_ops_block}, {banned_patterns_block},
    {required_ops_block} that get filled with bullet-list renderings of the
    config — keeps the judge wholly vocabulary-driven.
    """
    template_path = _PROMPTS_DIR / _BUNDLE_JUDGE_TEMPLATE
    assert template_path.exists(), f"Bundle judge template not found: {template_path}"

    allowed = compliance.get("allowed_ops") or []
    banned = compliance.get("banned_patterns") or []
    required = compliance.get("required_ops") or []

    if allowed:
        allowed_block = "\n".join(f"- `{name}`" for name in allowed)
    else:
        allowed_block = "(no allowlist — any callable name is permitted)"

    if banned:
        banned_block = "\n".join(
            f"- `{e['pattern']}` — {e['fix']}" for e in banned
        )
    else:
        banned_block = "(no extra banned patterns)"

    if required:
        required_block = "\n".join(f"- `{name}`" for name in required)
    else:
        required_block = "(no required operators)"

    template = template_path.read_text()
    return template.format(
        allowed_ops_block=allowed_block,
        banned_patterns_block=banned_block,
        required_ops_block=required_block,
    )


def _get_precompute_source(kernel_name: str) -> str:
    """Extract the source of the precompute.py function registered for `kernel_name`.

    Finds the @register(kernel_name) decorator via AST and returns the full
    function source including all of its @register(...) decorators (one
    function may handle several kernels).
    """
    precompute_path = _STEPDB_DIR / "precompute.py"
    assert precompute_path.exists(), f"precompute.py not found: {precompute_path}"
    source = precompute_path.read_text()
    source_lines = source.splitlines()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "register"
                    and len(dec.args) == 1
                    and isinstance(dec.args[0], ast.Constant)
                    and dec.args[0].value == kernel_name):
                start = node.decorator_list[0].lineno
                end = node.end_lineno
                return "\n".join(source_lines[start - 1:end])

    assert False, f"No @register('{kernel_name}') found in {precompute_path}"


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
                           dsl_code: str = None,
                           reference_code_override: str | None = None,
                           precompute_source_override: str | None = None) -> str:
    """Build a pass's user prompt with reference code, dims, and previous output.

    ``reference_code_override`` (when set) replaces the bench_config-loaded
    reference. ``precompute_source_override`` replaces the StepDB precompute
    source shown to the model. Used by the planner flow for tree-node passes
    where the kernel doesn't live in bench_config.

    Args:
        dsl_code: DSL-refactored code (from refactor pass). Passed to translator passes
                  as a translation guide — each DSL call maps 1:1 to a STeP node.
    """
    if reference_code_override is not None:
        reference_code = reference_code_override
    else:
        config = _load_stepdb_config()
        assert kernel_name in config, f"Kernel '{kernel_name}' not found in bench_config.yaml"
        ref_path = _STEPDB_DIR / config[kernel_name]["problem"]
        assert ref_path.exists(), f"Reference file not found: {ref_path}"
        reference_code = ref_path.read_text()

    dims_json = json.dumps(dims, indent=2)

    # Determine which function name this pass expects
    pass_info = None
    for p in LOWERING_PASSES + TRANSLATOR_PASSES + DIRECT_TRANSLATOR_PASSES + DIRECT_TRANSLATOR_PASSES_NO_FUNCTIONAL:
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
        if precompute_source_override is not None:
            precompute_src = precompute_source_override
        else:
            precompute_src = _get_precompute_source(kernel_name)
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
            "These tensors are produced by the following precompute function (from `StepDB/precompute.py`):",
            "",
            "```python",
            precompute_src,
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


def build_pass1_user_prompt(
    *,
    node_name: str,
    is_root: bool,
    reference_code: str,
    dims: dict,
    tensors: dict,
    contract,            # src.contract.Contract for non-root, None for root
    children_signatures: list[tuple[
        str,                                # child_name
        tuple[str, ...],                    # arg_names
        tuple,                              # arg_specs (ArgSpec per arg)
        tuple[tuple[int, ...], ...],        # out_shapes (one entry per output)
        bool,                               # out_is_tuple
    ]],
    function_signature: str,
) -> str:
    """Build the Pass-1 user prompt for a single planner node.

    Parameters
    ----------
    node_name:
        The planner node's name (e.g. "attention"). Used as the function name
        for non-root nodes.
    is_root:
        True if this is the root node (function signature is
        ``tiled_reference(dims, tensors)``).
    reference_code:
        The PyTorch reference for this node (the planner node's sub-Model
        forward or the full reference for root).
    dims:
        Dimension dict (scalar ints, e.g. ``{"B": 2, "H": 8}``).
    tensors:
        Tensors dict available at the node level (may be empty for non-root).
    contract:
        ``Contract`` for non-root nodes, ``None`` for root.
    children_signatures:
        List of ``(child_name, arg_names, arg_specs, out_shapes,
        out_is_tuple)`` 5-tuples, one per declared child. ``arg_specs``
        is a tuple of ``ArgSpec`` values (one per arg) — see
        ``src.node_signature``. Empty for leaf nodes.
    function_signature:
        The exact signature string the LLM must produce (e.g.
        ``"def attention(Q, *, out_shapes):"``).
    """
    dims_json = json.dumps(dims, indent=2)

    has_children = bool(children_signatures)
    if has_children:
        ref_header = "### PyTorch Reference (planner-decomposed parent)"
        ref_note = (
            "This is the planner's reference decomposition. Its "
            "`self.<child_name>(args)` call sites correspond 1:1 to the "
            "blackboxes listed below; replacing them with "
            "`<child_name>(args, out_shapes=...)` is the expected default. "
            "Inlining a child's logic directly into your DSL is also "
            "acceptable when it produces correct output — see the "
            "'Child Blackboxes Available' section. Do not change the "
            "function signature."
        )
    else:
        ref_header = "### PyTorch Reference"
        ref_note = None
    lines = [
        f"## Node: {node_name}",
        "",
        ref_header,
        "",
        "```python",
        reference_code.rstrip(),
        "```",
    ]
    if ref_note is not None:
        lines.extend(["", ref_note])
    lines.extend([
        "",
        "### Dimensions",
        "",
        "```json",
        dims_json,
        "```",
    ])

    # Tensors dict — only shown when non-empty (root always has tensors; non-root may not)
    if tensors:
        lines.extend([
            "",
            "### Available Tensors",
            "",
            "```",
            _format_tensors_description(tensors),
            "```",
        ])

    # Contract block — non-root only
    if not is_root:
        assert contract is not None, "Non-root node must have a Contract"
        assert len(contract.arg_is_raw) == len(contract.arg_names), (
            "Non-root contract must have arg_is_raw stamped (length must "
            f"match arg_names); got arg_is_raw={contract.arg_is_raw!r}, "
            f"arg_names={contract.arg_names!r}"
        )
        # Leaves have no children, so the "may pass to child blackbox" /
        # "or child blackbox in parent's body" clauses are stripped.
        raw_tag = (
            "  - **RAW**: the parent forwarded an off-chip tensor that has "
            "not yet passed through a DSL source operator. Before feeding it "
            "to any DSL consumer (binary_*, unary_*, accum_*, reshape_*, "
            "promote_*, …) you must load it with `offchip_load` (or another "
            "producer like `select_gen` / `metadata_gen`)."
        )
        if has_children:
            raw_tag += " It may still be passed straight to a child blackbox without loading."
        onchip_tag = (
            "  - **on-chip**: already produced by a sibling DSL op"
            + (" or child blackbox" if has_children else "")
            + " in the parent's body — pass it directly to consumers."
        )
        lines.extend([
            "",
            "### Contract (declared by parent)",
            "",
            "Your parent has called your stub with these inputs. You must handle",
            "exactly this call site — do not change the function signature.",
            "",
            "Each input is tagged **RAW** or **on-chip**:",
            raw_tag,
            onchip_tag,
            "",
        ])
        from src.node_signature import (
            ListOfIntArg as _ListOfIntArg,
            ListOfTensorArg as _ListOfTensorArg,
            TensorArg as _TensorArg,
        )
        for arg_name, spec, tiled_shape, raw in zip(
            contract.arg_names, contract.arg_specs, contract.tiled_shapes,
            contract.arg_is_raw,
        ):
            tag = "**RAW**" if raw else "**on-chip**"
            if isinstance(spec, _TensorArg):
                lines.append(
                    f"  `{arg_name}`: vanilla shape {spec.shape}, "
                    f"tiled shape declared by parent {tiled_shape} — {tag}"
                )
            elif isinstance(spec, _ListOfTensorArg):
                lines.append(
                    f"  `{arg_name}`: list[Tensor{spec.elem_shape}] x "
                    f"{spec.length} — {tag}. Iterate at host time and call "
                    f"`offchip_load` per element (e.g. "
                    f"`[offchip_load({arg_name}[i], ...) for i in range(len({arg_name}))]`); "
                    f"do NOT pass the list itself to a DSL consumer."
                )
            else:
                assert isinstance(spec, _ListOfIntArg)
                lines.append(
                    f"  `{arg_name}`: list[int] x {spec.length} — {tag}. "
                    f"Convert to a tensor with `torch.tensor({arg_name})` "
                    f"before any DSL consumer (e.g. feed into `metadata_gen`)."
                )
        lines.extend([
            "",
            f"  Required output shapes (one per produced tensor): "
            f"`{list(contract.out_shapes)}`",
            f"  Number of outputs: {len(contract.out_shapes)} "
            f"({'tuple' if len(contract.out_shapes) > 1 else 'single tensor'})",
        ])

    # Child blackboxes — non-leaf only
    if children_signatures:
        lines.extend([
            "",
            "### Child Blackboxes Available",
            "",
            "The following child callables are pre-imported and available. Each",
            "implements the semantics of its corresponding PyTorch reference.",
            "Calling them is optional — these are tools available to you.",
            "Loops and conditionals around blackbox calls are permitted; calling a",
            "child zero, one, or multiple times is all fine. If you can produce a",
            "correct DSL implementation without invoking a particular child, that's",
            "acceptable.",
            "",
            "Each blackbox is invoked with the keyword arg ``out_shapes`` "
            "(tuple of per-output shapes). Every ``out_shapes`` entry must "
            "be a tile-stream shape with rank >= 3 (at least 1 stream dim "
            "+ 2 tile dims) — vanilla 2D shapes like ``(M, N)`` are not "
            "valid stream shapes. Single-output children still take "
            "1-tuples (e.g. ``out_shapes=((S, T_R, T_C),)``). Multi-output "
            "children must be destructured at the call site (e.g. "
            "``q, k, v = preprocess_heads(x, out_shapes=(s_q, s_k, s_v))``). "
            "The stub reshapes the reference's raw output to ``out_shape`` "
            "and returns. If you need a different stream layout (e.g. seq "
            "axis at the front), express the permutation explicitly on the "
            "stub's return via DSL ops (``bufferize`` + ``streamify`` with "
            "proper strides).",
            "",
            "KEY (MUST-READ): If you find that the input or output tensors to a blackbox child",
            "will be dynamic at runtime, you should not use the blackbox child.",
            "Instead, you should complete the full inlined implementation yourself."
        ])
        from src.node_signature import format_arg_spec
        for entry in children_signatures:
            child_name, arg_names, arg_specs, out_shapes, out_is_tuple = entry
            sig_args = ", ".join(arg_names)
            lines.append(f"  `{child_name}({sig_args}, *, out_shapes)`")
            for arg_name, spec in zip(arg_names, arg_specs):
                lines.append(f"    - `{arg_name}` {format_arg_spec(spec)}")
            if out_is_tuple:
                lines.append(
                    f"    - returns a tuple of {len(out_shapes)} tensors; "
                    f"per-output vanilla shapes: {list(out_shapes)}"
                )
            else:
                lines.append(
                    f"    - returns a single tensor (vanilla shape {out_shapes[0]})"
                )
        lines.extend([
            "",
            "**Call-site rule:** no tensor-method transform may appear between a tensor",
            "source and a blackbox call site — the stub recovers vanilla shape internally.",
            "Forbidden: `.reshape(...)`, `.permute(...)`, `.transpose(...)`, indexing,",
            "arithmetic, `.squeeze()`, `.unsqueeze()`, `.expand()`, `.flatten()`. If a",
            "stream's shape needs to change, use a DSL op (`reshape_stream`,",
            "`reshape_pad_stream`, `streamify`, `flatten`, etc.), never `tensor.reshape(...)`.",
        ])

    # Required function signature
    lines.extend([
        "",
        "### Required Function Signature",
        "",
        "```python",
        function_signature,
        "```",
        "",
        "Output a single Python function with this exact signature. "
        "No imports. No new torch tensors. "
        "Above the function definition, include a comment with your implementation reasoning.",
        "IMPORTANTLY, your implementation MUST match the numerical outputs of the reference pytorch code.",
        "It is NOT sufficient to simply match the required shape of the output tensors, although that is also required.",
    ])

    return "\n".join(lines)


def build_planner_system_prompt() -> str:
    """Load the planner agent's system prompt.

    The template is static — no source-file substitutions.
    """
    template_path = _PROMPTS_DIR / "planner_system.txt"
    assert template_path.exists(), f"Template not found: {template_path}"
    return template_path.read_text()


def build_planner_user_prompt(*, reference_code: str, dims: dict,
                               precompute_source: str | None = None,
                               hf_module_source: str | None = None) -> str:
    """Initial planner-call user prompt: reference + dims (+ precompute when known)."""
    dims_json = json.dumps(dims, indent=2)
    parts = [
        "## Reference code (the node you are deciding on)\n\n"
        "```python\n"
        f"{reference_code.rstrip()}\n"
        "```\n\n"
        "## Dimensions\n\n"
        "```json\n"
        f"{dims_json}\n"
        "```\n",
    ]
    if precompute_source is not None:
        parts.append(
            "\n## Precompute source (defines the `tensors` dict the reference receives)\n\n"
            "```python\n"
            f"{precompute_source.rstrip()}\n"
            "```\n"
        )
    if hf_module_source is not None:
        parts.append(
            "\n## Wrapped HF module source (the actual PyTorch graph "
            "that `self.model` executes — read-only context, not part of "
            "the reference)\n\n"
            "```python\n"
            f"{hf_module_source.rstrip()}\n"
            "```\n"
        )
    parts.append(
        "\nDecide whether to leaf this kernel or split it. "
        "Use the response format described in your system prompt."
    )
    return "".join(parts)


def build_replan_user_prompt(*, reference_code: str, dims: dict,
                              replan_iteration: int,
                              node_path: str,
                              failing_node: str,
                              last_turn_messages: list[str],
                              sibling_results: list[tuple[str, str]],
                              precompute_source: str | None = None,
                              hf_module_source: str | None = None) -> str:
    """Re-plan user prompt with failure context."""
    dims_json = json.dumps(dims, indent=2)
    last_turns = "\n".join(
        f"  Turn -{len(last_turn_messages) - i}: {msg}"
        for i, msg in enumerate(last_turn_messages)
    ) or "  (no per-turn messages captured)"

    if sibling_results:
        siblings_block = "\n".join(
            f"  {path}:\n```python\n{dsl.rstrip()}\n```"
            for path, dsl in sibling_results
        )
    else:
        siblings_block = "  (no successful siblings)"

    precompute_block = ""
    if precompute_source is not None:
        precompute_block = (
            "\n## Precompute source (defines the `tensors` dict the reference receives)\n\n"
            "```python\n"
            f"{precompute_source.rstrip()}\n"
            "```\n"
        )

    hf_block = ""
    if hf_module_source is not None:
        hf_block = (
            "\n## Wrapped HF module source (the actual PyTorch graph "
            "that `self.model` executes — read-only context, not part of "
            "the reference)\n\n"
            "```python\n"
            f"{hf_module_source.rstrip()}\n"
            "```\n"
        )

    return (
        "RE-PLAN CONTEXT\n"
        f"This is your re-plan #{replan_iteration} for subtree at {node_path}.\n"
        f"Refactor pass at node {failing_node} failed.\n"
        "Last turns' error / judge feedback:\n"
        f"{last_turns}\n"
        "Sibling nodes that succeeded (verified DSLs as reference):\n"
        f"{siblings_block}\n"
        "\n"
        "Common reasons your prior split failed:\n"
        "  - the failing piece is still too complex; split it further\n"
        "  - the failing piece has odd shape constraints; consider a different "
        "splitting axis\n"
        "  - the parent's compose may be cleaner with a different child boundary\n"
        "\n"
        "## Reference code (the subtree you are re-planning)\n\n"
        "```python\n"
        f"{reference_code.rstrip()}\n"
        "```\n\n"
        "## Dimensions\n\n"
        "```json\n"
        f"{dims_json}\n"
        "```\n"
        f"{precompute_block}"
        f"{hf_block}"
        "\nRe-plan this subtree. Use the response format from your system prompt."
    )