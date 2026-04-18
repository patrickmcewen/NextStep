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
# Helper: strip ops.py boilerplate for LLM consumption
# ---------------------------------------------------------------------------
def _strip_ops_boilerplate(source: str) -> str:
    """Reduce ops.py to its API surface: imports, class attrs, and __init__ sigs.

    Drops internal machinery (cost-model methods, stream/input accessors,
    __str__, replace_input, module-level helpers and constants) so the LLM
    sees only what matters for constructing STeP graph nodes.
    """
    tree = ast.parse(source)
    kept_module = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept_module.append(node)
            continue
        if not isinstance(node, ast.ClassDef):
            continue

        kept_cls = []
        for child in node.body:
            if isinstance(child, ast.AnnAssign):
                kept_cls.append(child)
            elif (isinstance(child, ast.Expr)
                  and isinstance(child.value, ast.Constant)
                  and isinstance(child.value.value, str)):
                kept_cls.append(child)  # class docstring
            elif isinstance(child, ast.FunctionDef) and child.name == "__init__":
                new_body = []
                if (child.body
                        and isinstance(child.body[0], ast.Expr)
                        and isinstance(child.body[0].value, ast.Constant)
                        and isinstance(child.body[0].value.value, str)):
                    new_body.append(child.body[0])  # preserve __init__ docstring
                new_body.append(ast.Expr(value=ast.Constant(value=...)))
                child.body = new_body
                kept_cls.append(child)

        if not kept_cls:
            kept_cls = [ast.Expr(value=ast.Constant(value=...))]
        node.body = kept_cls
        kept_module.append(node)

    tree.body = kept_module
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


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
    #{"name": "tiler", "template": "decomposer_system.txt"},
    #{"name": "router", "template": "pass_router_system.txt"},
    #{"name": "retiler", "template": "pass_retiler_system.txt"},
    #{"name": "canonicalize", "template": "canonicalize_system.txt"},
    #{"name": "refactor_load", "template": "refactor_load_system.txt", "executor": "dsl"},
    #{"name": "refactor_compute", "template": "refactor_compute_system.txt", "executor": "dsl"},
    #{"name": "refactor_shape", "template": "refactor_shape_system.txt", "executor": "dsl"},
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
    #"canonicalize": "canonicalize_judge_system.txt",
    #"refactor_load": "refactor_load_judge_system.txt",
    #"refactor_compute": "refactor_compute_judge_system.txt",
    #"refactor_shape": "refactor_shape_judge_system.txt",
    "refactor_final": "refactor_final_judge_system.txt",
    "translate": "translate_judge_system.txt",
    "translate_full": "translate_judge_system.txt",
}


def build_judge_system_prompt(pass_name: str) -> str:
    """Build a judge agent's system prompt from its template."""
    assert pass_name in _JUDGE_TEMPLATES, f"No judge template for pass '{pass_name}'"
    template_path = _PROMPTS_DIR / _JUDGE_TEMPLATES[pass_name]
    assert template_path.exists(), f"Judge template not found: {template_path}"
    return template_path.read_text()


def build_annotator_system_prompt() -> str:
    """Build the annotator agent's system prompt."""
    template_path = _PROMPTS_DIR / "annotate_system.txt"
    assert template_path.exists(), f"Annotator template not found: {template_path}"
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

        # Include DSL code as translation guide for translator passes
        if dsl_code is not None and is_translator:
            lines.extend([
                "### DSL Translation Guide (verified correct, each call = one STeP node)",
                "",
                "Each DSL function call below maps 1:1 to a STeP graph node. "
                "Use this as your blueprint for constructing the STeP graph:",
                "",
                "| DSL call | STeP node |",
                "|---|---|",
                "| `offchip_load(...)` | `LinearOffChipLoad(...)` |",
                "| `offchip_load_ref(ref, w, ...)` | `LinearOffChipLoadRef(graph, ref, w, ...)` |",
                "| `select_gen(...)` | `SelectGen(...)` |",
                "| `metadata_gen(tensor)` | `MetadataGen(tensor=tensor)` |",
                "| `binary_matmul(a, b)` | `BinaryMap(graph, a, b, map_fn.Matmul(), ...)` |",
                "| `binary_mul(a, b)` | `BinaryMap(graph, a, b, map_fn.Mul(), ...)` |",
                "| `binary_add(a, b)` | `BinaryMap(graph, a, b, map_fn.Add(), ...)` |",
                "| `binary_div(a, b)` | `BinaryMap(graph, a, b, map_fn.Div(), ...)` |",
                "| `unary_silu(x)` | `UnaryMap(graph, x, map_fn.Silu(), ...)` |",
                "| `unary_square(x)` | `UnaryMap(graph, x, map_fn.Square(), ...)` |",
                "| `unary_exp(x)` | `UnaryMap(graph, x, map_fn.Exp(), ...)` |",
                "| `unary_rsqrt(x)` | `UnaryMap(graph, x, map_fn.Rsqrt(), ...)` |",
                "| `unary_mul_imm(x, c)` | `UnaryMap(graph, x, map_fn.MulImmediate(c), ...)` |",
                "| `unary_add_imm(x, c)` | `UnaryMap(graph, x, map_fn.AddImmediate(c), ...)` |",
                "| `unary_sub_imm(x, c)` | `UnaryMap(graph, x, map_fn.SubImmediate(c), ...)` |",
                "| `unary_rowwise_sum(x)` | `UnaryMap(graph, x, map_fn.RowWiseSum(), ...)` |",
                "| `accum_add(x, rank)` | `Accum(graph, x, ..., accum_fn.Add(), ..., accum_rank=rank)` |",
                "| `accum_retile_row(x)` | `Accum(graph, x, ..., accum_fn.RetileRow(), ...)` |",
                "| `accum_retile_col(x)` | `Accum(graph, x, ..., accum_fn.RetileCol(), ...)` |",
                "| `promote(x, rank)` | `Promote(graph, x, promote_rank=rank)` |",
                "| `promote_outer(x)` | `PromoteOuter(graph, x)` |",
                "| `expand_ref(x, ref, expand_rank)` | `ExpandRef(graph, x, ref, expand_rank=...)` — static shapes only |",
                "| `repeat_ref(x, ref)` | `RepeatRef(graph, x, ref)` |",
                "| `repeat_static(x, factor)` | `RepeatStatic(graph, x, repeat_factor=factor)` |",
                "| `flatten(x, min_r, max_r)` | `Flatten(graph, x, min_rank=min_r, max_rank=max_r)` |",
                "| `reshape_stream(x, chunk, rank)` | `Reshape(graph, x, chunk_size=chunk, reshape_rank=rank, ...)` |",
                "| `retile_streamify(x, chunk, split_row)` | `RetileStreamify(graph, x, split_row=split_row, chunk=chunk)` |",
                "| `flat_partition(x, ctrl, n)` | `FlatPartition(graph, x, ctrl, ...)` |",
                "| `flat_reassemble(ins, ctrl)` | `FlatReassemble(graph, ins, ctrl, ...)` |",
                "| `offchip_store(x)` | `OffChipStore(graph, x, ...)` |",
                "",
                "```python",
                dsl_code.rstrip(),
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