"""Prompt construction for StepGenFlow5 — progressive translation pipeline.

Phase 1: PyTorch lowering passes (tiler, refactor_final)
Phase 2: STeP translation (single pass — DSL calls map 1:1 to STeP graph nodes)

Each phase produces verifiable intermediate code that matches gold.
"""
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
# Constructor cheat sheet — compact reference for the Writer/Translator agents
# ---------------------------------------------------------------------------
CONSTRUCTOR_CHEAT_SHEET = """\
### Source Operators (no graph arg — these are root nodes)

LinearOffChipLoad(underlying: torch.Tensor, stride: Tuple[int,...], out_shape_tiled: Tuple[int,...],
    tile_row: int, tile_col: int, par_dispatch: int, transposed: bool = False)
  Output stream shape: (1,) + out_shape_tiled
  stride and out_shape_tiled work together to define how the 2D underlying tensor maps
  to a multi-dimensional tile stream. len(stride) == len(out_shape_tiled).
  Each element of stride says "how many tile-columns to advance for one step in this dim."
  Use 0 in stride to BROADCAST a dimension (the tensor doesn't vary along that axis).
  Example for 2D (R,C) with tiles (tr,tc): stride=(C//tc, 1), out_shape_tiled=(R//tr, C//tc)
  Example for GEMM — both loads use out_shape_tiled=(M//tm, N//tn, K//tk):
    A (M,K): stride=(K//tk, 0, 1)    — 0 on N dim means A is broadcast across N
    B (K,N): stride=(0, 1, N//tn)    — 0 on M dim means B is broadcast across M

LinearOffChipLoadRef(graph, ref, underlying: torch.Tensor, stride: Tuple[int,...],
    out_shape_tiled: Tuple[int,...], tile_row: int, tile_col: int, par_dispatch: int,
    transposed: bool = False, trigger_rank: int = 0)
  Like LinearOffChipLoad but inherits outer stream dims from a `ref` node.
  Output stream shape: ref.stream.shape[:-trigger_rank] + out_shape_tiled
  IMPORTANT: Unlike LinearOffChipLoad, this takes `graph` as first arg and auto-registers.
  Use this when loading weights that must match a DYNAMIC stream shape (e.g., from FlatPartition).
  ExpandRef CANNOT bridge static→dynamic shapes, so when the DSL code does:
      w = offchip_load(weight, ...)       # static shape
      w_exp = expand_ref(w, dynamic_ref)  # expand to match dynamic partition
  The correct STeP translation is a SINGLE LinearOffChipLoadRef:
      w = LinearOffChipLoadRef(graph, dynamic_ref, weight, ...)  # inherits dynamic shape

SelectGen(is_multihot: bool, tensor: torch.Tensor, n: int)
  SOURCE. Generates a selection stream from a pre-computed routing tensor.
  Used to drive FlatPartition / FlatReassemble for routing.

MetadataGen(tensor: torch.Tensor)
  SOURCE. Streams out a tensor as scalar Uint64 tiles.

### Compute Operators

UnaryMap(graph, input, fn: MapFn, write_back_mu: bool, compute_bw: int)
BinaryMap(graph, in1, in2, fn: MapFn, write_back_mu: bool, compute_bw: int)
  Both inputs must have identical stream shapes.
BinaryMapAccum(graph, in1, in2, fn: MapAccumFn, init_fn: InitFn, rank: int,
    write_back_mu: bool, compute_bw: int)
  Binary op + accumulation. rank = number of inner dims to reduce.

Accum(graph, input, output_stream_dtype: Tile, fn: AccumFn, init_fn: InitFn,
    accum_rank: int, write_back_mu: bool, compute_bw: int)

### Sink Operators

OffChipStore(graph, input, par_dispatch: int, store_file_name: str = "output")

### Shape Operators

Broadcast(graph, input, num_consumers: int)
Promote(graph, input, promote_rank: int)
PromoteOuter(graph, input)
Flatten(graph, input, min_rank, max_rank)
Reshape(graph, input, chunk_size, reshape_rank, write_back_mu, add_outer_dim=False, pad_fn=None)
RepeatStatic(graph, input, repeat_factor: int)
RetileStreamify(graph, input, split_row: bool, filter_mask: bool = False, chunk: int = 1)
ExpandRef(graph, input, ref, expand_rank: int)
  Broadcast last `expand_rank` singleton stream dims of input to match ref's stream shape.
  Constraints: input's last expand_rank stream dims must all be 1, and
  input's leading stream dims must exactly match ref's leading stream dims.
  CANNOT bridge static→dynamic shapes — use LinearOffChipLoadRef instead.
RepeatRef(graph, input, ref)
  Broadcast input by appending ref's innermost stream dim.
  Constraint: input.stream.shape must equal ref.stream.shape[:-1].

### Routing Operators

FlatPartition(graph, input, control, partition_rank: int,
    switch_cycles: List[int], write_back_mu: bool, num_consumers: int)
  partition_rank: number of trailing input stream dims to KEEP in each output.
    Output stream per consumer: (DynDim,) + input.stream.shape[-partition_rank:]
    The DSL flat_partition always flattens the entire stream, so use partition_rank=0.
FlatReassemble(graph, inputs: List, control, reassemble_rank: int,
    switch_cycles: List[int], write_back_mu: bool)
  reassemble_rank: same concept as partition_rank — use 0 to match DSL flat_reassemble.

### Data Types

Tile(tile_dtype=Float32(), shape=(r, c))

### Functions

map_fn: Add(), Mul(), Div(), Silu(), Exp(), Pow2(), Rsqrt(), Square(), RowWiseSum(),
         Matmul(weight_transposed=False), MulImmediate(c), AddImmediate(c), SubImmediate(c)
map_accum_fn: Matmul(weight_transposed=False)
accum_fn: Add(), RetileRow(), RetileCol()
init_fn: Zero(shape=(r,c), dtype=Float32()), Empty(shape, dtype)

### CRITICAL: `graph` is the first positional arg

Source ops (LinearOffChipLoad, SelectGen, MetadataGen) do NOT take graph — call graph.add_node(op).
ALL other ops take `graph` as the FIRST positional arg:
  WRONG: Promote(input, promote_rank=2)
  RIGHT: Promote(graph, input, promote_rank=2)
  WRONG: BinaryMap(a, b, map_fn.Add(), False, 1024)
  RIGHT: BinaryMap(graph, a, b, map_fn.Add(), False, 1024)

BinaryMapAccum uses map_accum_fn.Matmul(), NOT map_fn.Matmul().

### Key Pattern: Always end build_graph with
  graph = infer_broadcast(graph)
  return graph, output_op"""


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
        replacements["ops_code"] = ops_path.read_text()
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
                "| `expand_ref(x, ref)` | `ExpandRef(graph, x, ref, expand_rank=...)` — static shapes only |",
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

    if is_translator:
        lines.extend([
            "",
            "### STeP Constructor Reference",
            "",
            CONSTRUCTOR_CHEAT_SHEET,
        ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writer prompt (fallback for non-pipeline mode)
# ---------------------------------------------------------------------------

def build_writer_system_prompt() -> str:
    """Build the Writer agent's system prompt with executable spec, imports, and cheat sheet."""
    template_path = _PROMPTS_DIR / "writer_system.txt"
    assert template_path.exists(), f"Template not found: {template_path}"

    template = template_path.read_text()

    assert _FUNCTIONAL_PY.exists(), f"Functional emulator not found: {_FUNCTIONAL_PY}"
    executable_spec = _FUNCTIONAL_PY.read_text()

    pitfalls = """\
1. **`graph` is the first positional arg for most operators.** Source ops (LinearOffChipLoad,
   SelectGen, MetadataGen) do NOT take graph — you must call graph.add_node(op) manually.
   ALL other ops take graph as FIRST positional arg and auto-register.
   WRONG: BinaryMapAccum(in1=a, in2=b, fn=...) — missing graph!
   RIGHT: BinaryMapAccum(graph, a, b, fn=...)

2. **ALL operands to BinaryMap/BinaryMapAccum MUST have identical stream shapes.**
   When the tiled blueprint uses .unsqueeze(d).expand(...), the STeP equivalent is
   stride=0 in LinearOffChipLoad. Do NOT use Promote/ExpandRef for this.

3. **map_accum_fn vs map_fn**: BinaryMapAccum uses map_accum_fn.Matmul(), NOT map_fn.Matmul().

4. **LinearOffChipLoad stride computation.** INNERMOST stream dim gets stride=1,
   outer dims get stride = product of inner tiled dimensions."""

    return template.format(
        constructor_cheat_sheet=CONSTRUCTOR_CHEAT_SHEET,
        import_scaffold=IMPORT_SCAFFOLD,
        executable_spec=executable_spec,
        pitfalls=pitfalls,
    )


def build_writer_user_prompt(kernel_name: str, dims: dict, diagnosis: str = None,
                             tiled_code: str = None, tensors: dict = None) -> str:
    """Build the Writer agent's user prompt."""
    config = _load_stepdb_config()
    assert kernel_name in config, f"Kernel '{kernel_name}' not found in bench_config.yaml"

    ref_path = _STEPDB_DIR / config[kernel_name]["problem"]
    assert ref_path.exists(), f"Reference file not found: {ref_path}"
    reference_code = ref_path.read_text()

    dims_json = json.dumps(dims, indent=2)

    lines = [
        f"## Kernel: {kernel_name}",
        "",
        "### PyTorch Reference",
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

    if tensors is not None:
        lines.extend([
            "",
            "### Pre-computed Tensors",
            "",
            "Your function signature is `build_graph(dims, tensors)`. The `tensors` dict contains:",
            "",
            "```",
            _format_tensors_description(tensors),
            "```",
            "",
            "**Do NOT call `torch.manual_seed`, `torch.randn`, `torch.rand`, `@`, `torch.topk`, or `torch.softmax`.**",
            "All tensors and routing metadata are pre-created. Access them via `tensors[\"key\"]`.",
        ])

    if tiled_code is not None:
        lines.extend([
            "",
            "### Tiled PyTorch Blueprint (verified correct)",
            "",
            "```python",
            tiled_code.rstrip(),
            "```",
        ])

    sig = "build_graph(dims, tensors)" if tensors is not None else "build_graph(dims)"
    lines.extend([
        "",
        f"Write a `{sig}` function that implements this computation.",
        "I will automatically run your code through the emulator and correctness checker.",
    ])

    if diagnosis is not None:
        lines.extend([
            "",
            "### Diagnosis from Previous Attempt",
            "",
            diagnosis,
        ])

    return "\n".join(lines)


def build_analyst_prompt(kernel_name: str, dims: dict, traces: list) -> str:
    """Build the Analyst agent's prompt with kernel info and failed traces."""
    template_path = _PROMPTS_DIR / "analyst_system.txt"
    assert template_path.exists(), f"Template not found: {template_path}"

    system = template_path.read_text()
    dims_json = json.dumps(dims, indent=2)

    lines = [
        system.rstrip(),
        "",
        "---",
        "",
        f"## Kernel: {kernel_name}",
        "",
        "### Dimensions",
        "",
        "```json",
        dims_json,
        "```",
    ]

    assert len(traces) > 0, "At least one trace must be provided"
    for i, trace in enumerate(traces):
        assert "code" in trace, f"Trace {i} missing 'code' key"
        assert "tool_outputs" in trace, f"Trace {i} missing 'tool_outputs' key"

        lines.extend([
            "",
            f"### Attempt {i + 1}",
            "",
            "#### Code",
            "",
            "```python",
            trace["code"].rstrip(),
            "```",
            "",
            "#### Tool Outputs",
        ])
        for j, output in enumerate(trace["tool_outputs"]):
            lines.extend([
                "",
                f"**Output {j + 1}:**",
                "```",
                output.rstrip(),
                "```",
            ])

    return "\n".join(lines)
