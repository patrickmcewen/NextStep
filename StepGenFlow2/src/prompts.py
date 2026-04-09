"""Prompt construction for StepGenFlow Writer and Analyst agents.

Builds system and user prompts by filling templates with the executable spec,
import scaffold, constructor cheat sheet, and kernel-specific context.
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
# Constructor cheat sheet — compact reference for the Writer agent
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
  This is the standard way to give two operands matching stream shapes for BinaryMap/BinaryMapAccum.
  Example for 2D (R,C) with tiles (tr,tc): stride=(C//tc, 1), out_shape_tiled=(R//tr, C//tc)
  Example for GEMM — both loads use out_shape_tiled=(M//tm, N//tn, K//tk):
    A (M,K): stride=(K//tk, 0, 1)    — 0 on N dim means A is broadcast across N
    B (K,N): stride=(0, 1, N//tn)    — 0 on M dim means B is broadcast across M

LinearOffChipLoadRef(graph, ref, underlying: torch.Tensor, stride, out_shape_tiled,
    tile_row, tile_col, par_dispatch, transposed=False, trigger_rank=0)
  Triggered by ref stream. Output shape: ref.shape[:-trigger_rank] + out_shape_tiled

SelectGen(is_multihot: bool, tensor: torch.Tensor, n: int)
  SOURCE. Generates a selection stream from a pre-computed routing tensor.
  Output dtype: MultiHot(n) if is_multihot else Index(n)
  Output shape: (1,) + tensor.shape[:-1]
  Used to drive FlatPartition / FlatReassemble for expert routing.

MetadataGen(tensor: torch.Tensor)
  SOURCE. Streams out a tensor as scalar Uint64 tiles.
  Output shape: (1,) + tensor.shape, tile = (1,1)

### Compute Operators

UnaryMap(graph, input, fn: MapFn, write_back_mu: bool, compute_bw: int)
  Applies fn element-wise. Shape preserved.

BinaryMap(graph, in1, in2, fn: MapFn, write_back_mu: bool, compute_bw: int)
  Applies fn to pairs. Both inputs must have identical stream shapes.
  NOTE: This does NOT accumulate. For matmul with reduction, use BinaryMapAccum.

BinaryMapAccum(graph, in1, in2, fn: MapAccumFn, init_fn: InitFn, rank: int,
    write_back_mu: bool, compute_bw: int)
  Binary op + accumulation. rank = number of inner dims to reduce.
  e.g., for GEMM [M,N,K] with rank=1: accumulates over K, output shape is [M,N].

Accum(graph, input, output_stream_dtype: Tile, fn: AccumFn, init_fn: InitFn,
    accum_rank: int, write_back_mu: bool, compute_bw: int)
  Reduces last accum_rank dims of input stream.

### Sink Operators

OffChipStore(graph, input, par_dispatch: int, store_file_name: str = "output")

### Shape Operators

Broadcast(graph, input, num_consumers: int) -> access outputs as (broadcast_op, i)
Promote(graph, input, promote_rank: int) — inserts dim of 1
PromoteOuter(graph, input) — prepends dim of 1
Flatten(graph, input, min_rank, max_rank) — merges dim range
Reshape(graph, input, chunk_size, reshape_rank, write_back_mu, add_outer_dim=False, pad_fn=None)
  Splits dim at reshape_rank into (ceil(D/chunk_size), chunk_size).
  If reshape_rank==0 and not evenly divisible, pad_fn must be provided.
Bufferize(graph, input, rank: int) — collects tiles into Buffers
Streamify(graph, input, repeat_factor: List[int], rank: int) — streams out Buffers
DynStreamify(graph, input, ref, repeat_rank, bufferized_rank) — dynamic repeat via ref
ExpandRef(graph, input, ref, expand_rank) — broadcast 1-dims to match ref
RepeatStatic(graph, input, repeat_factor: int) — repeat each element N times
RepeatRef(graph, input, ref) — repeat each element to match ref inner dim
Parallelize(graph, input, parallelize_rank, num_consumers)
StaticReassemble(graph, inputs: List, merge_rank)

### Routing Operators (for MoE / conditional)

FlatPartition(graph, input, control, partition_rank: int,
    switch_cycles: List[int], write_back_mu: bool, num_consumers: int)
  Routes data to num_consumers output streams based on control (MultiHot/Index from SelectGen).
  Access outputs as (flat_partition_op, i). Output dim 0 is DynDim.

FlatReassemble(graph, inputs: List, control, reassemble_rank: int,
    switch_cycles: List[int], write_back_mu: bool)
  Inverse of FlatPartition — merges expert outputs back in original order.

EagerMerge(graph, inputs: List, input_rank: int)
  Merges inputs in arrival order. Two outputs:
    (eager_merge, 0) = merged data, (eager_merge, 1) = selection MultiHot

RetileStreamify(graph, input, split_row: bool, filter_mask: bool = False, chunk: int = 1)
  Splits tiles along rows/cols into smaller tiles and streams them.
  If filter_mask=True, only emits valid (non-padded) chunks.

### Data Types

Tile(tile_dtype=Float32(), shape=(r, c))
  Specifies the tile shape and dtype for output streams. Used as the `output_stream_dtype`
  argument to Accum and BinaryMapAccum to set the output tile dimensions.
  Example: Tile(tile_dtype=Float32(), shape=(tile_m, 1)) for a row-reduce producing scalars.

### Functions

map_fn: Add(), Mul(), Div(), Silu(), Exp(), Pow2(), Rsqrt(), Square(), RowWiseSum(),
         Matmul(weight_transposed=False), DynMatmul(weight_transposed=False),
         MulImmediate(constant), AddImmediate(constant), SubImmediate(constant),
         IsEqual(), MaskRow(tile), SetOffset(), RowWiseAppend(),
         SelectToScalar(), ToConstInt(constant)

map_accum_fn: Matmul(weight_transposed=False), DynMatmul(weight_transposed=False)

accum_fn: Add(), Mul(), RetileRow(), RetileCol(), SignalReqAllRead()

init_fn: Zero(shape=(r,c), dtype=Float32()), Empty(shape, dtype), DynEmpty(shape, dtype)

### Key Pattern: Always end with
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
# Public API
# ---------------------------------------------------------------------------

def build_decomposer_system_prompt() -> str:
    """Build the Decomposer agent's system prompt."""
    template_path = _PROMPTS_DIR / "decomposer_system.txt"
    assert template_path.exists(), f"Template not found: {template_path}"
    return template_path.read_text()


def build_decomposer_user_prompt(kernel_name: str, dims: dict, diagnosis: str = None) -> str:
    """Build the Decomposer agent's user prompt with PyTorch reference and dims."""
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
        "",
        "Rewrite this as a `tiled_reference(dims)` function that computes the same result using tiled tensors.",
        "I will automatically run your code and compare the output against the reference.",
    ]

    if diagnosis is not None:
        lines.extend([
            "",
            "### Diagnosis from Previous Attempt",
            "",
            diagnosis,
        ])

    return "\n".join(lines)


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
   ALL other ops (Compute, Shape, Sink, and LinearOffChipLoadRef) take graph as the FIRST
   positional arg and auto-register. Do NOT call graph.add_node() for these.
   WRONG: BinaryMapAccum(in1=a, in2=b, fn=...) — missing graph!
   RIGHT: BinaryMapAccum(graph, a, b, fn=...)

2. **ALL operands to BinaryMap/BinaryMapAccum MUST have identical stream shapes.**
   This is the #1 source of errors. When the tiled blueprint uses .unsqueeze(d).expand(...)
   to broadcast, the STeP equivalent is to include that dimension in out_shape_tiled with
   stride=0 in the LinearOffChipLoad. Do NOT use Promote/ExpandRef for this — bake
   broadcasting into the load.
   Example: if the blueprint tiles x to (B_tile, 1, tn, D) and expands to (B_tile, E, tn, D),
   then load x with out_shape_tiled=(B_tile, E) and stride=(1, 0).
   Both operands must end up with the EXACT same out_shape_tiled.

3. **map_accum_fn vs map_fn**: BinaryMapAccum uses map_accum_fn.Matmul(), NOT map_fn.Matmul().
   BinaryMap uses map_fn. These are different namespaces.

4. **LinearOffChipLoad stride computation.** The underlying tensor is viewed as a flat grid
   of tiles. stride tells the loader how many tiles to advance in the flat grid for each
   step in each stream dimension. The INNERMOST stream dim gets stride=1 (contiguous tiles),
   and outer dims get stride = product of inner tiled dimensions.
   Example: weight tensor (n_experts, D, F_dim), tile=(D, tile_f), F_tiles=F_dim//tile_f.
   Tile grid is (n_experts, F_tiles) in row-major. To load with out_shape_tiled=(B, E, F_tiles):
     stride=(0, F_tiles, 1) — B broadcasts (0), E strides by F_tiles, F_tiles strides by 1.
   NOT stride=(0, 1, F_tiles) — that would swap experts and F-tiles!"""

    return template.format(
        constructor_cheat_sheet=CONSTRUCTOR_CHEAT_SHEET,
        import_scaffold=IMPORT_SCAFFOLD,
        executable_spec=executable_spec,
        pitfalls=pitfalls,
    )


def build_writer_user_prompt(kernel_name: str, dims: dict, diagnosis: str = None,
                             tiled_code: str = None) -> str:
    """Build the Writer agent's user prompt with PyTorch reference, dims, and optional tiled blueprint."""
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

    if tiled_code is not None:
        lines.extend([
            "",
            "### Tiled PyTorch Blueprint (verified correct)",
            "",
            "The following tiled PyTorch code produces the correct output. Each operation",
            "maps to a STeP graph node. Use it as your guide for constructing the graph.",
            "",
            "```python",
            tiled_code.rstrip(),
            "```",
        ])

    lines.extend([
        "",
        "Write a `build_graph(dims)` function that implements this computation.",
        "I will automatically run your code through the emulator and correctness checker and show you the results.",
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
