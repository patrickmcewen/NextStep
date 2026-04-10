"""Tool wrappers for step_py functional simulation and timing model.

Exposes helpers for executing build_graph, tiled_reference, and hybrid_reference
code, plus @function_tool wrappers for the LLM agent.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import networkx
import sympy
import torch
from agents import function_tool

# ---------------------------------------------------------------------------
# Path setup — make step_py and validate_functional importable
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
# Import validate_functional module via importlib
# ---------------------------------------------------------------------------
_vf_spec = importlib.util.spec_from_file_location(
    "validate_functional", str(_STEPDB_DIR / "validate_functional.py"))
_validate_functional_mod = importlib.util.module_from_spec(_vf_spec)
_vf_spec.loader.exec_module(_validate_functional_mod)

IMPORT_SCAFFOLD = _validate_functional_mod.IMPORT_SCAFFOLD
_strip_imports = _validate_functional_mod._strip_imports

from step_py.ops import StepOps
from step_py.functional import execute, execute_values
from step_py.timing import analyze_timing


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _exec_build_graph(code: str, dims: dict, tensors: dict = None):
    """Execute user code that defines build_graph() and return (graph, output_op).

    If tensors is provided, calls build_graph(dims, tensors).
    Otherwise falls back to build_graph(dims) for backward compat.
    """
    StepOps._counter = 0
    stripped = _strip_imports(code)
    full_code = IMPORT_SCAFFOLD + "\n" + stripped
    scaffold_lines = IMPORT_SCAFFOLD.count("\n") + 1
    namespace = {}
    exec(full_code, namespace)
    assert "build_graph" in namespace, "Code must define a build_graph(dims) function"
    try:
        if tensors is not None:
            graph, output_op = namespace["build_graph"](dims, tensors)
        else:
            graph, output_op = namespace["build_graph"](dims)
    except Exception as exc:
        raise _enhance_error(exc, stripped, scaffold_lines) from exc
    return graph, output_op


def _enhance_error(exc: Exception, user_code: str, scaffold_lines: int) -> Exception:
    """Re-create an exception with better context for the LLM.

    - Maps <string> line numbers to user code lines
    - Shows the offending line and surrounding context
    - For AssertionError in ops.py, resolves locals to show actual shape values
    """
    import traceback as tb

    user_code_lines = user_code.split("\n")
    enhanced_parts = [f"{type(exc).__name__}: {exc}"]

    # Walk the raw traceback to get frame locals (extract_tb doesn't capture these)
    raw_frames = []
    tb_cursor = exc.__traceback__
    while tb_cursor is not None:
        raw_frames.append((tb_cursor.tb_frame, tb_cursor.tb_lineno))
        tb_cursor = tb_cursor.tb_next

    extracted = tb.extract_tb(exc.__traceback__)

    for i, frame_info in enumerate(extracted):
        if frame_info.filename == "<string>":
            user_line = frame_info.lineno - scaffold_lines
            if 1 <= user_line <= len(user_code_lines):
                enhanced_parts.append(f"\nError at line {user_line} of your code:")
                start = max(0, user_line - 3)
                end = min(len(user_code_lines), user_line + 2)
                for j in range(start, end):
                    marker = ">>>" if j == user_line - 1 else "   "
                    enhanced_parts.append(f"  {marker} {j + 1:4d} | {user_code_lines[j]}")

        elif "ops.py" in frame_info.filename or "datatype.py" in frame_info.filename:
            enhanced_parts.append(f"\nSTeP frontend error in {frame_info.filename.split('/')[-1]}:{frame_info.lineno}")
            if frame_info.line:
                enhanced_parts.append(f"  >>> {frame_info.line}")

            # For assertions in ops.py, show the local variables that matter
            if isinstance(exc, AssertionError) and i < len(raw_frames):
                frame_obj = raw_frames[i][0]
                locals_dict = frame_obj.f_locals
                # Show shape-related locals
                shape_vars = {}
                for name, val in locals_dict.items():
                    if name.startswith("_"):
                        continue
                    if "shape" in name or "stream" in name or "dtype" in name:
                        shape_vars[name] = _safe_repr(val)
                    elif isinstance(val, tuple) and all(isinstance(x, int) for x in val):
                        shape_vars[name] = val
                if shape_vars:
                    enhanced_parts.append("  Local variables:")
                    for name, val in shape_vars.items():
                        enhanced_parts.append(f"    {name} = {val}")

    return type(exc)("\n".join(enhanced_parts))


def _safe_repr(val) -> str:
    """Safe repr for error messages — handles streams, dtypes, shapes."""
    if hasattr(val, "shape") and hasattr(val, "stream_dtype"):
        return f"Stream(dtype={val.stream_dtype}, shape={val.shape})"
    if isinstance(val, tuple):
        return str(val)
    return repr(val)[:200]


def enhance_emulator_error(exc: Exception, user_code: str) -> str:
    """Enhance an error from execute_values/execute with node + code context.

    Walks the traceback to find:
    - Which node was being dispatched (from _dispatch frame locals)
    - Which line in functional.py failed and why
    - Maps the node back to a line in the user's code
    Returns a formatted error string for the LLM.
    """
    import re as _re
    import traceback as tb

    parts = [f"{type(exc).__name__}: {exc}"]

    # Walk raw frames to get locals
    tb_cursor = exc.__traceback__
    failing_node = None
    functional_frames = []

    while tb_cursor is not None:
        frame = tb_cursor.tb_frame
        filename = frame.f_code.co_filename
        lineno = tb_cursor.tb_lineno
        func_name = frame.f_code.co_name

        # Capture the node from _dispatch
        if func_name == "_dispatch" and "node" in frame.f_locals:
            failing_node = frame.f_locals["node"]

        # Capture frames in functional.py
        if "functional.py" in filename:
            source_line = ""
            # Try to read the source line
            try:
                import linecache
                source_line = linecache.getline(filename, lineno).strip()
            except Exception:
                pass
            functional_frames.append((func_name, lineno, source_line, frame.f_locals))

        tb_cursor = tb_cursor.tb_next

    # Report which node failed
    if failing_node is not None:
        nid = failing_node.instance_id
        op_type = type(failing_node).__name__
        stream_info = ""
        if hasattr(failing_node, "stream"):
            s = failing_node.stream
            stream_info = f", output_stream=Stream(dtype={s.stream_dtype}, shape={s.shape})"
        parts.append(f"\nFailing node: [{nid}] {op_type}{stream_info}")

        # Try to find where this node was created in user code
        # Search for the op type constructor call
        user_lines = user_code.split("\n")
        constructor_pattern = _re.compile(rf'\b{op_type}\s*\(')
        matches = [(i, line) for i, line in enumerate(user_lines) if constructor_pattern.search(line)]
        if matches:
            # Show the most likely match (if multiple, show all)
            parts.append(f"\nIn your code, [{nid}] {op_type} was likely created at:")
            for line_idx, line in matches:
                parts.append(f"  line {line_idx + 1}: {line.strip()}")

    # Report the functional.py error details
    if functional_frames:
        last = functional_frames[-1]
        func_name, lineno, source_line, locals_dict = last
        parts.append(f"\nEmulator error in functional.py:{lineno} ({func_name}):")
        if source_line:
            parts.append(f"  >>> {source_line}")

        # Show relevant locals from the failing frame
        shape_vars = {}
        for name, val in locals_dict.items():
            if name.startswith("_"):
                continue
            if hasattr(val, "shape") and hasattr(val, "numel"):
                shape_vars[name] = f"tensor shape={tuple(val.shape)}"
            elif "shape" in name or "stream" in name or "dtype" in name:
                shape_vars[name] = _safe_repr(val)
            elif isinstance(val, (int, float, bool, str)):
                shape_vars[name] = val
            elif isinstance(val, tuple) and len(val) < 10:
                shape_vars[name] = val
        if shape_vars:
            parts.append("  Local variables:")
            for name, val in shape_vars.items():
                parts.append(f"    {name} = {val}")

    return "\n".join(parts)


TILED_SCAFFOLD = "import torch\nimport torch.nn.functional as F\nimport math\n"

# DSL scaffold: basic imports + all step_dsl functions injected into namespace
_DSL_PY = Path(__file__).resolve().parent / "step_dsl.py"
DSL_SCAFFOLD = TILED_SCAFFOLD + "\n" + _DSL_PY.read_text() + "\n"

# Hybrid scaffold: all STeP imports + execute_values for inline emulator calls
HYBRID_SCAFFOLD = IMPORT_SCAFFOLD + "\nfrom step_py.functional import execute_values\nimport torch.nn.functional as F\n"


def _exec_tiled_ref(code: str, dims: dict, tensors: dict = None) -> torch.Tensor:
    """Execute user code that defines tiled_reference() and return the output tensor."""
    scaffold_lines = TILED_SCAFFOLD.count("\n") + 1
    namespace = {}
    exec(TILED_SCAFFOLD + "\n" + code, namespace)
    assert "tiled_reference" in namespace, "Code must define a tiled_reference function"
    try:
        if tensors is not None:
            result = namespace["tiled_reference"](dims, tensors)
        else:
            result = namespace["tiled_reference"](dims)
    except Exception as exc:
        raise _enhance_hybrid_error(exc, code, scaffold_lines) from exc
    assert isinstance(result, torch.Tensor), f"tiled_reference must return a torch.Tensor, got {type(result)}"
    return result


def _exec_dsl_ref(code: str, dims: dict, tensors: dict = None) -> torch.Tensor:
    """Execute user code with DSL functions available. Returns output tensor.

    The DSL scaffold injects all step_dsl functions (offchip_load, binary_matmul,
    etc.) into the execution namespace so the refactored code can call them directly.
    """
    scaffold_lines = DSL_SCAFFOLD.count("\n") + 1
    namespace = {}
    exec(DSL_SCAFFOLD + "\n" + code, namespace)
    assert "tiled_reference" in namespace, "Code must define a tiled_reference function"
    try:
        if tensors is not None:
            result = namespace["tiled_reference"](dims, tensors)
        else:
            result = namespace["tiled_reference"](dims)
    except Exception as exc:
        raise _enhance_hybrid_error(exc, code, scaffold_lines) from exc
    assert isinstance(result, torch.Tensor), f"tiled_reference must return a torch.Tensor, got {type(result)}"
    return result


def validate_tiling(code: str, dims: dict, tensors: dict) -> list[str]:
    """Validate tiled tensor structure by tracing execution.

    Works for both raw tiled PyTorch (tiler/canonicalize output) and DSL code
    (refactor output). Instruments key operations to verify that data tensors
    maintain tiled form (*stream_dims, tile_r, tile_c) throughout.

    Checks:
    1. torch.matmul inputs: at least one must be >= 3D (tiled)
    2. offchip_load (DSL): must have streaming (out_shape_tiled with dim > 1)
    3. binary_matmul (DSL): inputs must be >= 3D

    Returns list of violation strings. Empty = all shapes valid.
    """
    violations = []
    matmul_call_count = [0]  # mutable counter for closure

    # Determine scaffold
    has_dsl = "offchip_load" in code
    scaffold = DSL_SCAFFOLD if has_dsl else TILED_SCAFFOLD

    namespace = {}
    exec(scaffold + "\n" + code, namespace)

    # -- Instrument torch.matmul to check tiled form --
    original_matmul = torch.matmul

    def checking_matmul(a, b):
        matmul_call_count[0] += 1
        if a.ndim < 3 and b.ndim < 3:
            violations.append(
                f"torch.matmul #{matmul_call_count[0]}: both inputs are 2D "
                f"(a={tuple(a.shape)}, b={tuple(b.shape)}) — tiling has been lost. "
                f"At least one input must be >= 3D (*stream, tile_r, tile_c)."
            )
        return original_matmul(a, b)

    # Patch torch.matmul in the exec namespace
    namespace["torch"].matmul = checking_matmul

    # -- Instrument offchip_load for DSL code --
    if has_dsl and "offchip_load" in namespace:
        original_load = namespace["offchip_load"]

        def checking_load(underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False):
            result = original_load(underlying, stride, out_shape_tiled, tile_row, tile_col, transposed)
            R, C = underlying.shape[-2], underlying.shape[-1]
            total_tiles = (R // tile_row) * (C // tile_col)
            if total_tiles > 1 and all(s <= 1 for s in out_shape_tiled):
                violations.append(
                    f"offchip_load: underlying ({R}, {C}) → {total_tiles} tiles but "
                    f"out_shape_tiled={out_shape_tiled} — no streaming"
                )
            return result

        namespace["offchip_load"] = checking_load

    # -- Instrument binary_matmul for DSL code --
    if has_dsl and "binary_matmul" in namespace:
        original_bmatmul = namespace["binary_matmul"]

        def checking_bmatmul(a, b, weight_transposed=False):
            if a.ndim < 3 and b.ndim < 3:
                violations.append(
                    f"binary_matmul: both inputs are 2D "
                    f"(a={tuple(a.shape)}, b={tuple(b.shape)}) — tiling lost."
                )
            return original_bmatmul(a, b, weight_transposed)

        namespace["binary_matmul"] = checking_bmatmul

    # -- Runtime check: catch indexing of untiled (2D) data tensors --
    # Wrap Tensor.__getitem__: if a 2D tensor is indexed with another tensor
    # (gather pattern like x[idx]), that's a tiling violation.
    _original_getitem = torch.Tensor.__getitem__

    def _checking_getitem(self, key):
        if (self.ndim == 2
                and isinstance(key, torch.Tensor)
                and self.shape[0] > 1 and self.shape[1] > 1):
            violations.append(
                f"2D tensor {tuple(self.shape)} indexed with tensor "
                f"(gather on untiled data). Tile first, then use "
                f"flat_partition on the tiled data."
            )
        return _original_getitem(self, key)

    # -- Single instrumented execution with all checks active --
    func_name = "tiled_reference" if "tiled_reference" in namespace else "build_graph"
    assert func_name in namespace, f"Code must define {func_name}"

    torch.Tensor.__getitem__ = _checking_getitem
    try:
        if tensors is not None:
            namespace[func_name](dims, tensors)
        else:
            namespace[func_name](dims)
    except Exception:
        pass  # execution failed — correctness check handles it
    finally:
        torch.matmul = original_matmul
        torch.Tensor.__getitem__ = _original_getitem

    # -- Static check: tile dimensions from dims must be used in the code --
    tile_dim_keys = [k for k in dims if k.startswith("tile_")]
    for key in tile_dim_keys:
        if key not in code:
            violations.append(
                f'dims["{key}"]={dims[key]} is not used in the code — '
                f"tiling dimension was dropped."
            )

    return violations


def _strip_all_imports(code: str) -> str:
    """Aggressively strip ALL import/from lines — the scaffold provides everything."""
    lines = code.split("\n")
    result = []
    in_multiline = False
    for line in lines:
        s = line.strip()
        if in_multiline:
            if ")" in s:
                in_multiline = False
            continue
        if s.startswith(("import ", "from ")):
            if "(" in s and ")" not in s:
                in_multiline = True
            continue
        result.append(line)
    return "\n".join(result)


def _exec_hybrid_ref(code: str, dims: dict, tensors: dict = None) -> torch.Tensor:
    """Execute hybrid_reference or build_graph from user code, return output tensor.

    Accepts either function name. If build_graph is found, runs it through the
    emulator to get the output tensor. If hybrid_reference is found, calls it directly.
    """
    StepOps._counter = 0
    stripped = _strip_all_imports(code)
    full_code = HYBRID_SCAFFOLD + "\n" + stripped
    scaffold_lines = HYBRID_SCAFFOLD.count("\n") + 1
    namespace = {}
    exec(full_code, namespace)

    # build_graph is the primary function name; hybrid_reference accepted for compat
    if "build_graph" in namespace:
        try:
            if tensors is not None:
                graph, output_op = namespace["build_graph"](dims, tensors)
            else:
                graph, output_op = namespace["build_graph"](dims)
            from step_py.functional import execute as _execute
            result = _execute(graph, output_op)
        except Exception as exc:
            raise _enhance_hybrid_error(exc, stripped, scaffold_lines) from exc
        assert isinstance(result, torch.Tensor), f"build_graph emulator output must be a torch.Tensor, got {type(result)}"
        return result

    if "hybrid_reference" in namespace:
        # Legacy compat
        try:
            if tensors is not None:
                result = namespace["hybrid_reference"](dims, tensors)
            else:
                result = namespace["hybrid_reference"](dims)
        except Exception as exc:
            raise _enhance_hybrid_error(exc, stripped, scaffold_lines) from exc
        assert isinstance(result, torch.Tensor)
        return result

    assert False, "Code must define a build_graph(dims, tensors) function"


def _enhance_hybrid_error(exc: Exception, user_code: str, scaffold_lines: int) -> Exception:
    """Enhance errors from hybrid/tiled code with line context and tensor shapes.

    If the error originated inside the emulator (functional.py), uses
    enhance_emulator_error to identify the failing node and its inputs.
    Otherwise shows the failing user code line with tensor shapes.
    """
    import traceback as tb

    # Check if the error went through the emulator (functional.py)
    cursor = exc.__traceback__
    in_emulator = False
    while cursor is not None:
        if "functional.py" in cursor.tb_frame.f_code.co_filename:
            in_emulator = True
            break
        cursor = cursor.tb_next

    if in_emulator:
        enhanced = enhance_emulator_error(exc, user_code)
        return type(exc)(enhanced)

    # Error in user code itself — show line context and tensor shapes
    user_code_lines = user_code.split("\n")
    parts = [f"{type(exc).__name__}: {exc}"]

    extracted = tb.extract_tb(exc.__traceback__)
    raw_frames = []
    cursor = exc.__traceback__
    while cursor is not None:
        raw_frames.append((cursor.tb_frame, cursor.tb_lineno))
        cursor = cursor.tb_next

    for i, frame_info in enumerate(extracted):
        if frame_info.filename != "<string>":
            continue
        user_line = frame_info.lineno - scaffold_lines
        if not (1 <= user_line <= len(user_code_lines)):
            continue

        parts.append(f"\nFailing line of your code:")
        start = max(0, user_line - 3)
        end = min(len(user_code_lines), user_line + 2)
        for j in range(start, end):
            marker = ">>>" if j == user_line - 1 else "   "
            parts.append(f"  {marker} {user_code_lines[j]}")

        # Show shapes of tensor locals at the failing frame
        if i < len(raw_frames):
            frame_obj = raw_frames[i][0]
            tensor_shapes = {}
            for name, val in frame_obj.f_locals.items():
                if name.startswith("_"):
                    continue
                if hasattr(val, "shape") and hasattr(val, "numel"):
                    tensor_shapes[name] = f"shape={tuple(val.shape)}"
            if tensor_shapes:
                parts.append("  Tensor shapes at point of failure:")
                for name, shape_str in sorted(tensor_shapes.items()):
                    parts.append(f"    {name} = {shape_str}")

    return type(exc)("\n".join(parts))


def _format_node_values(values: dict, graph) -> str:
    """Format per-node tensor info for all nodes in topological order."""
    lines = []
    for node in networkx.topological_sort(graph):
        nid = node.instance_id
        op_type = node.__class__.__name__
        val = values.get(nid)

        if val is None:
            lines.append(f"[{nid}] {op_type}: None (sink node)")
            continue

        if isinstance(val, list):
            # Multi-output node
            parts = []
            for i, t in enumerate(val):
                parts.append(f"  output[{i}]: shape={tuple(t.shape)}")
                if t.is_floating_point():
                    parts.append(f"    min={t.min().item():.6f} max={t.max().item():.6f} mean={t.mean().item():.6f}")
            lines.append(f"[{nid}] {op_type}: {len(val)} outputs")
            lines.extend(parts)
            continue

        # Single tensor output
        shape_str = f"shape={tuple(val.shape)}"
        stats = ""
        if val.is_floating_point():
            stats = f" min={val.min().item():.6f} max={val.max().item():.6f} mean={val.mean().item():.6f}"
        lines.append(f"[{nid}] {op_type}: {shape_str}{stats}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# @function_tool wrappers
# ---------------------------------------------------------------------------

@function_tool
def execute_and_inspect(code: str, dims_json: str) -> str:
    """Execute a build_graph() function and inspect all intermediate tensor values."""
    try:
        dims = json.loads(dims_json)
        graph, _output_op = _exec_build_graph(code, dims)
        values = execute_values(graph)
        return _format_node_values(values, graph)
    except Exception:
        return traceback.format_exc()


@function_tool
def check_correctness(code: str, kernel_name: str, dims_json: str) -> str:
    """Check if a build_graph() produces output matching the PyTorch reference."""
    try:
        dims = json.loads(dims_json)
        graph, output_op = _exec_build_graph(code, dims)
        sim = execute(graph, output_op)

        config = _validate_functional_mod.load_config()
        gold = _validate_functional_mod.run_reference(kernel_name, dims, config)

        if gold.shape != sim.shape:
            return (
                f"SHAPE MISMATCH: gold {tuple(gold.shape)} vs sim {tuple(sim.shape)}\n"
                f"match=False"
            )

        max_err = (gold - sim).abs().max().item()
        rel_err = max_err / (gold.abs().max().item() + 1e-12)
        match = rel_err < 1e-5

        result = (
            f"match={match}\n"
            f"max_abs_err={max_err:.2e}\n"
            f"rel_err={rel_err:.2e}\n"
            f"output_shape={tuple(sim.shape)}"
        )

        if not match:
            diff = (gold - sim).abs()
            worst_idx = diff.argmax().item()
            worst_multi = torch.unravel_index(diff.argmax(), diff.shape)
            result += (
                f"\nworst_error_index={tuple(i.item() for i in worst_multi)}"
                f"\ngold_value={gold.flatten()[worst_idx].item():.6e}"
                f"\nsim_value={sim.flatten()[worst_idx].item():.6e}"
            )

        return result
    except Exception:
        return traceback.format_exc()


@function_tool
def analyze_performance(code: str, dims_json: str) -> str:
    """Run the analytical timing model on a build_graph() function."""
    try:
        dims = json.loads(dims_json)
        graph, _output_op = _exec_build_graph(code, dims)
        result = analyze_timing(graph)

        total = result["total_cycles"]
        sym_subs = result.get("sym_subs")

        # Handle symbolic expressions
        if hasattr(total, 'free_symbols') and total.free_symbols:
            subs = sym_subs if sym_subs else {s: 1 for s in total.free_symbols}
            total = total.xreplace(subs)

        total_val = int(sympy.N(total))

        lines = [f"total_cycles={total_val}"]
        lines.append("")
        lines.append("per_node breakdown:")

        bottleneck_id = None
        bottleneck_end = -1

        for nid, info in result["per_node"].items():
            node = info["node"]
            op_type = node.__class__.__name__

            st_val = int(sympy.N(info["st"]))
            end_val = int(sympy.N(info["end"]))
            oci_val = int(sympy.N(info["OCI"]))
            oti_val = float(sympy.N(info["OTI"]))

            lines.append(
                f"  [{nid}] {op_type}: st={st_val} end={end_val} OCI={oci_val} OTI={oti_val:.1f}"
            )

            if end_val > bottleneck_end:
                bottleneck_end = end_val
                bottleneck_id = nid

        assert bottleneck_id is not None, "Graph has no nodes"
        bn_node = result["per_node"][bottleneck_id]["node"]
        lines.append("")
        lines.append(
            f"bottleneck: [{bottleneck_id}] {bn_node.__class__.__name__} (end={bottleneck_end})"
        )

        return "\n".join(lines)
    except Exception:
        return traceback.format_exc()
