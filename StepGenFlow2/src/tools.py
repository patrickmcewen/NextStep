"""Tool wrappers for step_py functional simulation and timing model.

Exposes three @function_tool decorated functions for use by an LLM Writer
agent: execute_and_inspect, check_correctness, analyze_performance.
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

def _exec_build_graph(code: str, dims: dict):
    """Execute user code that defines build_graph() and return (graph, output_op).

    On error, re-raises with an enhanced message that maps <string> line numbers
    back to the user's code and includes actual values for failed assertions.
    """
    StepOps._counter = 0
    stripped = _strip_imports(code)
    full_code = IMPORT_SCAFFOLD + "\n" + stripped
    scaffold_lines = IMPORT_SCAFFOLD.count("\n") + 1  # +1 for the joining "\n"
    namespace = {}
    exec(full_code, namespace)
    assert "build_graph" in namespace, "Code must define a build_graph(dims) function"
    try:
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


def _exec_tiled_ref(code: str, dims: dict) -> torch.Tensor:
    """Execute user code that defines tiled_reference(dims) and return the output tensor."""
    namespace = {}
    exec(TILED_SCAFFOLD + "\n" + code, namespace)
    assert "tiled_reference" in namespace, "Code must define a tiled_reference(dims) function"
    result = namespace["tiled_reference"](dims)
    assert isinstance(result, torch.Tensor), f"tiled_reference must return a torch.Tensor, got {type(result)}"
    return result


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
