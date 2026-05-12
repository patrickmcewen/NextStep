"""Execution helpers for the orchestrator.

Two executors backed by ``exec`` of the LLM-emitted code:
    - ``_exec_dsl_ref``     runs ``tiled_reference(dims, tensors)`` against the
                            mounted DSL surface (standalone or bundle abstraction)
                            and returns the output tensor.
    - ``_exec_build_graph`` runs ``build_graph(dims, tensors)`` and returns the
                            ``(graph, output_op)`` pair; correctness then dispatches
                            it through the STeP simulator.

Both share the IMPORT_SCAFFOLD and the user-code line-mapping enhancement
machinery so that errors point back into the LLM's source.
"""
import importlib.util
import sys
from pathlib import Path

import torch

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

from step_py.ops import StepOps

from src.step_dsl import StepTensor, Tile, _elem_from_torch
from src.node_signature import TensorArg


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _exec_build_graph(code: str, dims: dict, tensors: dict):
    """Execute user code that defines build_graph() and return (graph, output_op)."""
    StepOps._counter = 0
    full_code = IMPORT_SCAFFOLD + "\n" + code
    scaffold_lines = IMPORT_SCAFFOLD.count("\n") + 1
    namespace = {}
    exec(full_code, namespace)
    assert "build_graph" in namespace, "Code must define a build_graph(dims, tensors) function"
    try:
        graph, output_op = namespace["build_graph"](dims, tensors)
    except Exception as exc:
        raise _enhance_error(exc, code, scaffold_lines) from exc
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


# DSL scaffold: torch + step_dsl source injected into the exec'd namespace so
# the LLM-emitted tiled_reference can call DSL ops without writing imports.
# In bundle mode the orchestrator registers the bundle's abstraction.py as
# ``sys.modules["step_dsl"]`` before this runs; we resolve the source from
# whichever module is currently bound to that name so the bundle's vocabulary
# (gated_mlp, linear, …) actually lands in the exec namespace, instead of the
# standalone ops (offchip_load, binary_matmul, …) the system prompt doesn't
# describe.
_DSL_IMPORTS = (
    "import torch\n"
    "import torch.nn.functional as F\n"
    "import math\n"
    "from step_dsl import *\n"
)


def _build_dsl_scaffold() -> str:
    # Wire the DSL surface via ``from step_dsl import *`` rather than
    # re-exec'ing the module source. The old approach created a parallel
    # ``StepTensor``/``Tile`` class set inside the user namespace, so values
    # produced by Python helpers that imported ``StepTensor`` from the
    # module directly (e.g. blackbox stubs) failed ``isinstance`` checks in
    # DSL ops running in the user namespace. With a single import the two
    # namespaces share class identity.
    #
    # Two run modes need this name to resolve:
    #   * Bundle mode — the orchestrator pre-registers the bundle's
    #     ``abstraction.py`` as ``sys.modules["step_dsl"]`` before any
    #     scaffold-using code runs.
    #   * Standalone mode — nothing else registers the name, so we point
    #     it at ``src.step_dsl`` here. Done lazily and only when unset so
    #     bundle's prior binding is never overwritten.
    if "step_dsl" not in sys.modules:
        from src import step_dsl as _src_step_dsl
        sys.modules["step_dsl"] = _src_step_dsl
    return _DSL_IMPORTS


def _wrap_on_chip_call_args(
    call_args: tuple,
    arg_specs: tuple,
    arg_is_raw: tuple[bool, ...],
    tiled_shapes: tuple[tuple[int, ...], ...],
) -> tuple:
    """Wrap on-chip ``TensorArg`` entries as ``StepTensor`` before a non-root
    DSL entry-point invocation.

    Pass-1 hands the child raw ``torch.Tensor``s for every positional input,
    while Pass-2 composition feeds it ``StepTensor``s from the parent's DSL
    ops. The contract block in the LLM prompt promises that on-chip args
    "may be passed directly to DSL consumers" — wrapping here makes that
    true in Pass-1 so the two passes share a calling convention and the
    LLM does not need an ``isinstance(x, StepTensor)`` guard at every leaf.

    RAW args stay raw (the LLM is told to load them with ``offchip_load``
    before any DSL consumer). List args also pass through unchanged — the
    contract block instructs the LLM to iterate at host time and load each
    element separately, so their elements are not wrapped here.
    """
    assert len(call_args) == len(arg_specs) == len(arg_is_raw) == len(tiled_shapes), (
        f"length mismatch: call_args={len(call_args)} arg_specs={len(arg_specs)} "
        f"arg_is_raw={len(arg_is_raw)} tiled_shapes={len(tiled_shapes)}"
    )
    wrapped = []
    for value, spec, raw, tshape in zip(
        call_args, arg_specs, arg_is_raw, tiled_shapes
    ):
        if raw or not isinstance(spec, TensorArg):
            wrapped.append(value)
            continue
        assert isinstance(value, torch.Tensor), (
            f"on-chip TensorArg must be a torch.Tensor, "
            f"got {type(value).__name__}")
        assert len(tshape) >= 2, (
            f"on-chip TensorArg tiled shape {tshape} must be rank >= 2 "
            f"(last two dims are the tile)")
        tile_shape = (int(tshape[-2]), int(tshape[-1]))
        stream_dtype = Tile(_elem_from_torch(value.dtype), tile_shape)
        wrapped.append(StepTensor(value, stream_dtype=stream_dtype))
    return tuple(wrapped)


def _exec_dsl_ref(code: str, dims: dict, tensors: dict, *,
                  extra_globals: dict | None = None,
                  entry_point: str = "tiled_reference",
                  call_args: tuple | None = None,
                  call_kwargs: dict | None = None):
    """Execute user code with DSL functions available. Returns the DSL output.

    The DSL scaffold injects the active step_dsl source — the standalone
    ``src/step_dsl.py`` in normal runs, or the bundle's ``abstraction.py``
    when bundle mode has registered it as ``sys.modules["step_dsl"]`` —
    into the execution namespace so the refactored code can call DSL ops
    directly without writing imports.

    extra_globals: Optional dict of names to inject into the execution namespace.
    Intended for Pass-1 blackbox stubs or Pass-2 verified child DSL functions.
    Injected before exec so user code can call these as if they were imports.

    entry_point/call_args/call_kwargs: select which top-level function to
    invoke after exec'ing ``code``. Defaults invoke ``tiled_reference(dims,
    tensors)`` — the root-node convention. For non-root planner nodes the
    LLM emits ``def <node_name>(<arg_1>, ..., *, out_shapes,
    out_perms=None)`` (positional names from ``Contract.arg_names``);
    callers pass ``entry_point=node_name``,
    ``call_args=parent_contract.tiled_values``, and ``call_kwargs={
    "out_shapes": ..., "out_perms": ...}`` to invoke that signature.

    Returns either a single ``torch.Tensor`` or a tuple/list of tensors.
    Intermediate planner nodes whose ``Model.forward`` returns a tuple
    (e.g. ``return Q, K, V``) produce tuple-returning DSL refs;
    ``_compare_against_gold`` walks tuples element-wise to compare against
    the matching tuple gold.

    Sandbox: snapshot ``builtins.isinstance`` and ``torch.Tensor`` before
    exec and restore them after, even if user code raises. LLM-emitted DSL
    refs have been observed monkey-patching these to fool framework checks
    (``builtins.isinstance = _patched_isinstance``, ``torch.Tensor = tuple``).
    Since ``builtins`` is process-global, patches accumulate across turns —
    each retry captures the already-patched callable as its "original" and
    chains another layer in front, eventually triggering RecursionError
    once the chain exceeds Python's recursion limit.
    """
    import builtins as _builtins
    if call_args is None:
        call_args = (dims, tensors)
    if call_kwargs is None:
        call_kwargs = {}
    scaffold = _build_dsl_scaffold()
    scaffold_lines = scaffold.count("\n") + 1
    namespace = {}
    if extra_globals:
        namespace.update(extra_globals)
    _saved_isinstance = _builtins.isinstance
    _saved_torch_tensor = torch.Tensor
    try:
        exec(scaffold + "\n" + code, namespace)
        assert entry_point in namespace, (
            f"Code must define a {entry_point} function"
        )
        try:
            result = namespace[entry_point](*call_args, **call_kwargs)
        except Exception as exc:
            raise _enhance_user_code_error(exc, code, scaffold_lines) from exc
    finally:
        _builtins.isinstance = _saved_isinstance
        torch.Tensor = _saved_torch_tensor
    # Unwrap StepTensor returns to raw torch.Tensor for downstream consumers
    # (gold comparison uses ``result.reshape(-1)``, which only works on raw
    # tensors). A non-root planner node naturally ends with a DSL op that
    # produces a StepTensor; requiring callers to append offchip_store just
    # to satisfy this boundary would be artificial — offchip_store is a sink.
    if isinstance(result, (tuple, list)):
        unwrapped = type(result)(
            t.underlying_tensor if isinstance(t, StepTensor) else t for t in result
        )
        for i, t in enumerate(unwrapped):
            assert isinstance(t, torch.Tensor), (
                f"{entry_point} returned a {type(result).__name__}; element "
                f"[{i}] must be a torch.Tensor or StepTensor, got "
                f"{type(result[i]).__name__}"
            )
        result = unwrapped
    else:
        if isinstance(result, StepTensor):
            result = result.underlying_tensor
        assert isinstance(result, torch.Tensor), (
            f"{entry_point} must return a torch.Tensor or StepTensor (or "
            f"tuple/list of those for tuple-returning planner nodes), "
            f"got {type(result).__name__}"
        )
    return result


def _enhance_user_code_error(exc: Exception, user_code: str, scaffold_lines: int) -> Exception:
    """Enhance an error raised by exec'd user code with line context and tensor shapes.

    Maps ``<string>`` traceback frames back to the user's code, quotes the failing
    line plus a few lines of context, and lists tensor-shape locals at that frame.
    Used by ``_exec_dsl_ref`` (the DSL executor); the build_graph executor uses
    ``_enhance_error`` which additionally inspects ops.py / datatype.py frames.
    """
    import traceback as tb

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
