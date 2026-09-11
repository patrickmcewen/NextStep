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

from src.step_dsl import StepTensor, StepRawTensor, Tile, _elem_from_torch
from src.node_signature import IntArg, TensorArg, ListOfTensorArg


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

    RAW tensor args are wrapped as ``StepRawTensor`` so the LLM-emitted child
    can only feed them into DSL source ops (``offchip_load`` etc.); arithmetic
    / torch.* / dynamic indexing on the raw input fail at the wrapper boundary
    instead of silently sneaking past the DSL.

    RAW ``ListOfTensorArg`` entries (e.g. a forwarded ``list[Tensor]`` of per-
    expert weights) get each element wrapped as ``StepRawTensor`` — the
    contract block instructs the LLM to do ``offchip_load(arg[i], ...)`` per
    element, so each ``arg[i]`` must satisfy ``_assert_raw``. On-chip
    ``ListOfTensorArg`` is rare (no DSL op produces a list of streams in
    practice today) and currently passes through unchanged. ``ListOfIntArg``
    is unchanged — the LLM converts it via ``torch.tensor(...)`` and feeds
    the result into ``metadata_gen``.
    """
    assert len(call_args) == len(arg_specs) == len(arg_is_raw) == len(tiled_shapes), (
        f"length mismatch: call_args={len(call_args)} arg_specs={len(arg_specs)} "
        f"arg_is_raw={len(arg_is_raw)} tiled_shapes={len(tiled_shapes)}"
    )
    wrapped = []
    for value, spec, raw, tshape in zip(
        call_args, arg_specs, arg_is_raw, tiled_shapes
    ):
        if isinstance(spec, IntArg):
            # Host-side scalar: parent fed the child a Python int (e.g. a head
            # count read out of ``dims``). It is neither raw nor on-chip — it
            # never flows through a DSL op — so we pass it through unchanged
            # and the child uses it directly in shape math / control flow.
            assert isinstance(value, int) and not isinstance(value, bool), (
                f"IntArg must be a Python int, got {type(value).__name__}")
            wrapped.append(value)
            continue
        if isinstance(spec, TensorArg):
            assert isinstance(value, torch.Tensor), (
                f"TensorArg must be a torch.Tensor, got {type(value).__name__}")
            if raw:
                wrapped.append(StepRawTensor(value))
                continue
            assert len(tshape) >= 2, (
                f"on-chip TensorArg tiled shape {tshape} must be rank >= 2 "
                f"(last two dims are the tile)")
            tile_shape = (int(tshape[-2]), int(tshape[-1]))
            stream_dtype = Tile(_elem_from_torch(value.dtype), tile_shape)
            wrapped.append(StepTensor(value, stream_dtype=stream_dtype))
            continue
        if isinstance(spec, ListOfTensorArg) and raw:
            # Match the contract-block instruction `offchip_load(arg[i], ...)`:
            # each list element must be a StepRawTensor so the source op's
            # _assert_raw is satisfied. The contract stores raw torch.Tensors
            # in tiled_values (the blackbox stub unwraps before recording), so
            # we wrap each element here.
            assert isinstance(value, list), (
                f"ListOfTensorArg must hand a list, got {type(value).__name__}")
            for i, elem in enumerate(value):
                assert isinstance(elem, torch.Tensor), (
                    f"ListOfTensorArg[{i}] must be a torch.Tensor, got "
                    f"{type(elem).__name__}")
            wrapped.append([StepRawTensor(elem) for elem in value])
            continue
        # ListOfTensorArg + on-chip, ListOfIntArg, and anything unknown:
        # pass through unchanged.
        wrapped.append(value)
    return tuple(wrapped)


def _wrap_input_tensors(tensors: dict) -> dict:
    """Wrap a ``tensors[...]`` input dict for LLM DSL exec.

    Every ``torch.Tensor`` value becomes a ``StepRawTensor`` so the LLM can
    only flow it into a DSL source op. 0-d scalar tensors auto-unwrap to
    Python floats/ints: ``unary_add_imm(x, tensors["eps"])`` lowers to
    ``AddImmediate(<scalar>)`` and expects a Python scalar at IR build time.
    Nested ``list[Tensor]`` (e.g. ``tensors["W_per_expert"] = [W0, W1, ...]``)
    is recursed element-wise so ``tensors["W_per_expert"][i]`` still yields a
    ``StepRawTensor``.

    Non-tensor, non-list values (ints, strings, etc.) pass through unchanged.
    """
    def _wrap(v):
        if isinstance(v, torch.Tensor):
            if v.dim() == 0:
                return v.item()
            return StepRawTensor(v)
        if isinstance(v, list):
            return [_wrap(x) for x in v]
        if isinstance(v, tuple):
            return tuple(_wrap(x) for x in v)
        return v
    return {k: _wrap(v) for k, v in tensors.items()}


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
    LLM emits ``def <node_name>(<arg_1>, ..., *, out_shapes)`` (positional
    names from ``Contract.arg_names``); callers pass
    ``entry_point=node_name``, ``call_args=parent_contract.tiled_values``,
    and ``call_kwargs={"out_shapes": ...}`` to invoke that signature.

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
        # Root entry: wrap each raw input tensor as StepRawTensor so the
        # LLM-emitted ``tiled_reference`` body can only feed `tensors[...]`
        # into a DSL source op. Bare ``tensors["x"] * 0.0`` /
        # ``torch.nn.functional.linear(tensors["x"], ...)`` / dynamic gathers
        # all fail at the wrapper boundary instead of silently sneaking past
        # the DSL. Non-root entry points get their args pre-wrapped by
        # ``_wrap_on_chip_call_args``, so we only wrap on the default path.
        call_args = (dims, _wrap_input_tensors(tensors))
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
    # Validate return type, then unwrap StepTensor → torch.Tensor for downstream
    # consumers (gold comparison uses ``result.reshape(-1)`` which only works
    # on raw tensors).
    #
    # Root (``tiled_reference``) ends with ``offchip_store`` — a sink that
    # returns a raw torch.Tensor — so raw returns are legal at the root.
    #
    # Non-root planner nodes hand their results to a parent for further DSL
    # chaining. The parent's pass1 stub wraps any return as a StepTensor (see
    # ``blackbox_stub.make_stub``), so pass2 composition assumes the real
    # child does the same. A raw torch.Tensor return silently passes the
    # correctness gate (values compare equal) but blows up at pass2 the first
    # time a parent DSL op meets a ``Tensor`` where a ``StepTensor`` is
    # required — e.g. ``random_offchip_store(k_cache, ...)`` followed by
    # ``return k_cache, v_cache`` would crash a downstream
    # ``accum_retile_row`` / ``.underlying_tensor`` access. Enforce here so
    # pass1 surfaces the type error against the offending child.
    is_root_entry = (entry_point == "tiled_reference")

    def _check_element(t, *, where: str) -> torch.Tensor:
        if isinstance(t, StepTensor):
            return t.underlying_tensor
        if is_root_entry and isinstance(t, torch.Tensor):
            return t
        if is_root_entry:
            raise AssertionError(
                f"{entry_point} must return a torch.Tensor or StepTensor "
                f"(or tuple/list of those for tuple-returning planner nodes), "
                f"got {type(t).__name__} at {where}"
            )
        raise AssertionError(
            f"{entry_point} (non-root planner node) must return a "
            f"StepTensor (or tuple/list of StepTensors) at {where}; "
            f"got {type(t).__name__}. The parent's pass1 stub wraps "
            f"non-root returns as StepTensor, so pass2 composition will "
            f"fail when a parent DSL op meets a raw torch.Tensor. If the "
            f"node wrote into a raw input via random_offchip_store, "
            f"re-load the updated buffer (e.g. offchip_load) and return "
            f"the on-chip stream rather than the raw input."
        )

    if isinstance(result, (tuple, list)):
        result = type(result)(
            _check_element(t, where=f"element [{i}]")
            for i, t in enumerate(result)
        )
    else:
        result = _check_element(result, where="single return")
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
