"""Auto-generated blackbox stub for a planner node.

During Pass 1 of a parent node's refactor, each child planner node is
exposed as a callable stub injected into the executor namespace. The
stub replicates the child's PyTorch reference semantics with a fixed
input/output reshape protocol.

Tensor inputs are restricted to pure reshape — the parent must not
permute, slice, or otherwise non-bijectively transform a tensor before
passing it to a stub. The stub's input recovery is
``flatten().reshape(vanilla)``, which is correct only for invertible
reshape. List inputs (``list[Tensor]`` or ``list[int]``, classified by
``ArgSpec``) are passed through unchanged — they correspond to static
graph-construction-time iteration at the DSL/STeP target (e.g. per-
expert weight stacks, per-batch sequence-length metadata) and must
preserve their list-ness.

Outputs are template-driven and always plural: the parent calls the stub
with ``out_shapes=(<shape_0>, <shape_1>, ...)``. For each output the
stub reshapes the reference's raw tensor to ``out_shape`` and wraps the
result in a ``StepTensor``. The tile shape is the last two dims of
``out_shape`` (each ``out_shape`` must therefore be rank >= 2); the
remaining leading dims are stream dims. If the underlying reference
returns a single Tensor (``len(out_shapes) == 1``) the stub returns a
single ``StepTensor``; if it returns a tuple, the stub returns a tuple
of ``StepTensor``s of the same length. If the parent needs a layout
that pure reshape cannot reach (e.g. moving the seq axis to the front),
it must express that permutation explicitly via DSL ops
(``bufferize`` + ``streamify`` with proper strides) on the stub's
return; the stub itself does not permute.

The first call to a given stub records a ``Contract`` on the supplied
``ContractRecorder``; subsequent calls are pure passthrough. The
implementer is free to call a stub zero, one, or multiple times
(including inside loops or conditionals); when called multiple times,
the recorded contract reflects the first invocation, and the child's
Pass-1 verification runs against that first call site's shapes.
Implementers may also choose not to call a stub at all — if the
parent's DSL is correct without invoking the child, that's accepted.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.contract import Contract
from src.node_signature import (
    ArgSpec, IntArg, ListOfIntArg, ListOfTensorArg, TensorArg)
from src.step_dsl import StepTensor, StepRawTensor, Tile, _elem_from_torch


@dataclass
class ContractRecorder:
    contract: Contract | None = None


def _vanillify(name: str, value, spec: ArgSpec):
    """Recover the underlying-reference input from a tile-stream value.

    Tensor args are reshape-recovered via ``.reshape(-1).reshape(vanilla)``;
    int / list args pass through unchanged (host-side scalars and per-
    iteration host-side loads, not stream tensors).
    """
    if isinstance(spec, TensorArg):
        assert isinstance(value, torch.Tensor), (
            f"stub arg {name!r} was declared TensorArg but the parent "
            f"passed a {type(value).__name__}")
        return value.reshape(-1).reshape(spec.shape)
    if isinstance(spec, IntArg):
        assert isinstance(value, int) and not isinstance(value, bool), (
            f"stub arg {name!r} was declared IntArg but the parent passed "
            f"a {type(value).__name__}")
        return value
    if isinstance(spec, ListOfTensorArg):
        assert isinstance(value, list) and len(value) == spec.length, (
            f"stub arg {name!r} was declared ListOfTensorArg(length={spec.length}) "
            f"but the parent passed a {type(value).__name__} of length "
            f"{len(value) if hasattr(value, '__len__') else '?'}")
        for i, t in enumerate(value):
            assert isinstance(t, torch.Tensor), (
                f"stub arg {name!r}[{i}] is not a Tensor (got {type(t).__name__})")
        return value
    assert isinstance(spec, ListOfIntArg)
    assert isinstance(value, list) and len(value) == spec.length, (
        f"stub arg {name!r} was declared ListOfIntArg(length={spec.length}) "
        f"but the parent passed {type(value).__name__}")
    return value


def _tiled_shape_of(value, spec: ArgSpec) -> tuple[int, ...]:
    """Tile-stream shape for tensor args; ``()`` for int / list args."""
    if isinstance(spec, TensorArg):
        return tuple(value.shape)
    return ()


def _unwrap_steptensor(v):
    """Recursively unwrap StepTensor / StepRawTensor → torch.Tensor through lists.

    Top-level wrapper → its underlying ``.underlying_tensor``. List → element-
    wise unwrap (handles mixed ``list[StepTensor]``, ``list[StepRawTensor]``,
    ``list[Tensor]`` uniformly). Anything else (raw Tensor, int, list[int])
    passes through untouched.
    """
    if isinstance(v, (StepTensor, StepRawTensor)):
        return v.underlying_tensor
    if isinstance(v, list):
        return [_unwrap_steptensor(x) for x in v]
    return v


def _clone_value(value, spec: ArgSpec):
    """Detach-and-clone for tensor args; copy for int / list args."""
    if isinstance(spec, TensorArg):
        return value.detach().clone()
    if isinstance(spec, IntArg):
        return int(value)
    if isinstance(spec, ListOfTensorArg):
        return [t.detach().clone() for t in value]
    assert isinstance(spec, ListOfIntArg)
    return list(value)


def make_stub(*, ref_module: nn.Module,
              arg_names: tuple[str, ...],
              arg_specs: tuple[ArgSpec, ...],
              recorder: ContractRecorder,
              max_tile: int | None = None):
    assert len(arg_names) == len(arg_specs), (
        f"arg_names ({len(arg_names)}) and arg_specs "
        f"({len(arg_specs)}) length mismatch")
    assert max_tile is None or (isinstance(max_tile, int) and max_tile >= 1), (
        f"make_stub max_tile must be a positive int or None, got {max_tile!r}")

    def stub(*tiled_args, out_shapes):
        assert len(tiled_args) == len(arg_names), (
            f"stub expected {len(arg_names)} positional args "
            f"({arg_names}), got {len(tiled_args)}")
        assert isinstance(out_shapes, tuple) and len(out_shapes) >= 1 and all(
            isinstance(s, tuple) for s in out_shapes), (
            f"out_shapes must be a non-empty tuple of shape tuples, "
            f"got {out_shapes!r}")

        # Parent DSL ops may hand us StepTensor-wrapped inputs (chained stub
        # calls produce StepTensors; ``list(parallelize(...))`` produces a
        # list of StepTensors). Strip the wrapper recursively so ref_module
        # and the contract helpers (_vanillify / _clone_value /
        # _tiled_shape_of) always see raw torch.Tensors.
        unwrapped_args = tuple(_unwrap_steptensor(a) for a in tiled_args)

        vanilla_args = [
            _vanillify(name, value, spec)
            for name, value, spec in zip(arg_names, unwrapped_args, arg_specs)
        ]

        with torch.no_grad():
            raw = ref_module(*vanilla_args)
        raw_outputs = raw if isinstance(raw, tuple) else (raw,)
        assert len(raw_outputs) == len(out_shapes), (
            f"output count mismatch: ref_module returned {len(raw_outputs)} "
            f"tensor(s), but parent requested {len(out_shapes)} via out_shapes")

        results = [
            raw_out.reshape(out_shape)
            for raw_out, out_shape in zip(raw_outputs, out_shapes)
        ]

        if recorder.contract is None:
            recorder.contract = Contract(
                arg_names=arg_names,
                vanilla_shapes=tuple(
                    spec.shape if isinstance(spec, TensorArg) else ()
                    for spec in arg_specs),
                tiled_shapes=tuple(
                    _tiled_shape_of(v, spec)
                    for v, spec in zip(unwrapped_args, arg_specs)),
                tiled_values=tuple(
                    _clone_value(v, spec)
                    for v, spec in zip(unwrapped_args, arg_specs)),
                out_shapes=tuple(tuple(s) for s in out_shapes),
                tiled_outputs=tuple(r.detach().clone() for r in results),
                out_is_tuple=isinstance(raw, tuple),
                arg_specs=arg_specs,
                max_tile=max_tile,
            )

        # Wrap each output as a StepTensor so the parent's DSL ops can
        # chain on the return without re-attaching the wrapper. Tile shape
        # is the last two dims of out_shape; the remaining leading dims
        # are stream dims (rank may be 0).
        wrapped = []
        for raw_out, out_shape in zip(results, out_shapes):
            assert len(out_shape) >= 2, (
                f"stub output shape {out_shape} must be rank >= 2 to wrap "
                f"as a tile-stream StepTensor (last 2 dims are the tile)")
            tile_shape = (int(out_shape[-2]), int(out_shape[-1]))
            stream_dtype = Tile(_elem_from_torch(raw_out.dtype), tile_shape)
            wrapped.append(StepTensor(raw_out, stream_dtype=stream_dtype))

        if isinstance(raw, tuple):
            return tuple(wrapped)
        return wrapped[0]

    stub.__name__ = "blackbox_stub"
    return stub
