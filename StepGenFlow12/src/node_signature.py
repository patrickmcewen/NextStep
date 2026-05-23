"""Introspect a planner node's reference Module signature.

Given the ``reference_code`` string and the kernel's canonical input
dict, construct an instance of ``Model``, run ``forward(*canonical_args)``
once, and report the names + per-arg specs of intermediate inputs, the
per-output shapes (always plural — single-output forwards produce a
1-tuple), and the names of internal Parameters (weights).

Per-arg specs come in four flavors. Forwards may take any combination:
- ``TensorArg(shape)``: a single ``torch.Tensor`` (rank >= 1).
- ``IntArg()``: a Python ``int`` scalar (e.g. a head count, a tile bound).
  At the STeP target these are host-side constants — they flow through
  the stub unchanged and the child uses them in shape math / control
  flow, never as a stream tensor. 0-D ``torch.Tensor``s are rejected at
  signature time; use ``IntArg`` instead.
- ``ListOfTensorArg(length, elem_shape)``: a Python list of identically
  shaped tensors (e.g. per-expert weight stacks). At the STeP target
  these become per-element structural graph nodes — the list-ness is
  load-bearing metadata, not a stand-in for a stacked tensor.
- ``ListOfIntArg(length)``: a Python list of ints (e.g. per-batch
  sequence lengths). At the STeP target these are converted to a 1D
  int tensor and consumed via ``metadata_gen``/``cache_*_addr_gen``.
Any other arg type is rejected at signature time.
"""

import inspect
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TensorArg:
    shape: tuple[int, ...]


@dataclass(frozen=True)
class IntArg:
    pass


@dataclass(frozen=True)
class ListOfTensorArg:
    length: int
    elem_shape: tuple[int, ...]


@dataclass(frozen=True)
class ListOfIntArg:
    length: int


ArgSpec = Union[TensorArg, IntArg, ListOfTensorArg, ListOfIntArg]


def format_arg_spec(spec: ArgSpec) -> str:
    """Human-readable rendering used in Pass-1 prompts.

    ``TensorArg(shape=(M, N))`` → ``"vanilla shape (M, N)"``
    ``IntArg()`` → ``"Python int (host-side scalar)"``
    ``ListOfTensorArg(length=N, elem_shape=(D, F))`` → ``"list[Tensor(D, F)] x N"``
    ``ListOfIntArg(length=B)`` → ``"list[int] x B"``
    """
    if isinstance(spec, TensorArg):
        return f"vanilla shape {spec.shape}"
    if isinstance(spec, IntArg):
        return "Python int (host-side scalar)"
    if isinstance(spec, ListOfTensorArg):
        return f"list[Tensor{spec.elem_shape}] x {spec.length}"
    assert isinstance(spec, ListOfIntArg)
    return f"list[int] x {spec.length}"


def classify_arg(name: str, value) -> ArgSpec:
    """Map one forward arg to an ArgSpec; assertion-fail on unsupported types."""
    # Python int scalar (host-side knob: head count, tile bound, etc.). Checked
    # before ``torch.Tensor`` because ``bool`` is an ``int`` subclass and we
    # want both branches to reject it (no ``BoolArg`` spec exists).
    if isinstance(value, int) and not isinstance(value, bool):
        return IntArg()
    if isinstance(value, torch.Tensor):
        # 0-D scalar tensors fundamentally cannot be tile streams (last two
        # dims are the tile, by DSL definition). Reject loudly at signature
        # time so the planner LLM re-rolls the reference instead of crashing
        # deep in pass-1 wrap with ``tiled shape () must be rank >= 2``.
        # Hosts that want to pass a scalar should use a Python int + IntArg.
        assert value.dim() >= 1, (
            f"forward arg {name!r} is a 0-D torch.Tensor; the DSL has no "
            f"scalar-tensor stream type. Pass the value as a Python int "
            f"instead — it will be classified as IntArg and forwarded to the "
            f"child unchanged")
        return TensorArg(shape=tuple(value.shape))
    assert isinstance(value, list) and len(value) > 0, (
        f"forward arg {name!r} has unsupported type {type(value).__name__}; "
        f"only torch.Tensor (rank >= 1), Python int, non-empty list[Tensor], "
        f"and non-empty list[int] are accepted")
    if all(isinstance(x, torch.Tensor) for x in value):
        elem_shape = tuple(value[0].shape)
        for i, x in enumerate(value):
            assert tuple(x.shape) == elem_shape, (
                f"forward arg {name!r} is a list[Tensor] with mismatched "
                f"element shapes: index 0 has {elem_shape}, index {i} has "
                f"{tuple(x.shape)}; all elements must share the same shape")
        return ListOfTensorArg(length=len(value), elem_shape=elem_shape)
    if all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        return ListOfIntArg(length=len(value))
    types = sorted({type(x).__name__ for x in value})
    raise AssertionError(
        f"forward arg {name!r} is a list with mixed/unsupported element types "
        f"({types}); supported list element types are Tensor or int")


@dataclass(frozen=True)
class NodeSignature:
    arg_names: tuple[str, ...]
    arg_specs: tuple[ArgSpec, ...]
    out_shapes: tuple[tuple[int, ...], ...]
    weight_names: tuple[str, ...]
    out_is_tuple: bool

    @property
    def arg_shapes(self) -> tuple[tuple[int, ...], ...]:
        """Per-arg tensor shape, asserting all args are TensorArg.

        Many downstream consumers (blackbox stub creation, Pass-1 prompt
        rendering of child arg shapes) only handle plain tensors today.
        Calling this on a signature that contains list args is a loud
        failure, by design — once those consumers learn list support,
        they should read ``arg_specs`` directly instead.
        """
        shapes = []
        for name, spec in zip(self.arg_names, self.arg_specs):
            assert isinstance(spec, TensorArg), (
                f"arg_shapes is only valid when every arg is a Tensor; "
                f"arg {name!r} is {type(spec).__name__}. Read arg_specs "
                f"instead.")
            shapes.append(spec.shape)
        return tuple(shapes)


def extract_signature(reference_code: str, canonical_inputs: dict,
                      init_inputs: tuple | list = ()) -> NodeSignature:
    namespace: dict = {}
    exec(reference_code, namespace)
    assert "Model" in namespace, "reference_code must define a class Model(nn.Module)"
    model = namespace["Model"](*init_inputs)
    assert isinstance(model, nn.Module)

    forward_sig = inspect.signature(model.forward)
    arg_names = tuple(p for p in forward_sig.parameters if p != "self")

    for name in arg_names:
        assert name in canonical_inputs, (
            f"forward arg {name!r} has no entry in canonical_inputs "
            f"(available: {sorted(canonical_inputs)})")

    canonical_args = tuple(canonical_inputs[n] for n in arg_names)
    arg_specs = tuple(classify_arg(n, v) for n, v in zip(arg_names, canonical_args))

    with torch.no_grad():
        out = model(*canonical_args)

    if isinstance(out, torch.Tensor):
        out_shapes = (tuple(out.shape),)
        out_is_tuple = False
    else:
        assert isinstance(out, tuple) and all(isinstance(o, torch.Tensor) for o in out), (
            f"node forward must return a Tensor or a tuple of Tensors "
            f"(got {type(out).__name__})")
        out_shapes = tuple(tuple(o.shape) for o in out)
        out_is_tuple = True

    weight_names = tuple(n for n, _ in model.named_parameters())

    return NodeSignature(
        arg_names=arg_names,
        arg_specs=arg_specs,
        out_shapes=out_shapes,
        weight_names=weight_names,
        out_is_tuple=out_is_tuple,
    )
