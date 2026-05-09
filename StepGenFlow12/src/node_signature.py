"""Introspect a planner node's reference Module signature.

Given the ``reference_code`` string and the kernel's canonical tensor
dict, construct an instance of ``Model``, run ``forward(*canonical_args)``
once, and report the names + shapes of intermediate inputs, the per-output
shapes (always plural — single-output forwards produce a 1-tuple), and
the names of internal Parameters (weights).
"""

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NodeSignature:
    arg_names: tuple[str, ...]
    arg_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]
    weight_names: tuple[str, ...]
    out_is_tuple: bool


def extract_signature(reference_code: str, canonical_inputs: dict) -> NodeSignature:
    namespace: dict = {}
    exec(reference_code, namespace)
    assert "Model" in namespace, "reference_code must define a class Model(nn.Module)"
    model = namespace["Model"]()
    assert isinstance(model, nn.Module)

    forward_sig = inspect.signature(model.forward)
    arg_names = tuple(p for p in forward_sig.parameters if p != "self")

    for name in arg_names:
        assert name in canonical_inputs, (
            f"forward arg {name!r} has no entry in canonical_inputs "
            f"(available: {sorted(canonical_inputs)})")

    canonical_args = tuple(canonical_inputs[n] for n in arg_names)
    arg_shapes = tuple(tuple(t.shape) for t in canonical_args)

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
        arg_shapes=arg_shapes,
        out_shapes=out_shapes,
        weight_names=weight_names,
        out_is_tuple=out_is_tuple,
    )
