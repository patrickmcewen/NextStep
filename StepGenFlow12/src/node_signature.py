"""Introspect a planner node's reference Module signature.

Given the ``reference_code`` string and the kernel's canonical tensor
dict, construct an instance of ``Model``, run ``forward(*canonical_args)``
once, and report the names + shapes of intermediate inputs, the names of
internal Parameters (weights), and the output shape.
"""

import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NodeSignature:
    arg_names: tuple[str, ...]
    arg_shapes: tuple[tuple[int, ...], ...]
    out_shape: tuple[int, ...]
    weight_names: tuple[str, ...]


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
    assert isinstance(out, torch.Tensor), (
        f"node forward must return a single Tensor in the v1 contract "
        f"(got {type(out).__name__})")
    out_shape = tuple(out.shape)

    weight_names = tuple(n for n, _ in model.named_parameters())

    return NodeSignature(
        arg_names=arg_names,
        arg_shapes=arg_shapes,
        out_shape=out_shape,
        weight_names=weight_names,
    )
