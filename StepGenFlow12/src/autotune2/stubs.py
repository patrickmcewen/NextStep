"""Variant stub generator for autotune2.

A variant stub wraps a child node's reference (``nn.Module``) so the
parent's DSL can call it with arguments in any ``TensorContract``
layout (reshape + permutation of the vanilla shape) and receive
outputs in any ``TensorContract`` layout. The adapter pair lives
*inside* the stub; the LLM only ever sees declarative contract
metadata.

Each child node accumulates its library entries in a single file
``<child_dir>/variants.py`` whose only top-level binding is

    variant_registry: dict[int, dict] = {
        <index>: {
            "input_contracts":  {arg_name:  TensorContract(...)},
            "output_contracts": {out_name:  TensorContract(...)},
        },
        ...
    }

The callable ``<child_name>_<index>`` is built at runtime by
``make_variant_stub`` from this registry plus the child's
``ref_module`` / ``arg_specs`` (which the autotuner driver already
holds). The pass-1 baseline is always variant 0 with identity
contracts on every on-chip arg and output.

(Divergence note from HANDOFF.md: the handoff sketched
``emit_adapter_pair`` / ``emit_wrapped_reference`` that would emit per-
variant Python *function bodies* into ``variants.py``. That shape was
abandoned for the declarative registry above. Reasons: (a) the wrapper
body is identical across variants — only contracts differ — so per-
variant source emission duplicates adapter logic into every run dir
and makes bug fixes touch N files; (b) the source body would have to
reach a live ``ref_module`` Python object, which forces awkward global-
state plumbing or a ``functools.partial``-style curry; (c) a registry
diffs cleanly and is just as readable for humans inspecting a run dir.
The adapter logic lives once in this module.)

Layout invariants
-----------------
The parent's DSL realizes a contract by emitting STeP ops
(``streamify`` / ``bufferize`` / ``reshape_stream`` / ...). The
resulting tile-stream tensor's flat row-major order matches the
contracted (post-reshape, post-permute) layout. So the variant stub
recovers vanilla by::

    contracted = tiled.reshape(-1).reshape(post_permute_shape)
    vanilla    = contracted.permute(inverse_perm).reshape(vanilla_shape)

and produces tile-stream outputs by the mirror image::

    contracted = vanilla_out.reshape(contract.reshape).permute(perm)
    tiled_out  = contracted.reshape(out_shape)

Identity contracts (``reshape == vanilla_shape`` + identity
permutation) flow through unchanged — both adapters degenerate to the
same pure-reshape behavior as ``blackbox_stub``.
"""

from __future__ import annotations

import importlib.util
from math import prod
from pathlib import Path

import torch
import torch.nn as nn

from src.autotune2.contracts import TensorContract
from src.node_signature import ArgSpec, ListOfIntArg, ListOfTensorArg, TensorArg
from src.step_dsl import StepTensor, StepRawTensor, Tile, _elem_from_torch


# ---------------------------------------------------------------------------
# Adapter primitives
# ---------------------------------------------------------------------------


def _inverse_permutation(perm: tuple[int, ...]) -> tuple[int, ...]:
    """``inv`` such that ``inv[perm[i]] == i``."""
    n = len(perm)
    inv = [0] * n
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


def invert_input_contract(
    tiled: torch.Tensor,
    vanilla_shape: tuple[int, ...],
    contract: TensorContract,
) -> torch.Tensor:
    """Tile-stream tensor in contract layout -> vanilla layout tensor."""
    assert tiled.numel() == prod(vanilla_shape), (
        f"invert_input_contract: tiled element count {tiled.numel()} != "
        f"vanilla element count {prod(vanilla_shape)} for vanilla_shape="
        f"{vanilla_shape!r}, contract={contract!r}"
    )
    contracted = tiled.reshape(-1).reshape(contract.post_permute_shape())
    inv = _inverse_permutation(contract.permutation)
    return contracted.permute(*inv).reshape(vanilla_shape)


def apply_output_contract(
    vanilla_out: torch.Tensor,
    contract: TensorContract,
    out_shape: tuple[int, ...],
) -> torch.Tensor:
    """Vanilla output -> contracted layout, reshaped to tile-stream out_shape."""
    assert vanilla_out.numel() == prod(out_shape), (
        f"apply_output_contract: vanilla output element count "
        f"{vanilla_out.numel()} != tile-stream out_shape element count "
        f"{prod(out_shape)}; vanilla_out.shape={tuple(vanilla_out.shape)!r}, "
        f"contract={contract!r}, out_shape={out_shape!r}"
    )
    contracted = vanilla_out.reshape(contract.reshape).permute(*contract.permutation)
    return contracted.reshape(out_shape)


# ---------------------------------------------------------------------------
# StepTensor unwrap helper (parity with blackbox_stub._unwrap_steptensor)
# ---------------------------------------------------------------------------


def _unwrap_steptensor(v):
    if isinstance(v, (StepTensor, StepRawTensor)):
        return v.underlying_tensor
    if isinstance(v, list):
        return [_unwrap_steptensor(x) for x in v]
    return v


# ---------------------------------------------------------------------------
# Variant stub factory
# ---------------------------------------------------------------------------


def make_variant_stub(
    *,
    ref_module: nn.Module,
    arg_names: tuple[str, ...],
    arg_specs: tuple[ArgSpec, ...],
    output_names: tuple[str, ...],
    input_contracts: dict[str, TensorContract],
    output_contracts: dict[str, TensorContract],
    variant_name: str,
):
    """Build a callable wrapping ``ref_module`` with the given contracts.

    The returned callable mirrors ``blackbox_stub``'s signature
    ``stub(*tiled_args, out_shapes=(...))`` but applies the input /
    output adapter pair derived from the contracts.

    ``input_contracts`` is keyed by tensor arg name. Entries must be
    absent for RAW tensor args and for list args. An absent contract
    for an on-chip TensorArg degenerates to identity.
    ``output_contracts`` is keyed by ``output_names``; absent entries
    degenerate to identity. ``output_names`` is conventional (the
    autotuner picks e.g. ``("out_0", "out_1", ...)``) and its length
    determines whether the wrapper returns a single ``StepTensor`` or
    a tuple, matching ``ref_module``'s native return form.
    """
    assert len(arg_names) == len(arg_specs), (
        f"make_variant_stub({variant_name!r}): arg_names ({len(arg_names)}) "
        f"and arg_specs ({len(arg_specs)}) length mismatch"
    )
    for name, spec in zip(arg_names, arg_specs):
        if not isinstance(spec, TensorArg):
            assert name not in input_contracts, (
                f"make_variant_stub({variant_name!r}): arg {name!r} is a "
                f"list arg ({type(spec).__name__}); list args cannot carry "
                f"a contract, got {input_contracts[name]!r}"
            )
    for k in input_contracts:
        assert k in arg_names, (
            f"make_variant_stub({variant_name!r}): input_contracts key "
            f"{k!r} is not in arg_names={arg_names!r}"
        )
    for k in output_contracts:
        assert k in output_names, (
            f"make_variant_stub({variant_name!r}): output_contracts key "
            f"{k!r} is not in output_names={output_names!r}"
        )

    def stub(*tiled_args, out_shapes):
        assert len(tiled_args) == len(arg_names), (
            f"variant stub {variant_name!r}: expected {len(arg_names)} "
            f"positional args ({arg_names}), got {len(tiled_args)}"
        )
        assert (
            isinstance(out_shapes, tuple)
            and len(out_shapes) == len(output_names)
            and all(isinstance(s, tuple) for s in out_shapes)
        ), (
            f"variant stub {variant_name!r}: out_shapes must be a tuple of "
            f"{len(output_names)} shape tuple(s) (one per output), got "
            f"{out_shapes!r}"
        )

        unwrapped = tuple(_unwrap_steptensor(a) for a in tiled_args)

        vanilla_args = []
        for name, value, spec in zip(arg_names, unwrapped, arg_specs):
            if isinstance(spec, TensorArg):
                assert isinstance(value, torch.Tensor), (
                    f"variant stub {variant_name!r}: arg {name!r} declared "
                    f"TensorArg but received {type(value).__name__}"
                )
                contract = input_contracts.get(name)
                if contract is None:
                    vanilla_args.append(value.reshape(-1).reshape(spec.shape))
                else:
                    vanilla_args.append(
                        invert_input_contract(value, spec.shape, contract)
                    )
            elif isinstance(spec, ListOfTensorArg):
                assert isinstance(value, list) and len(value) == spec.length, (
                    f"variant stub {variant_name!r}: arg {name!r} declared "
                    f"ListOfTensorArg(length={spec.length}), got "
                    f"{type(value).__name__} of length "
                    f"{len(value) if hasattr(value, '__len__') else '?'}"
                )
                vanilla_args.append(value)
            else:
                assert isinstance(spec, ListOfIntArg)
                assert isinstance(value, list) and len(value) == spec.length, (
                    f"variant stub {variant_name!r}: arg {name!r} declared "
                    f"ListOfIntArg(length={spec.length}), got "
                    f"{type(value).__name__}"
                )
                vanilla_args.append(value)

        with torch.no_grad():
            raw = ref_module(*vanilla_args)
        raw_outputs = raw if isinstance(raw, tuple) else (raw,)
        assert len(raw_outputs) == len(out_shapes), (
            f"variant stub {variant_name!r}: ref_module returned "
            f"{len(raw_outputs)} output(s) but parent requested "
            f"{len(out_shapes)} via out_shapes"
        )

        wrapped = []
        for name, raw_out, out_shape in zip(output_names, raw_outputs, out_shapes):
            assert len(out_shape) >= 2, (
                f"variant stub {variant_name!r}: output {name!r} requested "
                f"shape {out_shape!r} of rank {len(out_shape)}; tile-stream "
                f"outputs require rank >= 2 (last 2 dims are the tile)"
            )
            contract = output_contracts.get(name)
            if contract is None:
                tiled_out = raw_out.reshape(out_shape)
            else:
                tiled_out = apply_output_contract(raw_out, contract, out_shape)
            tile_shape = (int(out_shape[-2]), int(out_shape[-1]))
            stream_dtype = Tile(_elem_from_torch(tiled_out.dtype), tile_shape)
            wrapped.append(StepTensor(tiled_out, stream_dtype=stream_dtype))

        if isinstance(raw, tuple):
            return tuple(wrapped)
        return wrapped[0]

    stub.__name__ = variant_name
    return stub


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------


_REGISTRY_HEADER = '''\
"""Auto-generated variant registry for {child_name}.

Each entry maps integer variant index -> {{input_contracts,
output_contracts}}. The corresponding callable
``{child_name}_<index>`` is constructed at runtime by
``src.autotune2.stubs.make_variant_stub``. Do not edit by hand — this
file is overwritten by the autotuner each time a new variant is
admitted to {child_name}'s library.
"""

from src.autotune2.contracts import TensorContract


variant_registry = {{
'''


def _format_contract(c: TensorContract) -> str:
    return f"TensorContract(reshape={c.reshape!r}, permutation={c.permutation!r})"


def _format_contract_dict(d: dict[str, TensorContract]) -> str:
    if not d:
        return "{}"
    items = ", ".join(f"{k!r}: {_format_contract(v)}" for k, v in d.items())
    return "{" + items + "}"


def emit_variants_module(
    *,
    out_path: Path,
    child_name: str,
    variants: dict[int, dict],
) -> None:
    """Write ``variant_registry`` for ``child_name`` to ``out_path``.

    ``variants[index]`` is ``{"input_contracts": {arg: TensorContract},
    "output_contracts": {out: TensorContract}}``. The emitted file is
    deterministic in variant index order.
    """
    assert out_path.suffix == ".py", (
        f"emit_variants_module: out_path must be a .py file, got "
        f"{out_path!r}"
    )
    for idx, entry in variants.items():
        assert isinstance(idx, int) and idx >= 0, (
            f"emit_variants_module: variant index must be a non-negative int, "
            f"got {idx!r}"
        )
        assert set(entry.keys()) == {"input_contracts", "output_contracts"}, (
            f"emit_variants_module: variants[{idx}] must have keys "
            f"{{'input_contracts','output_contracts'}}, got {list(entry.keys())!r}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = [_REGISTRY_HEADER.format(child_name=child_name)]
    for idx in sorted(variants):
        entry = variants[idx]
        ic = _format_contract_dict(entry["input_contracts"])
        oc = _format_contract_dict(entry["output_contracts"])
        body.append(
            f"    {idx}: {{\n"
            f'        "input_contracts": {ic},\n'
            f'        "output_contracts": {oc},\n'
            f"    }},\n"
        )
    body.append("}\n")
    out_path.write_text("".join(body))


def load_variants_module(path: Path) -> dict[int, dict]:
    """Read a variants.py file and return its ``variant_registry`` dict."""
    assert path.exists(), f"load_variants_module: {path!r} does not exist"
    spec = importlib.util.spec_from_file_location("_autotune2_variants", str(path))
    assert spec is not None and spec.loader is not None, (
        f"load_variants_module: could not build import spec for {path!r}"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "variant_registry"), (
        f"load_variants_module: {path!r} has no top-level variant_registry"
    )
    reg = mod.variant_registry
    assert isinstance(reg, dict), (
        f"load_variants_module: variant_registry in {path!r} is "
        f"{type(reg).__name__}, expected dict"
    )
    return reg
