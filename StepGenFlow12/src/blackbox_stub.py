"""Auto-generated blackbox stub for a planner node.

During Pass 1 of a parent node's refactor, each child planner node is
exposed as a callable stub injected into the executor namespace. The
stub replicates the child's PyTorch reference semantics with a fixed
input/output reshape protocol.

Inputs are restricted to pure reshape — the parent must not permute,
slice, or otherwise non-bijectively transform a tensor before passing
it to a stub. The stub's input recovery is ``flatten().reshape(vanilla)``,
which is correct only for invertible reshape.

Outputs are template-driven: the parent calls the stub with
``out_shape=...`` (and optionally ``out_perm=...``); the stub applies
permute (if any) then reshape, returning the parent-declared shape.

The first call to a given stub records a ``Contract`` on the supplied
``ContractRecorder``; subsequent calls are pure passthrough. This means
the contract reflects the parent's first invocation — appropriate
because the parent should call each stub exactly once per Pass-1 gate
run (single straight-line call).
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.contract import Contract


@dataclass
class ContractRecorder:
    contract: Contract | None = None


def make_stub(*, ref_module: nn.Module,
              arg_names: tuple[str, ...],
              vanilla_shapes: tuple[tuple[int, ...], ...],
              recorder: ContractRecorder):
    assert len(arg_names) == len(vanilla_shapes), (
        f"arg_names ({len(arg_names)}) and vanilla_shapes "
        f"({len(vanilla_shapes)}) length mismatch")

    def stub(*tiled_args, out_shape, out_perm=None):
        assert len(tiled_args) == len(arg_names), (
            f"stub expected {len(arg_names)} positional args "
            f"({arg_names}), got {len(tiled_args)}")
        vanilla_args = []
        for t, vshape, name in zip(tiled_args, vanilla_shapes, arg_names):
            assert isinstance(t, torch.Tensor), (
                f"stub arg {name!r} must be a Tensor, got {type(t).__name__}")
            v = t.reshape(-1).reshape(vshape)
            vanilla_args.append(v)
        with torch.no_grad():
            out = ref_module(*vanilla_args)
        if out_perm is not None:
            out = out.permute(*out_perm)
        out = out.reshape(out_shape)
        if recorder.contract is None:
            recorder.contract = Contract(
                arg_names=arg_names,
                vanilla_shapes=vanilla_shapes,
                tiled_shapes=tuple(tuple(t.shape) for t in tiled_args),
                tiled_values=tuple(t.detach().clone() for t in tiled_args),
                out_shape=tuple(out_shape),
                out_perm=None if out_perm is None else tuple(out_perm),
            )
        return out

    stub.__name__ = "blackbox_stub"
    return stub
