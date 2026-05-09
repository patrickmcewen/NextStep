"""Auto-generated blackbox stub for a planner node.

During Pass 1 of a parent node's refactor, each child planner node is
exposed as a callable stub injected into the executor namespace. The
stub replicates the child's PyTorch reference semantics with a fixed
input/output reshape protocol.

Inputs are restricted to pure reshape — the parent must not permute,
slice, or otherwise non-bijectively transform a tensor before passing
it to a stub. The stub's input recovery is ``flatten().reshape(vanilla)``,
which is correct only for invertible reshape.

Outputs are template-driven and always plural: the parent calls the stub
with ``out_shapes=(<shape_0>, <shape_1>, ...)`` (and optionally
``out_perms=(<perm_0>, <perm_1>, ...)`` where each entry may be ``None``).
For each output the stub applies permute (if any) then reshape. If the
underlying reference returns a single Tensor (``len(out_shapes) == 1``)
the stub returns a Tensor; if it returns a tuple, the stub returns a
tuple of the same length.

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

    def stub(*tiled_args, out_shapes, out_perms=None):
        assert len(tiled_args) == len(arg_names), (
            f"stub expected {len(arg_names)} positional args "
            f"({arg_names}), got {len(tiled_args)}")
        assert isinstance(out_shapes, tuple) and len(out_shapes) >= 1 and all(
            isinstance(s, tuple) for s in out_shapes), (
            f"out_shapes must be a non-empty tuple of shape tuples, "
            f"got {out_shapes!r}")
        if out_perms is None:
            out_perms = (None,) * len(out_shapes)
        assert isinstance(out_perms, tuple) and len(out_perms) == len(out_shapes), (
            f"out_perms must be a tuple of length {len(out_shapes)} "
            f"(or None for all-None), got {out_perms!r}")

        vanilla_args = []
        for t, vshape, name in zip(tiled_args, vanilla_shapes, arg_names):
            assert isinstance(t, torch.Tensor), (
                f"stub arg {name!r} must be a Tensor, got {type(t).__name__}")
            vanilla_args.append(t.reshape(-1).reshape(vshape))

        with torch.no_grad():
            raw = ref_module(*vanilla_args)
        raw_outputs = raw if isinstance(raw, tuple) else (raw,)
        assert len(raw_outputs) == len(out_shapes), (
            f"output count mismatch: ref_module returned {len(raw_outputs)} "
            f"tensor(s), but parent requested {len(out_shapes)} via out_shapes")

        results = []
        for raw_out, out_shape, out_perm in zip(raw_outputs, out_shapes, out_perms):
            if out_perm is not None:
                raw_out = raw_out.permute(*out_perm)
            results.append(raw_out.reshape(out_shape))

        if recorder.contract is None:
            recorder.contract = Contract(
                arg_names=arg_names,
                vanilla_shapes=vanilla_shapes,
                tiled_shapes=tuple(tuple(t.shape) for t in tiled_args),
                tiled_values=tuple(t.detach().clone() for t in tiled_args),
                out_shapes=tuple(tuple(s) for s in out_shapes),
                out_perms=tuple(
                    None if p is None else tuple(p) for p in out_perms),
            )

        if isinstance(raw, tuple):
            return tuple(results)
        return results[0]

    stub.__name__ = "blackbox_stub"
    return stub
