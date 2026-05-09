"""Per-call-site contract captured at the parent's blackbox call.

A ``Contract`` is what ties a parent's Pass-1 verification to its child's
Pass-1 problem statement: the child receives the actual tiled input
tensors that flowed into its stub call site, plus the parent-declared
output shapes and optional per-output permutations.

Outputs are always represented as tuples (length 1 for single-output
forwards). The stub returns a Tensor when the underlying ref_module
returns a Tensor and returns a tuple when it returns a tuple — but the
parent-declared shapes/perms are always plural for uniform plumbing.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Contract:
    arg_names: tuple[str, ...]
    vanilla_shapes: tuple[tuple[int, ...], ...]
    tiled_shapes: tuple[tuple[int, ...], ...]
    tiled_values: tuple[torch.Tensor, ...]
    out_shapes: tuple[tuple[int, ...], ...]
    out_perms: tuple[tuple[int, ...] | None, ...]
