"""Per-call-site contract captured at the parent's blackbox call.

A ``Contract`` is what ties a parent's Pass-1 verification to its child's
Pass-1 problem statement: the child receives the actual tiled input
tensors that flowed into its stub call site, plus the parent-declared
output shape and optional output permutation.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Contract:
    arg_names: tuple[str, ...]
    vanilla_shapes: tuple[tuple[int, ...], ...]
    tiled_shapes: tuple[tuple[int, ...], ...]
    tiled_values: tuple[torch.Tensor, ...]
    out_shape: tuple[int, ...]
    out_perm: tuple[int, ...] | None
