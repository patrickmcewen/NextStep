"""Per-call-site contract captured at the parent's blackbox call.

A ``Contract`` is what ties a parent's Pass-1 verification to its child's
Pass-1 problem statement: the child receives the actual tiled input
tensors that flowed into its stub call site, plus the parent-declared
output shapes and optional per-output permutations.

Outputs are always represented as tuples (length 1 for single-output
forwards). The stub returns a Tensor when the underlying ref_module
returns a Tensor and returns a tuple when it returns a tuple — but the
parent-declared shapes/perms are always plural for uniform plumbing.

Stream-shape invariant
----------------------
In STeP every stream tensor carries at least 1 stream dimension plus 2
tile dimensions, so any tile-stream tensor has rank >= 3. ``out_shapes``
describes the tiled output the parent requests back from the stub —
the value the child's LLM-emitted DSL function must produce — and is
therefore subject to that invariant. Construction asserts it loudly so
a malformed call site fails at the boundary rather than corrupting the
child's Pass-1 problem statement. ``vanilla_shapes`` describes the
underlying PyTorch reference's pre-tile shapes and is **not** subject
to the invariant.
"""

from dataclasses import dataclass

import torch

# STeP streams = (stream_dims..., tile_row, tile_col). Minimum stream rank
# is 1 stream dim + 2 tile dims = 3.
_MIN_STREAM_RANK = 3


@dataclass(frozen=True)
class Contract:
    arg_names: tuple[str, ...]
    vanilla_shapes: tuple[tuple[int, ...], ...]
    tiled_shapes: tuple[tuple[int, ...], ...]
    tiled_values: tuple[torch.Tensor, ...]
    out_shapes: tuple[tuple[int, ...], ...]
    out_perms: tuple[tuple[int, ...] | None, ...]
    # Stub outputs at this call site, post permute+reshape per ``out_perms`` /
    # ``out_shapes``. Always stored as a tuple parallel to ``out_shapes`` even
    # when the underlying ref returns a single Tensor; ``out_is_tuple`` records
    # the original return form so the child's Pass-1 gold can be reconstructed
    # in the same shape the LLM-emitted function will return.
    tiled_outputs: tuple[torch.Tensor, ...] = ()
    out_is_tuple: bool = False
    # Per-arg rawness flag, parallel to ``arg_names``. ``True`` = the parent's
    # call site fed in a value that had not yet passed through a DSL source
    # operator (raw ``tensors[...]`` read or a raw forwarded positional arg) —
    # the child must call ``offchip_load`` (or another producer) on it before
    # feeding it to a DSL consumer. ``False`` = the parent's call site fed in
    # an on-chip value (DSL producer/consumer/blackbox output, or an already-
    # on-chip intermediate arg). Empty tuple = un-stamped (legacy/test
    # construction); the orchestrator stamps this after the parent's pass-1
    # code is verified, by static AST inspection of the call site.
    arg_is_raw: tuple[bool, ...] = ()

    def __post_init__(self):
        for i, shape in enumerate(self.out_shapes):
            assert len(shape) >= _MIN_STREAM_RANK, (
                f"Contract.out_shapes[{i}] has rank {len(shape)} (shape={shape}); "
                f"STeP streams require rank >= {_MIN_STREAM_RANK} "
                f"(>= 1 stream dim + 2 tile dims). The parent's stub call site "
                f"declared an output that cannot be a tile stream — fix the "
                f"call site (the parent must request a tiled output shape, "
                f"not a vanilla one)."
            )
        assert len(self.arg_is_raw) in (0, len(self.arg_names)), (
            f"Contract.arg_is_raw length ({len(self.arg_is_raw)}) must be "
            f"either 0 (un-stamped) or {len(self.arg_names)} (one entry per "
            f"arg_name); got arg_is_raw={self.arg_is_raw!r}, "
            f"arg_names={self.arg_names!r}"
        )
