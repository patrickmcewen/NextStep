"""Per-call-site contract captured at the parent's blackbox call.

A ``Contract`` is what ties a parent's Pass-1 verification to its child's
Pass-1 problem statement: the child receives the actual tiled input
values that flowed into its stub call site, plus the parent-declared
output shapes.

Outputs are always represented as tuples (length 1 for single-output
forwards). The stub returns a Tensor when the underlying ref_module
returns a Tensor and returns a tuple when it returns a tuple — but the
parent-declared shapes are always plural for uniform plumbing.

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

Arg kinds
---------
Each positional input is classified by ``arg_specs`` (parallel to
``arg_names``). Three kinds are supported (see ``node_signature.py``):
``TensorArg`` (a single tensor), ``ListOfTensorArg`` (a static list of
identically-shaped tensors — per-expert weight stacks), and
``ListOfIntArg`` (a static list of ints — per-batch sequence lengths).
For tensor args, ``vanilla_shapes[i]`` and ``tiled_shapes[i]`` hold the
tensor's vanilla / tile-stream shape. For list args those entries are
``()`` — read ``arg_specs[i]`` to get the per-element shape and length.
``tiled_values[i]`` is correspondingly a ``Tensor`` for tensor args,
``list[Tensor]`` for ``ListOfTensorArg``, or ``list[int]`` for
``ListOfIntArg``.
"""

from dataclasses import dataclass
from typing import Any

import torch

from src.node_signature import ArgSpec, ListOfIntArg, ListOfTensorArg, TensorArg

# STeP streams = (stream_dims..., tile_row, tile_col). Minimum stream rank
# is 1 stream dim + 2 tile dims = 3.
_MIN_STREAM_RANK = 3


@dataclass(frozen=True)
class Contract:
    arg_names: tuple[str, ...]
    vanilla_shapes: tuple[tuple[int, ...], ...]
    tiled_shapes: tuple[tuple[int, ...], ...]
    tiled_values: tuple[Any, ...]
    out_shapes: tuple[tuple[int, ...], ...]
    arg_specs: tuple[ArgSpec, ...] = ()
    # Stub outputs at this call site, the reshape of each raw ref output to
    # the parent-declared ``out_shapes`` entry. Always stored as a tuple
    # parallel to ``out_shapes`` even when the underlying ref returns a single
    # Tensor; ``out_is_tuple`` records the original return form so the child's
    # Pass-1 gold can be reconstructed in the same shape the LLM-emitted
    # function will return.
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
    # Live ``--max-tile N`` bound, mirroring step_dsl_max_tile.MAX_TILE_ROW /
    # MAX_TILE_COL at the moment the parent's pass-1 captured this contract.
    # ``None`` = max-tile mode is off (stock step_dsl). When set, the last two
    # dims of every tile-stream shape recorded here must satisfy
    # ``<= max_tile``:
    #   - ``out_shapes[i]``: always a tile-stream shape (Contract enforces
    #     rank >= 3), so always checked.
    #   - ``tiled_shapes[i]``: only checked for on-chip tensor args. Raw
    #     forwards (``arg_is_raw[i] = True``) store the *vanilla* shape there
    #     — the child's ``offchip_load`` is what later picks a tile within
    #     bounds, so the parent's call-site shape legitimately exceeds the
    #     cap. When ``arg_is_raw`` is un-stamped (``()``), the tiled_shapes
    #     check is deferred — the orchestrator re-validates by re-running
    #     ``__post_init__`` via ``dataclasses.replace(c, arg_is_raw=...)``.
    max_tile: int | None = None

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
        # ``arg_specs`` defaults to an all-``TensorArg`` view derived from
        # ``vanilla_shapes`` so legacy callers (and pre-Stage-1.5 tests) keep
        # working without supplying it explicitly. Stub creation populates it
        # explicitly when a list-typed arg is present.
        if not self.arg_specs:
            object.__setattr__(self, "arg_specs", tuple(
                TensorArg(shape=s) for s in self.vanilla_shapes))
        assert len(self.arg_specs) == len(self.arg_names), (
            f"Contract.arg_specs length ({len(self.arg_specs)}) must equal "
            f"arg_names length ({len(self.arg_names)})")
        for i, spec in enumerate(self.arg_specs):
            if isinstance(spec, TensorArg):
                continue
            assert isinstance(spec, (ListOfTensorArg, ListOfIntArg)), (
                f"Contract.arg_specs[{i}] has unsupported type "
                f"{type(spec).__name__}")
            assert self.vanilla_shapes[i] == () and self.tiled_shapes[i] == (), (
                f"Contract.arg_specs[{i}] is a list arg ({spec!r}); the "
                f"corresponding vanilla_shapes/tiled_shapes entries must be "
                f"`()` (read arg_specs for shape info). Got "
                f"vanilla={self.vanilla_shapes[i]!r}, "
                f"tiled={self.tiled_shapes[i]!r}")
        if self.max_tile is not None:
            assert isinstance(self.max_tile, int) and self.max_tile >= 1, (
                f"Contract.max_tile must be a positive int or None, got "
                f"{self.max_tile!r}"
            )
            for i, shape in enumerate(self.out_shapes):
                # out_shapes is asserted rank >= _MIN_STREAM_RANK (= 3) above,
                # so shape[-2:] is always the tile (tile_row, tile_col).
                assert shape[-2] <= self.max_tile and shape[-1] <= self.max_tile, (
                    f"Contract.out_shapes[{i}]={shape}: tile dims "
                    f"{shape[-2:]} exceed max_tile={self.max_tile}. Under "
                    f"--max-tile mode the parent's stub call site must "
                    f"request an output whose last two dims are within bounds."
                )
            # On-chip tiled_shapes entries — only checked when arg_is_raw is
            # stamped (so we can tell apart on-chip tile streams from raw
            # vanilla forwards). Un-stamped contracts skip this check; the
            # orchestrator re-runs __post_init__ after stamping rawness.
            for i, raw in enumerate(self.arg_is_raw):
                if raw:
                    continue
                shape = self.tiled_shapes[i]
                if shape == () or len(shape) < 2:
                    continue
                assert shape[-2] <= self.max_tile and shape[-1] <= self.max_tile, (
                    f"Contract.tiled_shapes[{i}]={shape} (on-chip arg "
                    f"{self.arg_names[i]!r}): tile dims {shape[-2:]} exceed "
                    f"max_tile={self.max_tile}. Under --max-tile mode every "
                    f"on-chip value handed to a stub must have its last two "
                    f"dims within bounds; the parent's DSL ops should have "
                    f"produced a tile-stream that respects the cap."
                )
