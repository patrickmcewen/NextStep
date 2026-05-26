"""Shared type wrappers for the STeP DSL variants.

`step_dsl.py` (general) and `step_dsl_max_tile.py` (tile-bounded) both define ops
that operate on the same value types: element-dtype tags, tile/dyn-tile/buffer
stream dtypes, the guarded shape view, and the `StepTensor` wrapper. Hosting
those definitions here — and importing them into each ops module — guarantees
a single class identity across DSL variants.

Why this matters: helpers like `tools._wrap_on_chip_call_args` and
`blackbox_stub.make_stub` import `StepTensor` statically. When the orchestrator
rebinds `sys.modules["step_dsl"]` between variants, statically-imported names
do *not* re-resolve, so producing a `StepTensor` in one helper and checking
`isinstance(x, StepTensor)` in another would silently see two different
classes. Centralizing the wrapper here keeps `step_dsl.StepTensor is
step_dsl_max_tile.StepTensor`.

Mirrors the IR's stream-dtype taxonomy (step_py/datatype.py) at the DSL
eager-runtime layer so that ops can gate on element type / tile kind the
same way ops.py does (e.g. retile_streamify requires Tile|DynTile,
streamify requires Buffer, flat_partition's control requires Select).

Definitions are kept local — not imported from step_py.datatype — because:
  * datatype.py pulls in sympy/DynDim; the DSL only needs type tags
  * the DSL tracks dynamic stream dims as a bool mask (DSL-specific), not
    as symbolic DynDim expressions — keeping the two namespaces separate
    avoids confusion about which kind of "dynamic" a piece of code means

Stream dynamism: each StepTensor carries `dyn_mask` (bool per stream dim)
and `dyn_origins` (producer provenance for each dyn slot). Newly synthesized
dynamic dims use fresh provenance tokens so eager DSL stream matching can
distinguish two equal-length runtime dims that would lower to different IR
DynDim symbols. Dynamic dims arise from flat_partition, flat_reassemble,
eager_merge (when any input has a dyn outer), flatmap_filter_row_streamify,
flatmap_counter, and flatten across a dynamic group. User-facing `x.shape[i]`
raises on a dynamic slot; DSL-internal code uses `x.underlying_tensor.shape`
as the escape hatch.
"""

import torch


class _ElemSingleton:
    """Base for element-type tags. One canonical instance per subclass."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        return type(self).__name__


class Float16(_ElemSingleton): pass
class Float32(_ElemSingleton): pass
class Uint32(_ElemSingleton): pass
class Uint64(_ElemSingleton): pass
class Bool(_ElemSingleton): pass  # used both as element type (Tile(Bool,...))
                                  # and as bare stream dtype (store-ack)


_TORCH_TO_ELEM = {
    torch.float32: Float32,
    torch.float16: Float16,
    torch.int32: Uint32,
    torch.int64: Uint64,
}


def _elem_from_torch(dtype):
    assert dtype in _TORCH_TO_ELEM, (
        f"_elem_from_torch: unsupported torch dtype {dtype}"
    )
    return _TORCH_TO_ELEM[dtype]()


class Tile:
    """Static tile: integer (r, c) shape, element dtype."""
    __slots__ = ("tile_dtype", "shape")

    def __init__(self, tile_dtype, shape):
        assert isinstance(tile_dtype, _ElemSingleton), \
            f"Tile.tile_dtype must be an element tag, got {tile_dtype!r}"
        assert len(shape) == 2 and all(isinstance(s, int) for s in shape), \
            f"Tile.shape must be (int, int), got {shape!r}"
        self.tile_dtype = tile_dtype
        self.shape = tuple(shape)

    def __eq__(self, other):
        return (isinstance(other, Tile)
                and self.tile_dtype == other.tile_dtype
                and self.shape == other.shape)

    def __hash__(self):
        return hash((Tile, self.tile_dtype, self.shape))

    def __repr__(self):
        return f"Tile({self.tile_dtype}, {self.shape})"


class DynTile:
    """Dynamic tile: at least one shape dim is a non-int (symbolic) marker."""
    __slots__ = ("tile_dtype", "shape")

    def __init__(self, tile_dtype, shape):
        assert isinstance(tile_dtype, _ElemSingleton)
        assert len(shape) == 2
        self.tile_dtype = tile_dtype
        self.shape = tuple(shape)

    def __repr__(self):
        return f"DynTile({self.tile_dtype}, {self.shape})"


class Buffer:
    """Buffer dtype: wraps a Tile/DynTile, carries the concrete buffer grid."""
    __slots__ = ("buff_dtype", "shape")

    def __init__(self, buff_dtype, shape):
        assert isinstance(buff_dtype, (Tile, DynTile)), \
            f"Buffer.buff_dtype must be Tile/DynTile, got {buff_dtype!r}"
        assert len(shape) >= 1, "Buffer.shape must have at least one dim"
        self.buff_dtype = buff_dtype
        self.shape = tuple(shape)

    def __repr__(self):
        return f"Buffer({self.buff_dtype}, {self.shape})"


class _SelectBase:
    """Base for Select-kind dtypes (MultiHot, Index)."""
    pass


class MultiHot(_SelectBase):
    __slots__ = ("total_n",)

    def __init__(self, n):
        assert isinstance(n, int) and n >= 1
        self.total_n = n

    def __repr__(self):
        return f"MultiHot({self.total_n})"


class Index(_SelectBase):
    """One-hot selector with total_n choices (called Index in IR)."""
    __slots__ = ("total_n",)

    def __init__(self, n):
        assert isinstance(n, int) and n >= 1
        self.total_n = n

    def __repr__(self):
        return f"Index({self.total_n})"


# ---------------------------------------------------------------------------
# Guarded shape view.
#
# Indexing into a stream dim that is known-dynamic raises. Tile dims (the
# last two) are always concrete and pass through. DSL-internal code that
# needs the raw shape can read `x.underlying_tensor.shape` directly — that's the
# documented escape hatch.
# ---------------------------------------------------------------------------


class _GuardedShape:
    __slots__ = ("_size", "_dyn_mask", "_origins", "_owner", "_elem_dims")

    def __init__(self, size, dyn_mask, origins, owner, elem_dims):
        assert len(dyn_mask) == len(size) - elem_dims, (
            f"_GuardedShape: dyn_mask len {len(dyn_mask)} must equal "
            f"ndim - elem_dims ({len(size)} - {elem_dims}) for size {tuple(size)}"
        )
        assert len(origins) == len(dyn_mask)
        self._size = tuple(int(s) for s in size)
        self._dyn_mask = tuple(dyn_mask)
        self._origins = tuple(origins)
        self._owner = owner
        self._elem_dims = elem_dims

    def __len__(self):
        return len(self._size)

    def __iter__(self):
        for i in range(len(self._size)):
            yield self[i]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return tuple(self[i] for i in range(*idx.indices(len(self._size))))
        i = idx + len(self._size) if idx < 0 else idx
        # Trailing elem_dims dims are element-area and always concrete.
        if i >= len(self._size) - self._elem_dims:
            return self._size[i]
        if self._dyn_mask[i]:
            origin = self._origins[i] or "<unknown>"
            raise AssertionError(
                f"{self._owner}.shape[{idx}] is a dynamic stream dim "
                f"(origin: {origin}). Reading its concrete size in DSL "
                f"eager code would hardcode a value the IR treats as "
                f"symbolic."
            )
        return self._size[i]

    def __eq__(self, other):
        if isinstance(other, _GuardedShape):
            other = other._size
        return self._size == tuple(other)

    def __hash__(self):
        return hash(self._size)

    def __repr__(self):
        stream_n = len(self._size) - self._elem_dims
        marks = [f"<dyn:{o or '?'}>" if d else str(s)
                 for s, d, o in zip(self._size[:stream_n],
                                     self._dyn_mask, self._origins)]
        marks.extend(str(s) for s in self._size[stream_n:])
        return "(" + ", ".join(marks) + ")"


# ---------------------------------------------------------------------------
# StepTensor: composition wrapper around torch.Tensor with stream dtype +
# per-stream-dim dynamism tracking.
# ---------------------------------------------------------------------------


def _elem_dims(stream_dtype):
    """How many trailing tensor dims are 'element area' (not stream dims).

    Tile/DynTile/Bool: 2 (rows × cols of the tile, or the 1×1 ack pair).
    Buffer:            2 + len(buffer.shape) (buffer grid + inner tile rc).
    Select:            1 (the choice dim 'n' lives in stream_dtype, not stream
                          shape — matches IR's Stream.rank for Select).
    """
    if isinstance(stream_dtype, Buffer):
        return 2 + len(stream_dtype.shape)
    if isinstance(stream_dtype, (Tile, DynTile, Bool)):
        return 2
    if isinstance(stream_dtype, _SelectBase):
        return 1
    raise AssertionError(f"_elem_dims: unsupported stream_dtype {stream_dtype!r}")


def _stream_rank(tensor, stream_dtype):
    """Number of stream dims a tensor has under a given stream_dtype."""
    return tensor.ndim - _elem_dims(stream_dtype)


class StepRawTensor:
    """Framework wrapper around raw off-chip input tensors.

    *** DO NOT CONSTRUCT MANUALLY. ***

    Every entry in the `tensors` dict that your `tiled_reference(dims, tensors)`
    receives is already a StepRawTensor — the framework wraps each raw input
    before exec'ing your code. To use an input you simply pass it through:

        x = offchip_load(tensors["x"], stride=..., out_shape_tiled=..., ...)
        sel = select_gen(tensors["expert_onehot"], ...)
        # static integer indexing into a stacked-tensor input is allowed:
        w_i = tensors["gate_weights"][i]   # still a StepRawTensor

    The wrapper exposes only what legitimate DSL code needs:
      - `.shape` (plain tuple — for tile-size derivations like
        `tile_row=tensors["W"][i].shape[0]`)
      - `.dtype`
      - `x[i]` / `x[i, j]` for static int indexing
      - `repr(x)`

    Arithmetic, `torch.*` dispatch, `.item()`, `.t()`, `.transpose`, iteration,
    and dynamic gathers (`x[idx_tensor]`) all raise — they would each be a
    way to execute compute outside the DSL on a raw input, which has no
    corresponding STeP IR node. Use DSL ops (`offchip_load`, `binary_*`,
    `unary_*`, `random_offchip_load`, `expert_addr_gen` + ref-load, etc.)
    instead.

    Internal access for DSL source ops is `.underlying_tensor` — the same
    escape hatch `StepTensor` uses. DSL code is statically prevented from
    typing that string by orchestrator.py's banned_patterns.
    """

    __slots__ = ("underlying_tensor",)

    def __init__(self, underlying_tensor):
        # The most common error here is a double-wrap from DSL code that
        # tried to "convert" `tensors["x"]` into a StepRawTensor — but
        # tensors[...] entries arrive pre-wrapped. Detect it specifically and
        # spell out the fix.
        assert not isinstance(underlying_tensor, StepRawTensor), (
            "StepRawTensor: refusing to wrap a value that is already a "
            "StepRawTensor. Inputs in the `tensors` dict are wrapped by the "
            "framework before your `tiled_reference` runs; you must NOT "
            "construct a StepRawTensor manually. Pass `tensors[\"...\"]` "
            "directly into the DSL source op:\n"
            "    BAD:  offchip_load(StepRawTensor(tensors[\"x\"]), ...)\n"
            "    GOOD: offchip_load(tensors[\"x\"], ...)"
        )
        assert isinstance(underlying_tensor, torch.Tensor), (
            f"StepRawTensor: expected torch.Tensor, got "
            f"{type(underlying_tensor).__name__}. Inputs in the `tensors` "
            f"dict are wrapped by the framework — never construct a "
            f"StepRawTensor manually; pass `tensors[\"...\"]` directly into "
            f"the DSL source op."
        )
        self.underlying_tensor = underlying_tensor

    @property
    def shape(self):
        return tuple(self.underlying_tensor.shape)

    @property
    def dtype(self):
        return self.underlying_tensor.dtype

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            assert all(isinstance(i, int) for i in idx), (
                f"StepRawTensor: tuple index must be all ints, got {idx!r}. "
                f"Dynamic gathers must go through random_offchip_load or "
                f"expert_addr_gen + linear_offchip_load_ref."
            )
        else:
            assert isinstance(idx, int), (
                f"StepRawTensor: only static int indexing allowed "
                f"(got {type(idx).__name__}). Dynamic gathers must go through "
                f"random_offchip_load or expert_addr_gen + linear_offchip_load_ref."
            )
        return StepRawTensor(self.underlying_tensor[idx])

    # Block torch.* dispatch (F.linear, F.sigmoid, torch.matmul, …) on this
    # type. NumPy-style "I'm not a torch.Tensor" — any registered torch
    # op called with a StepRawTensor argument raises TypeError.
    __torch_function__ = None

    def __iter__(self):
        # Python would otherwise fall back to __getitem__(0), __getitem__(1), …
        # which our __getitem__ happily accepts. Block iteration explicitly so
        # `for row in tensors["x"]:` fails loud.
        raise AssertionError(
            "StepRawTensor: iteration forbidden. Load with offchip_load and "
            "stream-process with DSL ops."
        )

    def __repr__(self):
        return (f"StepRawTensor(shape={tuple(self.underlying_tensor.shape)}, "
                f"dtype={self.underlying_tensor.dtype})")


class StepTensor:
    """torch.Tensor + stream_dtype + dyn-stream-dim mask.

    The wrapper is *composed*, not subclassed: real torch ops live on
    `.underlying_tensor`. DSL functions assert on `stream_dtype`, compute new metadata,
    and re-wrap on return. Iteration through `.shape` is guarded so callers
    cannot silently read a symbolic stream size.
    """

    __slots__ = ("underlying_tensor", "stream_dtype", "dyn_mask", "dyn_origins",
                 "offsets", "ragged_lengths")

    def __init__(self, underlying_tensor, stream_dtype, dyn_mask=None, dyn_origins=None,
                 offsets=None, ragged_lengths=None):
        assert isinstance(underlying_tensor, torch.Tensor), \
            f"StepTensor.underlying_tensor must be torch.Tensor, got {type(underlying_tensor).__name__}"
        sr = _stream_rank(underlying_tensor, stream_dtype)
        assert sr >= 0, (
            f"StepTensor: underlying_tensor.ndim={underlying_tensor.ndim} too small for "
            f"stream_dtype={stream_dtype!r}"
        )
        if dyn_mask is None:
            dyn_mask = (False,) * sr
        if dyn_origins is None:
            dyn_origins = (None,) * sr
        assert len(dyn_mask) == sr, (
            f"StepTensor: dyn_mask len {len(dyn_mask)} != stream_rank {sr} "
            f"(underlying_tensor.ndim={underlying_tensor.ndim}, stream_dtype={stream_dtype!r})"
        )
        assert len(dyn_origins) == sr
        self.underlying_tensor = underlying_tensor
        self.stream_dtype = stream_dtype
        self.dyn_mask = tuple(dyn_mask)
        self.dyn_origins = tuple(dyn_origins)
        self.offsets = offsets
        # `ragged_lengths` (when set) mirrors the functional sim's RaggedTensor:
        # a dict mapping a STREAM dim index (0 = outermost stream dim) to a
        # 1-D `lengths` tensor whose shape matches the leading stream dims
        # before that ragged dim. Set by `cache_read_addr_gen` /
        # `filter_last_tile`; consumed by `random_offchip_load` to mask padded
        # tiles to zero. Propagated through ops that preserve the ragged dim.
        self.ragged_lengths = dict(ragged_lengths) if ragged_lengths else None

    @property
    def shape(self):
        return _GuardedShape(self.underlying_tensor.shape, self.dyn_mask, self.dyn_origins,
                             "StepTensor", _elem_dims(self.stream_dtype))

    @property
    def ndim(self):
        return self.underlying_tensor.ndim

    @property
    def dtype(self):
        return self.underlying_tensor.dtype

    @property
    def stream_rank(self):
        return _stream_rank(self.underlying_tensor, self.stream_dtype)

    def __repr__(self):
        return (f"StepTensor(shape={tuple(self.underlying_tensor.shape)}, "
                f"stream_dtype={self.stream_dtype}, dyn_mask={self.dyn_mask})")
