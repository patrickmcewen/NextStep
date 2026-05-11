"""STeP DSL.

Each DSL function whose lowered STeP node carries a perf knob accepts that
knob as a keyword-only argument with default 1:

  - compute DSL calls (binary_*, unary_*, accum_*, binary_map_accum) accept
    ``compute_bw=N``.
  - off-chip DSL calls (offchip_load*, dyn_offchip_load, random_offchip_*,
    offchip_store) accept ``par_dispatch=N``.

The kwarg is asserted (>= 1) but otherwise inert at eager exec time —
the deterministic translator (dsl_to_step.py) reads it back from the AST
and forwards it to the STeP node constructor.
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Typed stream wrapper.
#
# Mirrors the IR's stream-dtype taxonomy (step_py/datatype.py) at the DSL
# eager-runtime layer so that ops can gate on element type / tile kind the
# same way ops.py does (e.g. retile_streamify requires Tile|DynTile,
# streamify requires Buffer, flat_partition's control requires Select).
#
# Definitions are kept local — not imported from step_py.datatype — because:
#  * datatype.py pulls in sympy/DynDim; the DSL only needs type tags
#  * the DSL tracks dynamic stream dims as a bool mask (DSL-specific), not
#    as symbolic DynDim expressions — keeping the two namespaces separate
#    avoids confusion about which kind of "dynamic" a piece of code means
#
# Stream dynamism: each StepTensor carries `dyn_mask` (bool per stream dim)
# and `dyn_origins` (op name that birthed each dyn slot). Dynamic dims arise
# from flat_partition, flat_reassemble, eager_merge (when any input has a
# dyn outer), flatmap_filter_row_streamify, flatmap_counter, and flatten
# across a dynamic group. User-facing `x.shape[i]` raises on a dynamic slot;
# DSL-internal code uses `x.tensor.shape` as the escape hatch.
# ---------------------------------------------------------------------------


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
# needs the raw shape can read `x.tensor.shape` directly — that's the
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


class StepTensor:
    """torch.Tensor + stream_dtype + dyn-stream-dim mask.

    The wrapper is *composed*, not subclassed: real torch ops live on
    `.tensor`. DSL functions assert on `stream_dtype`, compute new metadata,
    and re-wrap on return. Iteration through `.shape` is guarded so callers
    cannot silently read a symbolic stream size.
    """

    __slots__ = ("tensor", "stream_dtype", "dyn_mask", "dyn_origins", "offsets")

    def __init__(self, tensor, stream_dtype, dyn_mask=None, dyn_origins=None,
                 offsets=None):
        assert isinstance(tensor, torch.Tensor), \
            f"StepTensor.tensor must be torch.Tensor, got {type(tensor).__name__}"
        sr = _stream_rank(tensor, stream_dtype)
        assert sr >= 0, (
            f"StepTensor: tensor.ndim={tensor.ndim} too small for "
            f"stream_dtype={stream_dtype!r}"
        )
        if dyn_mask is None:
            dyn_mask = (False,) * sr
        if dyn_origins is None:
            dyn_origins = (None,) * sr
        assert len(dyn_mask) == sr, (
            f"StepTensor: dyn_mask len {len(dyn_mask)} != stream_rank {sr} "
            f"(tensor.ndim={tensor.ndim}, stream_dtype={stream_dtype!r})"
        )
        assert len(dyn_origins) == sr
        self.tensor = tensor
        self.stream_dtype = stream_dtype
        self.dyn_mask = tuple(dyn_mask)
        self.dyn_origins = tuple(dyn_origins)
        self.offsets = offsets

    @property
    def shape(self):
        return _GuardedShape(self.tensor.shape, self.dyn_mask, self.dyn_origins,
                             "StepTensor", _elem_dims(self.stream_dtype))

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def stream_rank(self):
        return _stream_rank(self.tensor, self.stream_dtype)

    def __repr__(self):
        return (f"StepTensor(shape={tuple(self.tensor.shape)}, "
                f"stream_dtype={self.stream_dtype}, dyn_mask={self.dyn_mask})")


# ---------------------------------------------------------------------------
# Migration helpers used by converted DSL ops.
# ---------------------------------------------------------------------------


def _unwrap(x):
    """Return the underlying torch.Tensor for either a raw tensor or StepTensor."""
    if isinstance(x, StepTensor):
        return x.tensor
    assert isinstance(x, torch.Tensor), \
        f"expected torch.Tensor or StepTensor, got {type(x).__name__}"
    return x


def _step_meta(x, op_name):
    """Require a StepTensor and return (stream_dtype, dyn_mask, dyn_origins).

    Used by ops that need to gate on element/tile type or propagate dynamism.
    Source ops (which take raw inputs like off-chip memory) don't call this.
    """
    assert isinstance(x, StepTensor), (
        f"{op_name}: input must be a StepTensor (typed). Got "
        f"{type(x).__name__}. Source ops produce StepTensor; chain DSL ops "
        f"to keep the wrapper attached, or wrap explicitly."
    )
    return x.stream_dtype, x.dyn_mask, x.dyn_origins


def _assert_tile_kind(stream_dtype, op_name, allowed=(Tile, DynTile)):
    assert isinstance(stream_dtype, allowed), (
        f"{op_name}: stream_dtype must be one of "
        f"{tuple(c.__name__ for c in allowed)}, got {stream_dtype!r}"
    )


def _assert_elem_in(stream_dtype, op_name, allowed):
    """Check that stream_dtype is a tile and its element type is allowed.

    `allowed` is a tuple of element-tag classes (e.g. (Float16, Float32)).
    """
    _assert_tile_kind(stream_dtype, op_name)
    assert isinstance(stream_dtype.tile_dtype, tuple(allowed)), (
        f"{op_name}: tile element dtype must be one of "
        f"{tuple(a.__name__ for a in allowed)}, got {stream_dtype.tile_dtype!r}"
    )


def offchip_load(underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False, *, par_dispatch=1):
    assert par_dispatch >= 1, f"offchip_load: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], f"offchip_load: underlying dtype must be float16 or float32, got {underlying.dtype}"
    # out_shape_tiled enumerates the stream positions to read; an empty tuple
    # produces a zero-rank stream which the rest of the DSL cannot represent
    # (every StepTensor has at least one stream dim). Catch this up front
    # with a concrete fix-up suggestion — otherwise the failure surfaces deep
    # in torch as "meshgrid expects a non-empty TensorList", which gives the
    # caller no signal about what to change.
    assert len(out_shape_tiled) >= 1, (
        f"offchip_load: out_shape_tiled must have at least one streaming "
        f"dimension, got {out_shape_tiled!r}. To load the underlying as a "
        f"single tile, use out_shape_tiled=(1,) with stride=(1,) and a "
        f"tile_row/tile_col that covers the whole matrix; the result has "
        f"stream shape (1,) and can be broadcast (e.g. via offchip_load_ref) "
        f"to a wider consumer stream."
    )
    R, C = underlying.shape[-2], underlying.shape[-1]

    # ---- Tiling invariant: must actually stream, not load as one giant tile ----
    # If the tensor is larger than one tile, there must be a streaming dimension.
    total_tiles = (R // tile_row) * (C // tile_col)
    if total_tiles > 1:
        assert any(s > 1 for s in out_shape_tiled), (
            f"offchip_load: tensor ({R}, {C}) has {total_tiles} tiles but "
            f"out_shape_tiled={out_shape_tiled} has no streaming dim > 1. "
            f"Use proper streaming, e.g. out_shape_tiled=({R // tile_row},) "
            f"with tile_row={tile_row}, tile_col={tile_col}."
        )
    grid_r, grid_c = R // tile_row, C // tile_col
    batch_shape = underlying.shape[:-2]

    # Truncate to evenly divisible size
    used_R, used_C = grid_r * tile_row, grid_c * tile_col
    if used_R != R or used_C != C:
        underlying = underlying[..., :used_R, :used_C]

    # Reshape to tile grid: (*batch, grid_r, tile_row, grid_c, tile_col)
    tiled = underlying.reshape(*batch_shape, grid_r, tile_row, grid_c, tile_col)
    # Permute so tile dims are last: (*batch, grid_r, grid_c, tile_row, tile_col)
    ndim = tiled.ndim
    perm = list(range(len(batch_shape))) + [ndim - 4, ndim - 2, ndim - 3, ndim - 1]
    tiled = tiled.permute(*perm)
    # Flatten batch + grid into single tile index
    flat = tiled.reshape(-1, tile_row, tile_col)

    # Compute linear tile index for every position in out_shape_tiled
    ranges = [torch.arange(s) for s in out_shape_tiled]
    grids = torch.meshgrid(*ranges, indexing="ij")
    linear_idx = sum(g.long() * int(s) for g, s in zip(grids, stride))

    result = flat[linear_idx.long()]  # (*out_shape_tiled, tile_row, tile_col)
    if transposed:
        result = result.transpose(-2, -1)
    result = result.unsqueeze(0)  # prepend leading 1
    tile_shape = (tile_col, tile_row) if transposed else (tile_row, tile_col)
    return StepTensor(
        result, stream_dtype=Tile(_elem_from_torch(underlying.dtype), tile_shape)
    )


def dyn_offchip_load(underlying, tensor_shape_tiled, tile_row, tile_col, *, par_dispatch=1):
    assert par_dispatch >= 1, f"dyn_offchip_load: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], (
        f"dyn_offchip_load: underlying dtype must be float16 or float32, got {underlying.dtype}"
    )
    R, C = underlying.shape[-2], underlying.shape[-1]
    assert R % tile_row == 0 and C % tile_col == 0, (
        f"dyn_offchip_load: ({R},{C}) not divisible by tile ({tile_row},{tile_col})"
    )
    grid_r, grid_c = R // tile_row, C // tile_col
    tiled = underlying.reshape(grid_r, tile_row, grid_c, tile_col).permute(0, 2, 1, 3)
    result = tiled.reshape(*tensor_shape_tiled, tile_row, tile_col).unsqueeze(0)
    # IR DynLinearOffChipLoad: stream_dtype is Tile (the "dyn" refers to the
    # stream shape, which may contain DynDims; the tile shape itself is static).
    # The DSL caller always supplies concrete ints for tensor_shape_tiled, so
    # we emit an all-static dyn_mask here.
    return StepTensor(
        result,
        stream_dtype=Tile(_elem_from_torch(underlying.dtype), (tile_row, tile_col)),
    )


def offchip_load_ref(ref, underlying, stride, out_shape_tiled, tile_row, tile_col, transposed=False, *, par_dispatch=1):
    assert par_dispatch >= 1, f"offchip_load_ref: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], f"offchip_load_ref: underlying dtype must be float16 or float32, got {underlying.dtype}"
    sd_ref, mask_ref, orig_ref = _step_meta(ref, "offchip_load_ref (ref)")
    _assert_tile_kind(sd_ref, "offchip_load_ref (ref)")
    loaded = offchip_load(underlying, stride, out_shape_tiled, tile_row, tile_col, transposed).tensor
    # loaded: (1, *out_shape_tiled, tile_row, tile_col)
    # target: (*ref_stream, *out_shape_tiled, tile_row, tile_col)
    ref_stream = list(ref.tensor.shape[:-2])
    target = ref_stream + list(out_shape_tiled) + [loaded.shape[-2], loaded.shape[-1]]
    # Prepend singleton dims so loaded is broadcastable to target
    while loaded.ndim < len(target):
        loaded = loaded.unsqueeze(0)
    result = loaded.expand(target).contiguous()
    tile_shape = (tile_col, tile_row) if transposed else (tile_row, tile_col)
    # Output stream = ref.stream + out_shape_tiled (out_shape_tiled is static).
    return StepTensor(
        result,
        stream_dtype=Tile(_elem_from_torch(underlying.dtype), tile_shape),
        dyn_mask=mask_ref + (False,) * len(out_shape_tiled),
        dyn_origins=orig_ref + (None,) * len(out_shape_tiled),
    )

def select_gen(underlying, is_multihot, n):
    underlying = _unwrap(underlying)
    _assert_int(underlying, "select_gen")
    assert isinstance(is_multihot, bool), (
        f"select_gen: is_multihot must be bool, got {type(is_multihot).__name__}"
    )
    assert underlying.shape[-1] == n, (
        f"select_gen: control's last dim must equal n={n}, "
        f"got shape {tuple(underlying.shape)}"
    )
    # Prepend a leading singleton so the DSL stream rank matches STeP's
    # SelectGen, which produces stream=(1,)+tensor.shape[:-1]. Without this,
    # downstream ops (notably expert_addr_gen) end up one rank shorter in DSL
    # than in the translated STeP graph, hiding rank-sensitive shape bugs
    # until the build_graph stage.
    out = underlying.unsqueeze(0)
    return StepTensor(
        out, stream_dtype=MultiHot(n) if is_multihot else Index(n)
    )

def metadata_gen(tensor):
    tensor = _unwrap(tensor)
    out = tensor.reshape(1, *tensor.shape, 1, 1)
    # IR MetadataGen always produces Tile(Uint64, (1,1)) regardless of the
    # eager-runtime torch dtype; the underlying tensor stores the metadata
    # values (addresses/sizes) which are conceptually uint64.
    return StepTensor(out, stream_dtype=Tile(Uint64(), (1, 1)))


def cache_read_addr_gen(idx, seq_len, row_offset):
    sd_i, _, _ = _step_meta(idx, "cache_read_addr_gen (idx)")
    sd_s, _, _ = _step_meta(seq_len, "cache_read_addr_gen (seq_len)")
    _assert_tile_kind(sd_i, "cache_read_addr_gen (idx)")
    _assert_tile_kind(sd_s, "cache_read_addr_gen (seq_len)")
    idx_t = idx.tensor
    seq_t = seq_len.tensor
    assert idx_t.shape[-2:] == (1, 1), (
        f"cache_read_addr_gen: idx tile shape must be (1,1), got {tuple(idx_t.shape[-2:])}"
    )
    assert seq_t.shape == idx_t.shape, (
        f"cache_read_addr_gen: idx {tuple(idx_t.shape)} and seq_len {tuple(seq_t.shape)} must match"
    )
    idx_flat = idx_t.reshape(-1).long()
    seq_len_flat = seq_t.reshape(-1).long()
    out = []
    for b in range(idx_flat.shape[0]):
        base = int(idx_flat[b]) * int(row_offset)
        n = int(seq_len_flat[b])
        assert n >= 0, f"cache_read_addr_gen: seq_len[{b}]={n} must be >= 0"
        # Per-batch output stream shape (1, n) with tile (1,1). Each n is
        # concrete eager-time but represents a ragged dim; we list one
        # StepTensor per batch so the raggedness lives in the Python list
        # rather than in dyn_mask of a single tensor.
        t = torch.arange(base, base + n, dtype=torch.float32).reshape(1, n, 1, 1)
        out.append(StepTensor(t, stream_dtype=Tile(Uint64(), (1, 1))))
    return out


def expert_addr_gen(x, expert_addr_base, num_tile_per_expert):
    sd, mask, orig = _step_meta(x, "expert_addr_gen")
    assert isinstance(sd, Index), (
        f"expert_addr_gen: input stream_dtype must be Index (one-hot Select), got {sd!r}"
    )
    t = x.tensor
    assert (t.sum(dim=-1) == 1).all(), (
        "expert_addr_gen: input must be one-hot (exactly one expert selected per element)"
    )
    expert_indices = t.argmax(dim=-1)
    base = expert_addr_base + expert_indices * num_tile_per_expert
    offsets = torch.arange(num_tile_per_expert, dtype=base.dtype)
    addrs = base.unsqueeze(-1) + offsets
    result = addrs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # Input stream rank = ndim - 1 (Select elem_dims=1); output stream rank =
    # input + 2 (new num_tile_per_expert dim + the synthesized (1,1) "leading"
    # tile pair before the final tile rc — see argmax+unsqueeze chain). The
    # appended dims are static.
    return StepTensor(
        result,
        stream_dtype=Tile(Uint64(), (1, 1)),
        dyn_mask=mask + (False, False),
        dyn_origins=orig + (None, None),
    )


def filter_last_tile(seq_len):
    sd, mask, orig = _step_meta(seq_len, "filter_last_tile")
    _assert_tile_kind(sd, "filter_last_tile")
    t = seq_len.tensor
    assert t.shape[-2:] == (1, 1), (
        f"filter_last_tile: input tile shape must be (1,1), got {tuple(t.shape[-2:])}"
    )
    stream_shape = t.shape[:-2]
    flat = t.reshape(-1).long()
    assert (flat >= 1).all(), (
        f"filter_last_tile: seq_len must be >= 1 (Rust ref assumes >=1), got min={int(flat.min())}"
    )
    max_seq = int(flat.max().item())

    out = torch.zeros(flat.shape[0], max_seq, 2)
    for i in range(flat.shape[0]):
        n = int(flat[i])
        out[i, :n - 1, 1] = 1.0  # non-last -> column 1
        out[i, n - 1, 0] = 1.0   # last     -> column 0
    result = out.reshape(*stream_shape, max_seq, 2)
    # IR FilterLastTile (utility_ops.py:167): MultiHot(2), stream =
    # in_stream.shape + (DynDim,). Append a new dyn stream slot here.
    # Input had stream rank len(stream_shape); output has +1 (the max_seq dim).
    # Output elem_dims=1 (MultiHot), so stream_rank = ndim - 1.
    # ndim = len(stream_shape) + 2; stream_rank = len(stream_shape) + 1.
    return StepTensor(
        result,
        stream_dtype=MultiHot(2),
        dyn_mask=mask + (True,),
        dyn_origins=orig + ("filter_last_tile",),
    )


def random_offchip_load(underlying, raddr, tile_row, tile_col, transposed=False, *, par_dispatch=1):
    assert par_dispatch >= 1, f"random_offchip_load: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], (
        f"random_offchip_load: underlying dtype must be float16 or float32, got {underlying.dtype}"
    )
    sd_r, mask_r, orig_r = _step_meta(raddr, "random_offchip_load (raddr)")
    _assert_tile_kind(sd_r, "random_offchip_load (raddr)")
    raddr_t = raddr.tensor
    assert raddr_t.shape[-2:] == (1, 1), (
        f"random_offchip_load: raddr tile shape must be (1,1), got {tuple(raddr_t.shape[-2:])}"
    )
    R, C = underlying.shape[-2], underlying.shape[-1]
    assert R % tile_row == 0 and C % tile_col == 0, (
        f"random_offchip_load: ({R},{C}) not divisible by tile ({tile_row},{tile_col})"
    )
    grid_r, grid_c = R // tile_row, C // tile_col
    batch_shape = underlying.shape[:-2]
    tiled = underlying.reshape(*batch_shape, grid_r, tile_row, grid_c, tile_col)
    ndim = tiled.ndim
    perm = list(range(len(batch_shape))) + [ndim - 4, ndim - 2, ndim - 3, ndim - 1]
    flat = tiled.permute(*perm).reshape(-1, tile_row, tile_col)

    stream_shape = raddr_t.shape[:-2]
    addrs = raddr_t.reshape(-1).long()
    assert (addrs >= 0).all() and (addrs < flat.shape[0]).all(), (
        f"random_offchip_load: address out of range [0, {flat.shape[0]}), "
        f"got min={int(addrs.min())}, max={int(addrs.max())}"
    )
    result = flat[addrs].reshape(*stream_shape, tile_row, tile_col)
    if transposed:
        result = result.transpose(-2, -1)
    tile_shape = (tile_col, tile_row) if transposed else (tile_row, tile_col)
    # Output stream shape == raddr stream shape, dyn_mask matches raddr.
    return StepTensor(
        result,
        stream_dtype=Tile(_elem_from_torch(underlying.dtype), tile_shape),
        dyn_mask=mask_r,
        dyn_origins=orig_r,
    )

def _assert_stream_match(a, b, op_name):
    """Strict: physical stream shape AND dyn_mask must agree on both sides."""
    sd_a, mask_a, orig_a = _step_meta(a, op_name)
    sd_b, mask_b, orig_b = _step_meta(b, op_name)
    a_elem = _elem_dims(sd_a)
    b_elem = _elem_dims(sd_b)
    a_stream = tuple(a.tensor.shape[: a.tensor.ndim - a_elem])
    b_stream = tuple(b.tensor.shape[: b.tensor.ndim - b_elem])
    assert a_stream == b_stream, (
        f"{op_name}: stream shape mismatch — a has stream {a_stream} "
        f"(shape {tuple(a.tensor.shape)}, dtype {sd_a!r}) but b has stream "
        f"{b_stream} (shape {tuple(b.tensor.shape)}, dtype {sd_b!r}). Both "
        f"operands must have identical stream shapes."
    )
    assert mask_a == mask_b, (
        f"{op_name}: dyn_mask mismatch — a has {mask_a} (origins {orig_a}); "
        f"b has {mask_b} (origins {orig_b}). Strict matching is required so "
        f"a symbolic dim isn't silently aligned to a static one."
    )


def _assert_float(x, op_name):
    """Stream-level: x is a StepTensor wrapping a Float16/Float32 Tile|DynTile."""
    sd, _, _ = _step_meta(x, op_name)
    _assert_elem_in(sd, op_name, (Float16, Float32))


def _assert_int(x, op_name):
    """Raw or stream: gate on int dtype. For source-op raw inputs (e.g.
    select_gen control), checks `x.dtype`; for StepTensor inputs, checks the
    tile element type against (Uint32, Uint64)."""
    if isinstance(x, StepTensor):
        _assert_elem_in(x.stream_dtype, op_name, (Uint32, Uint64))
        return
    assert isinstance(x, torch.Tensor) and x.dtype in (torch.int32, torch.int64), (
        f"{op_name}: input dtype must be int32 or int64, got {getattr(x, 'dtype', type(x))}."
    )


def binary_matmul(a, b, weight_transposed=False, *, compute_bw=1):
    assert compute_bw >= 1, f"binary_matmul: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_matmul")
    _assert_float(b, "binary_matmul")
    _assert_stream_match(a, b, "binary_matmul")
    if weight_transposed:
        result = torch.matmul(a.tensor, b.tensor.transpose(-2, -1))
        tile_shape = (a.stream_dtype.shape[0], b.stream_dtype.shape[0])
    else:
        result = torch.matmul(a.tensor, b.tensor)
        tile_shape = (a.stream_dtype.shape[0], b.stream_dtype.shape[1])
    return StepTensor(
        result,
        stream_dtype=Tile(a.stream_dtype.tile_dtype, tile_shape),
        dyn_mask=a.dyn_mask, dyn_origins=a.dyn_origins,
    )


def _elementwise_binary(a, b, op_name, fn):
    """Shared body for elementwise binary ops with identity tile/dtype passthrough.

    Output tile shape comes from torch broadcasting on the underlying tensors;
    output element type follows `a` (mixed-elem inputs are rejected upstream
    via _assert_float / _assert_elem_in if needed).
    """
    _assert_float(a, op_name)
    _assert_float(b, op_name)
    _assert_stream_match(a, b, op_name)
    result = fn(a.tensor, b.tensor)
    tile_shape = (int(result.shape[-2]), int(result.shape[-1]))
    return StepTensor(
        result,
        stream_dtype=Tile(a.stream_dtype.tile_dtype, tile_shape),
        dyn_mask=a.dyn_mask, dyn_origins=a.dyn_origins,
    )


def binary_mul(a, b, *, compute_bw=1):
    assert compute_bw >= 1, f"binary_mul: compute_bw must be >= 1, got {compute_bw}"
    return _elementwise_binary(a, b, "binary_mul", lambda x, y: x * y)


def binary_add(a, b, *, compute_bw=1):
    assert compute_bw >= 1, f"binary_add: compute_bw must be >= 1, got {compute_bw}"
    return _elementwise_binary(a, b, "binary_add", lambda x, y: x + y)


def binary_div(a, b, *, compute_bw=1):
    assert compute_bw >= 1, f"binary_div: compute_bw must be >= 1, got {compute_bw}"
    return _elementwise_binary(a, b, "binary_div", lambda x, y: x / y)


def binary_is_equal(a, b, *, compute_bw=1):
    assert compute_bw >= 1, f"binary_is_equal: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_is_equal")
    _assert_float(b, "binary_is_equal")
    _assert_stream_match(a, b, "binary_is_equal")
    result = (a.tensor == b.tensor).float()
    tile_shape = (int(result.shape[-2]), int(result.shape[-1]))
    # Output is logically a Bool tile (mirrors IR). The eager torch tensor is
    # stored as float32 for compatibility with downstream float ops, but the
    # wrapper-level dtype reports Bool so consumers can gate correctly.
    return StepTensor(
        result,
        stream_dtype=Tile(Bool(), tile_shape),
        dyn_mask=a.dyn_mask, dyn_origins=a.dyn_origins,
    )


def binary_set_offset(a, b, *, compute_bw=1):
    assert compute_bw >= 1, f"binary_set_offset: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_set_offset")
    _assert_float(b, "binary_set_offset")
    _assert_stream_match(a, b, "binary_set_offset")
    assert b.tensor.shape[-2:] == (1, 1), (
        f"binary_set_offset: b tile shape must be (1,1), got {tuple(b.tensor.shape[-2:])}"
    )
    offsets = b.tensor[..., 0, 0].long()
    # Carry `a` forward unchanged but attach offsets on the wrapper for the
    # downstream binary_row_wise_append to consume.
    return StepTensor(
        a.tensor, stream_dtype=a.stream_dtype,
        dyn_mask=a.dyn_mask, dyn_origins=a.dyn_origins,
        offsets=offsets,
    )


def binary_row_wise_append(a, b, *, compute_bw=1):
    assert compute_bw >= 1, f"binary_row_wise_append: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_row_wise_append")
    _assert_float(b, "binary_row_wise_append")
    _assert_stream_match(a, b, "binary_row_wise_append")
    data = a.tensor
    if a.offsets is not None:
        offsets = a.offsets
    else:
        offsets = torch.zeros(data.shape[:-2], dtype=torch.long)
    tile_r, tile_c = data.shape[-2], data.shape[-1]
    M = b.tensor.shape[-2]
    assert b.tensor.shape[-1] == tile_c, (
        f"binary_row_wise_append: column dim mismatch ({b.tensor.shape[-1]} vs {tile_c})"
    )
    assert (offsets + M <= tile_r).all(), (
        f"binary_row_wise_append: not enough space to append {M} rows "
        f"(tile_r={tile_r}, max offset={int(offsets.max())})"
    )
    stream_shape = data.shape[:-2]
    row_idx = offsets.unsqueeze(-1) + torch.arange(M, dtype=torch.long, device=data.device)
    row_idx = row_idx.unsqueeze(-1).expand(*stream_shape, M, tile_c)
    result = data.clone()
    result.scatter_(dim=-2, index=row_idx, src=b.tensor.to(data.dtype))
    return StepTensor(
        result, stream_dtype=a.stream_dtype,
        dyn_mask=a.dyn_mask, dyn_origins=a.dyn_origins,
    )


def binary_cache_write_addr_gen(idx, seq_len, row_offset, *, compute_bw=1):
    """Compute KV-cache write address: ``idx * row_offset + seq_len``.

    Mirrors step-perf/map_fn::cache_write_addr_gen. ``idx`` and ``seq_len`` are
    (*stream, 1, 1) scalar tiles; ``row_offset`` is a Python int.
    """
    assert compute_bw >= 1, f"binary_cache_write_addr_gen: compute_bw must be >= 1, got {compute_bw}"
    sd_i, mask_i, orig_i = _step_meta(idx, "binary_cache_write_addr_gen (idx)")
    sd_s, _, _ = _step_meta(seq_len, "binary_cache_write_addr_gen (seq_len)")
    _assert_tile_kind(sd_i, "binary_cache_write_addr_gen (idx)")
    _assert_tile_kind(sd_s, "binary_cache_write_addr_gen (seq_len)")
    _assert_stream_match(idx, seq_len, "binary_cache_write_addr_gen")
    idx_t, seq_t = idx.tensor, seq_len.tensor
    assert idx_t.shape[-2:] == (1, 1), (
        f"binary_cache_write_addr_gen: idx tile shape must be (1,1), got {tuple(idx_t.shape[-2:])}"
    )
    assert seq_t.shape == idx_t.shape, (
        f"binary_cache_write_addr_gen: idx {tuple(idx_t.shape)} and seq_len "
        f"{tuple(seq_t.shape)} must match"
    )
    assert isinstance(row_offset, int), (
        f"binary_cache_write_addr_gen: row_offset must be int, got {type(row_offset).__name__}"
    )
    result = idx_t * row_offset + seq_t
    return StepTensor(
        result, stream_dtype=Tile(Uint64(), (1, 1)),
        dyn_mask=mask_i, dyn_origins=orig_i,
    )


# ---------------------------------------------------------------------------
# Unary compute: UnaryMap → unary_*
# Mirrors: _apply_unary (L443) from functional.py
# ---------------------------------------------------------------------------

def _identity_unary(x, op_name, fn, allowed=(Float16, Float32)):
    """Apply `fn` to x.tensor and rewrap. Element type preserved; tile shape
    derived from the result (allowing ops like rowwise_sum to shrink it)."""
    sd, mask, orig = _step_meta(x, op_name)
    _assert_elem_in(sd, op_name, allowed)
    result = fn(x.tensor)
    tile_shape = (int(result.shape[-2]), int(result.shape[-1]))
    if isinstance(sd, DynTile):
        new_sd = DynTile(sd.tile_dtype, tile_shape)
    else:
        new_sd = Tile(sd.tile_dtype, tile_shape)
    return StepTensor(result, stream_dtype=new_sd,
                      dyn_mask=mask, dyn_origins=orig)


def unary_silu(x, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_silu: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_silu", F.silu)


def unary_square(x, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_square: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_square", lambda t: t ** 2)


def unary_exp(x, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_exp: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_exp", torch.exp)


def unary_rsqrt(x, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_rsqrt: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_rsqrt", torch.rsqrt)


def unary_pow2(x, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_pow2: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_pow2", lambda t: torch.pow(2.0, t))


def unary_mul_imm(x, constant, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_mul_imm: compute_bw must be >= 1, got {compute_bw}"
    assert constant != 0.0, "unary_mul_imm: constant must be nonzero."
    return _identity_unary(x, "unary_mul_imm", lambda t: t * constant)


def unary_add_imm(x, constant, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_add_imm: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_add_imm", lambda t: t + constant)


def unary_sub_imm(x, constant, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_sub_imm: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_sub_imm", lambda t: t - constant)


def unary_rowwise_sum(x, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_rowwise_sum: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_rowwise_sum",
                           lambda t: t.sum(dim=-1, keepdim=True))


def unary_mask_row(x, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_mask_row: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_mask_row",
                           lambda t: torch.ones_like(t[..., :1]))


def unary_select_to_scalar(x, *, compute_bw=1):
    assert compute_bw >= 1, f"unary_select_to_scalar: compute_bw must be >= 1, got {compute_bw}"
    return _identity_unary(x, "unary_select_to_scalar", lambda t: t)


def unary_to_const_int(x, constant, *, compute_bw=1):
    """Promote a float-tile stream to an integer-tile stream of constants.

    Output stream_dtype = Tile(Uint64, ...) per the IR convention; the eager
    torch tensor stays float32 for simulator compatibility (mirrors how
    metadata_gen / binary_cache_write_addr_gen declare Uint64 while the
    underlying storage is float).
    """
    assert compute_bw >= 1, f"unary_to_const_int: compute_bw must be >= 1, got {compute_bw}"
    sd, mask, orig = _step_meta(x, "unary_to_const_int")
    _assert_elem_in(sd, "unary_to_const_int", (Float16, Float32))
    result = torch.full_like(x.tensor, constant, dtype=torch.float32)
    tile_shape = (int(result.shape[-2]), int(result.shape[-1]))
    return StepTensor(
        result, stream_dtype=Tile(Uint64(), tile_shape),
        dyn_mask=mask, dyn_origins=orig,
    )

def _accum_reduce(x, rank, op_name, reduce_fn):
    """Drop the last `rank` stream dims by reducing along dim=-3 each time.

    Output element type and tile shape are preserved; stream rank shrinks
    by `rank`.
    """
    sd, mask, orig = _step_meta(x, op_name)
    _assert_elem_in(sd, op_name, (Float16, Float32))
    assert rank > 0, f"{op_name}: rank must be > 0, got {rank}"
    assert rank <= len(mask), (
        f"{op_name}: rank {rank} exceeds input stream rank {len(mask)}"
    )
    t = x.tensor
    for _ in range(rank):
        t = reduce_fn(t)
    return StepTensor(
        t, stream_dtype=sd,
        dyn_mask=mask[:len(mask) - rank],
        dyn_origins=orig[:len(orig) - rank],
    )


def accum_add(x, rank=1, *, compute_bw=1):
    assert compute_bw >= 1, f"accum_add: compute_bw must be >= 1, got {compute_bw}"
    return _accum_reduce(x, rank, "accum_add", lambda t: t.sum(dim=-3))


def accum_mul(x, rank=1, *, compute_bw=1):
    assert compute_bw >= 1, f"accum_mul: compute_bw must be >= 1, got {compute_bw}"
    return _accum_reduce(x, rank, "accum_mul", lambda t: t.prod(dim=-3))


def accum_max(x, rank=1, *, compute_bw=1):
    assert compute_bw >= 1, f"accum_max: compute_bw must be >= 1, got {compute_bw}"
    return _accum_reduce(x, rank, "accum_max", lambda t: t.amax(dim=-3))


def accum_retile_row(x, rank=1, *, compute_bw=1):
    """Absorb the innermost `rank` stream dims into tile_r.

    Each absorbed dim multiplies tile_r by its (static) size. Dynamic stream
    dims cannot be absorbed — the resulting tile shape would be symbolic,
    which the IR doesn't allow on a Tile (would need DynTile, and there's
    no clean DSL path to that).
    """
    assert compute_bw >= 1, f"accum_retile_row: compute_bw must be >= 1, got {compute_bw}"
    sd, mask, orig = _step_meta(x, "accum_retile_row")
    _assert_tile_kind(sd, "accum_retile_row")
    assert rank > 0, f"accum_retile_row: rank must be > 0, got {rank}"
    assert rank <= len(mask), (
        f"accum_retile_row: rank {rank} exceeds input stream rank {len(mask)}"
    )
    for i in range(rank):
        slot = len(mask) - 1 - i
        assert not mask[slot], (
            f"accum_retile_row: stream dim {slot} is dynamic "
            f"(origin: {orig[slot]}); cannot absorb into a static tile."
        )
    t = x.tensor
    tile_r = int(sd.shape[0])
    for _ in range(rank):
        s = t.shape
        tile_r = tile_r * int(s[-3])
        t = t.reshape(*s[:-3], s[-3] * s[-2], s[-1])
    tile_shape = (tile_r, int(sd.shape[1]))
    return StepTensor(
        t, stream_dtype=Tile(sd.tile_dtype, tile_shape),
        dyn_mask=mask[:len(mask) - rank],
        dyn_origins=orig[:len(orig) - rank],
    )


def accum_retile_col(x, rank=1, *, compute_bw=1):
    """Absorb the innermost `rank` stream dims into tile_c.

    See accum_retile_row for the dynamism caveat.
    """
    assert compute_bw >= 1, f"accum_retile_col: compute_bw must be >= 1, got {compute_bw}"
    sd, mask, orig = _step_meta(x, "accum_retile_col")
    _assert_tile_kind(sd, "accum_retile_col")
    assert rank > 0, f"accum_retile_col: rank must be > 0, got {rank}"
    assert rank <= len(mask), (
        f"accum_retile_col: rank {rank} exceeds input stream rank {len(mask)}"
    )
    for i in range(rank):
        slot = len(mask) - 1 - i
        assert not mask[slot], (
            f"accum_retile_col: stream dim {slot} is dynamic "
            f"(origin: {orig[slot]}); cannot absorb into a static tile."
        )
    t = x.tensor
    tile_c = int(sd.shape[1])
    for _ in range(rank):
        s = t.shape
        # Permute the accum dim (dim -3) next to the tile-col dim (dim -1)
        # so the row-major reshape produces col-concatenated tiles.
        ndim = t.ndim
        perm = list(range(ndim - 3)) + [ndim - 2, ndim - 3, ndim - 1]
        t = t.permute(perm).contiguous()
        tile_c = tile_c * int(s[-3])
        t = t.reshape(*s[:-3], s[-2], s[-3] * s[-1])
    tile_shape = (int(sd.shape[0]), tile_c)
    return StepTensor(
        t, stream_dtype=Tile(sd.tile_dtype, tile_shape),
        dyn_mask=mask[:len(mask) - rank],
        dyn_origins=orig[:len(orig) - rank],
    )


def accum_signal_req_all_read(x, rank=1, *, compute_bw=1):
    """Reduce `rank` stream dims, emitting a (1,1) all-ones ack tile per
    remaining stream element. Output stays a Float tile (matches IR)."""
    assert compute_bw >= 1, f"accum_signal_req_all_read: compute_bw must be >= 1, got {compute_bw}"
    sd, mask, orig = _step_meta(x, "accum_signal_req_all_read")
    _assert_elem_in(sd, "accum_signal_req_all_read", (Float16, Float32))
    assert rank > 0, f"accum_signal_req_all_read: rank must be > 0, got {rank}"
    assert rank <= len(mask), (
        f"accum_signal_req_all_read: rank {rank} exceeds input stream rank {len(mask)}"
    )
    stream_shape = x.tensor.shape[: x.tensor.ndim - 2 - rank]
    result = torch.ones(*stream_shape, 1, 1)
    return StepTensor(
        result, stream_dtype=Tile(sd.tile_dtype, (1, 1)),
        dyn_mask=mask[:len(mask) - rank],
        dyn_origins=orig[:len(orig) - rank],
    )

def eager_merge(inputs):
    """Concatenate `inputs` along their outermost stream dim and emit a
    MultiHot(num_inputs) selector that recovers each source.

    Outer-dim dynamism: if any input has a dynamic outer slot, the merged
    output (and the selector) inherit dynamism with origin "eager_merge".
    Inner stream dims must match exactly (incl dyn_mask) across inputs."""
    n = len(inputs)
    assert n > 0, "eager_merge: must have at least one input"
    sd0, mask0, _ = _step_meta(inputs[0], "eager_merge (inputs[0])")
    _assert_tile_kind(sd0, "eager_merge (inputs[0])")
    assert len(mask0) >= 1, (
        f"eager_merge: inputs must have at least one stream dim, got "
        f"shape {tuple(inputs[0].tensor.shape)}"
    )
    tile_shape = sd0.shape
    any_dyn_outer = bool(mask0[0])
    for i, p in enumerate(inputs):
        sd_i, mask_i, _ = _step_meta(p, f"eager_merge (inputs[{i}])")
        _assert_tile_kind(sd_i, f"eager_merge (inputs[{i}])")
        assert sd_i.shape == tile_shape, (
            f"eager_merge: input {i} tile shape {sd_i.shape} != {tile_shape}"
        )
        assert mask_i[1:] == mask0[1:], (
            f"eager_merge: input {i} inner dyn_mask {mask_i[1:]} != {mask0[1:]}"
        )
        any_dyn_outer = any_dyn_outer or bool(mask_i[0])

    data = torch.cat([p.tensor for p in inputs], dim=0)
    counts = [p.tensor.shape[0] for p in inputs]
    select = torch.zeros(sum(counts), n)
    offset = 0
    for i, c in enumerate(counts):
        select[offset:offset + c, i] = 1.0
        offset += c

    outer_origin = "eager_merge" if any_dyn_outer else None
    data_mask = (any_dyn_outer,) + mask0[1:]
    # Inner origins come from inputs[0] (all inputs share matching inner masks).
    data_origins = (outer_origin,) + inputs[0].dyn_origins[1:]
    return [
        StepTensor(data, stream_dtype=sd0,
                   dyn_mask=data_mask, dyn_origins=data_origins),
        StepTensor(select, stream_dtype=MultiHot(n),
                   dyn_mask=(any_dyn_outer,),
                   dyn_origins=(outer_origin,)),
    ]


def flat_partition(x, control, n):
    sd_x, _, _ = _step_meta(x, "flat_partition")
    sd_c, _, _ = _step_meta(control, "flat_partition")
    _assert_tile_kind(sd_x, "flat_partition (input)")
    assert isinstance(sd_c, _SelectBase), (
        f"flat_partition: control stream_dtype must be MultiHot/Index, "
        f"got {sd_c!r}"
    )
    assert sd_c.total_n == n, (
        f"flat_partition: control total_n={sd_c.total_n} != n={n}"
    )

    t = x.tensor
    c = control.tensor
    assert c.shape[-1] == n, (
        f"flat_partition: control's last dim must equal n={n} (num consumers), "
        f"got control shape {tuple(c.shape)}."
    )
    tile_r, tile_c = t.shape[-2], t.shape[-1]
    flat_inp = t.reshape(-1, tile_r, tile_c)
    flat_mh = c.reshape(-1, n)
    assert flat_inp.shape[0] == flat_mh.shape[0], \
        f"Tile count mismatch: input has {flat_inp.shape[0]} vs selector with {flat_mh.shape[0]}. The input stream and selector stream must have the same number of stream elements, meaning that x.reshape(-1, tile_r, tile_c) and control.reshape(-1, n) must resolve to the same number of elements in the upper (-1) dimensions."

    results = []
    for i in range(n):
        mask = flat_mh[:, i] > 0
        # Each output is a rank-1 stream whose single stream dim is dynamic
        # (its size depends on the runtime contents of the selector). Mirrors
        # ops.py:2785 — FlatPartition adds an outermost DynDim per consumer.
        results.append(StepTensor(
            flat_inp[mask],
            stream_dtype=sd_x,
            dyn_mask=(True,),
            dyn_origins=("flat_partition",),
        ))
    return results


def flat_reassemble(inputs, control):
    """Round-robin reassemble selected tokens per `control` row.

    Output stream shape = control.stream + (n_active,) where n_active is a
    new dynamic dim (mirrors IR ops.py:3120 — FlatReassemble inserts a
    DynDim between control's stream and the reassemble tail).
    """
    n = len(inputs)
    assert n > 0, "flat_reassemble: must have at least one input"
    sd0, _, _ = _step_meta(inputs[0], "flat_reassemble (inputs[0])")
    _assert_tile_kind(sd0, "flat_reassemble (inputs[0])")
    tile_shape = sd0.shape
    for i, p in enumerate(inputs):
        sd_i, _, _ = _step_meta(p, f"flat_reassemble (inputs[{i}])")
        _assert_tile_kind(sd_i, f"flat_reassemble (inputs[{i}])")
        assert sd_i.shape == tile_shape, (
            f"flat_reassemble: input {i} tile shape {sd_i.shape} != {tile_shape}"
        )
    sd_c, mask_c, orig_c = _step_meta(control, "flat_reassemble (control)")
    assert isinstance(sd_c, _SelectBase), (
        f"flat_reassemble: control stream_dtype must be Select, got {sd_c!r}"
    )
    assert sd_c.total_n == n, (
        f"flat_reassemble: control total_n={sd_c.total_n} != n={n}"
    )

    tile_r, tile_c = tile_shape
    ctrl_t = control.tensor
    flat_mh = ctrl_t.reshape(-1, n)
    total = flat_mh.shape[0]

    ptrs = [0] * n
    token_groups = []
    for t in range(total):
        group = []
        for i in range(n):
            if flat_mh[t, i] > 0:
                if ptrs[i] < inputs[i].tensor.shape[0]:
                    group.append(inputs[i].tensor[ptrs[i]])
                    ptrs[i] += 1
        if len(group) == 0:
            group.append(torch.zeros_like(inputs[0].tensor[0:1].squeeze(0)))
        token_groups.append(torch.stack(group, dim=0))

    output = torch.stack(token_groups, dim=0)
    # The DSL's pre-existing convention normalizes outer to 1 when not already.
    ctrl_stream_shape = ctrl_t.shape[:-1]
    prepended_one = False
    if ctrl_stream_shape[0] != 1:
        ctrl_stream_shape = (1,) + ctrl_stream_shape
        prepended_one = True
    n_active = output.shape[1]
    result = output.reshape(*ctrl_stream_shape, n_active, tile_r, tile_c)
    # Output dyn_mask: optionally-prepended-static + control's mask + new dyn.
    # Note control.dyn_mask covers Select's full stream rank (= ndim-1), which
    # equals len(ctrl_t.shape[:-1]) — so it lines up with the un-prepended
    # ctrl_stream_shape. If we prepended a (1,) we add a leading False slot.
    out_mask = (
        ((False,) if prepended_one else ())
        + mask_c
        + (True,)
    )
    out_orig = (
        ((None,) if prepended_one else ())
        + orig_c
        + ("flat_reassemble",)
    )
    return StepTensor(result, stream_dtype=sd0,
                      dyn_mask=out_mask, dyn_origins=out_orig)


def flatmap_filter_row_streamify(x, mask):
    """Filter input tile rows by `mask`, restreaming surviving rows as (1,c)
    tiles. The last stream dim is replaced with a new dynamic dim (count of
    surviving rows). Mirrors IR FlatmapFilterRowStreamify (ops.py:4133)."""
    sd_x, mask_x, orig_x = _step_meta(x, "flatmap_filter_row_streamify (x)")
    sd_m, _, _ = _step_meta(mask, "flatmap_filter_row_streamify (mask)")
    _assert_tile_kind(sd_x, "flatmap_filter_row_streamify (x)")
    _assert_tile_kind(sd_m, "flatmap_filter_row_streamify (mask)")
    assert len(mask_x) >= 1, (
        f"flatmap_filter_row_streamify: input must have >=1 stream dim, "
        f"got shape {tuple(x.tensor.shape)}"
    )
    t = x.tensor
    m = mask.tensor
    tile_r, tile_c = int(t.shape[-2]), int(t.shape[-1])
    flat_data = t.reshape(-1, tile_r, tile_c)
    flat_mask = m.reshape(-1, tile_r, 1)
    rows = []
    for i in range(flat_data.shape[0]):
        for r in range(tile_r):
            if flat_mask[i, r, 0] > 0:
                rows.append(flat_data[i, r:r + 1, :])
    assert len(rows) > 0, "flatmap_filter_row_streamify: no rows passed the mask"
    result = torch.cat(rows, dim=0)
    outer = t.shape[:-2][:-1]
    result = result.reshape(*outer, len(rows), 1, tile_c)
    # Output: same leading stream dims, replace last with dynamic, tile (1,c).
    return StepTensor(
        result,
        stream_dtype=Tile(sd_x.tile_dtype, (1, tile_c)),
        dyn_mask=mask_x[:-1] + (True,),
        dyn_origins=orig_x[:-1] + ("flatmap_filter_row_streamify",),
    )


def flatmap_counter(x):
    """Single-scalar in → arange stream out. Appends a new dynamic stream dim."""
    sd, mask, orig = _step_meta(x, "flatmap_counter")
    _assert_tile_kind(sd, "flatmap_counter")
    t = x.tensor
    flat = t.reshape(-1)
    assert flat.numel() == 1, (
        "flatmap_counter: only single-scalar input supported"
    )
    n = int(flat[0].item())
    stream_shape = t.shape[:-2]
    result = torch.arange(n, dtype=t.dtype).reshape(*stream_shape, n, 1, 1)
    # IR (ops.py:4233): stream = in_stream.shape + (DynDim,); tile (1,1).
    return StepTensor(
        result,
        stream_dtype=Tile(sd.tile_dtype, (1, 1)),
        dyn_mask=mask + (True,),
        dyn_origins=orig + ("flatmap_counter",),
    )


def promote(x, rank=1):
    """Insert a singleton stream dim at the `rank`-th-from-innermost slot.

    rank=0 → new dim becomes innermost stream dim (next to the tile pair).
    rank=k → new dim sits k slots before the innermost.
    """
    sd, mask, orig = _step_meta(x, "promote")
    _assert_tile_kind(sd, "promote")
    max_rank = x.tensor.ndim - 1
    assert rank >= 0, f"promote(rank={rank}): rank must be >= 0"
    assert rank <= max_rank, (
        f"promote(rank={rank}): tensor has {x.tensor.ndim} dims "
        f"({tuple(x.tensor.shape)}), max valid rank is {max_rank}."
    )
    result = x.tensor.unsqueeze(-(3 + rank))
    # Insert a static False slot at stream position (len(mask) - rank).
    n = len(mask)
    pos = n - rank
    new_mask = mask[:pos] + (False,) + mask[pos:]
    new_orig = orig[:pos] + (None,) + orig[pos:]
    return StepTensor(result, stream_dtype=sd,
                      dyn_mask=new_mask, dyn_origins=new_orig)


def promote_outer(x):
    """Prepend a singleton outer stream dim (mirrors IR's PromoteOuter)."""
    sd, mask, orig = _step_meta(x, "promote_outer")
    _assert_tile_kind(sd, "promote_outer")
    # STeP's PromoteOuter only operates on streams with rank >= 1, so the input
    # must be at least 3D (>=1 stream dim + 2 tile dims). Allowing a rank-0
    # stream here lets the DSL accept a "tile-only" tensor that the translator
    # cannot represent in STeP, surfacing only as a build_graph crash later.
    assert x.tensor.ndim >= 3, (
        f"promote_outer: input must have >=1 stream dim (>=3D total), "
        f"got shape {tuple(x.tensor.shape)}. If this is the output of a fully-"
        f"collapsing accum (e.g. accum_retile_row(rank=stream_rank)), keep "
        f"a stream dim — drop one rank from the accum or skip an upstream "
        f"flatten that consumed the leading singleton."
    )
    return StepTensor(x.tensor.unsqueeze(0), stream_dtype=sd,
                      dyn_mask=(False,) + mask,
                      dyn_origins=(None,) + orig)


def flatten(x, min_rank, max_rank):
    """Merge stream dims rank-range [min_rank, max_rank] (inner-indexed) into one.

    The merged dim is dynamic iff ANY of the absorbed dims is dynamic
    (mirrors IR Flatten: merged_dim becomes DynDim if any input is DynDim).
    """
    sd, mask, orig = _step_meta(x, "flatten")
    _assert_tile_kind(sd, "flatten")
    t = x.tensor
    tile_r, tile_c = int(t.shape[-2]), int(t.shape[-1])
    stream_shape = list(t.shape[:-2])
    n = len(stream_shape)
    assert n >= 1, (
        f"flatten: tensor {tuple(t.shape)} has no stream dims (need at least 3D)."
    )
    assert max_rank < n, (
        f"flatten(min_rank={min_rank}, max_rank={max_rank}): tensor {tuple(t.shape)} "
        f"has {n} stream dims, so max valid rank is {n - 1}."
    )
    assert 0 <= min_rank <= max_rank, (
        f"flatten: need 0 <= min_rank <= max_rank, got min_rank={min_rank}, max_rank={max_rank}."
    )
    min_idx = n - 1 - max_rank   # max_rank -> leftmost merged index
    max_idx = n - 1 - min_rank   # min_rank -> rightmost merged index
    merged = 1
    for i in range(min_idx, max_idx + 1):
        merged *= stream_shape[i]
    new_stream = stream_shape[:min_idx] + [merged] + stream_shape[max_idx + 1:]
    result = t.reshape(*new_stream, tile_r, tile_c)

    merged_dyn = any(mask[i] for i in range(min_idx, max_idx + 1))
    merged_origin = next(
        (orig[i] for i in range(min_idx, max_idx + 1) if mask[i]),
        None,
    ) if merged_dyn else None
    new_mask = mask[:min_idx] + (merged_dyn,) + mask[max_idx + 1:]
    new_orig = orig[:min_idx] + (merged_origin,) + orig[max_idx + 1:]
    return StepTensor(result, stream_dtype=sd,
                      dyn_mask=new_mask, dyn_origins=new_orig)


def expand_ref(x, ref, expand_rank):
    """Replace trailing `expand_rank` singleton stream dims with ref's matching
    dims. Leading stream dims (and their dyn-ness) must already match ref's."""
    sd_x, mask_x, orig_x = _step_meta(x, "expand_ref (input)")
    sd_r, mask_r, orig_r = _step_meta(ref, "expand_ref (ref)")
    _assert_tile_kind(sd_x, "expand_ref (input)")
    _assert_tile_kind(sd_r, "expand_ref (ref)")
    ref_stream = list(ref.tensor.shape[:-2])
    inp_stream = list(x.tensor.shape[:-2])
    assert expand_rank > 0, f"expand_rank must be > 0, got {expand_rank}"
    assert inp_stream[-expand_rank:] == [1] * expand_rank, (
        f"expand_ref: trailing {expand_rank} stream dims must be 1, got {inp_stream}"
    )
    assert inp_stream[:-expand_rank] == ref_stream[:-expand_rank], (
        f"expand_ref: leading stream dims must match: {inp_stream[:-expand_rank]} vs {ref_stream[:-expand_rank]}"
    )
    # Strict: leading dyn_mask must match too — otherwise we'd silently align
    # a static slot to a symbolic one.
    assert mask_x[:-expand_rank] == mask_r[:-expand_rank], (
        f"expand_ref: leading dyn_mask mismatch: "
        f"input {mask_x[:-expand_rank]} vs ref {mask_r[:-expand_rank]}"
    )
    expand_shape = ref_stream + list(x.tensor.shape[-2:])
    result = x.tensor.expand(expand_shape).contiguous()
    return StepTensor(result, stream_dtype=sd_x,
                      dyn_mask=mask_r, dyn_origins=orig_r)


def repeat_static(x, factor):
    """Insert a new static stream dim of `factor` just before the tile pair."""
    sd, mask, orig = _step_meta(x, "repeat_static")
    _assert_tile_kind(sd, "repeat_static")
    result = x.tensor.unsqueeze(-3)
    shape = list(result.shape)
    shape[-3] = factor
    result = result.expand(shape).contiguous()
    return StepTensor(result, stream_dtype=sd,
                      dyn_mask=mask + (False,),
                      dyn_origins=orig + (None,))


def reshape_stream(x, chunk_size, rank=0, add_outer_dim=False):
    """Split stream dim at position rank into (new_count, chunk_size).

    If the original dim is dynamic, the resulting `new_count` slot inherits
    that dynamism (chunk_size is a static int). Optional add_outer_dim
    prepends a static 1 to the stream shape.
    """
    sd, mask, orig = _step_meta(x, "reshape_stream")
    _assert_tile_kind(sd, "reshape_stream")
    t = x.tensor
    tile_r, tile_c = int(t.shape[-2]), int(t.shape[-1])
    stream_shape = list(t.shape[:-2])
    n = len(stream_shape)
    assert rank >= 0, f"reshape_stream(rank={rank}): rank must be >= 0"
    if add_outer_dim:
        assert n == 0, (
            f"reshape_stream(add_outer_dim=True): input stream rank must be 0 "
            f"(a single tile, x.ndim==2), got shape {tuple(t.shape)}."
        )
    else:
        assert n >= 1, (
            f"reshape_stream: tensor {tuple(t.shape)} has no stream dims."
        )
        assert rank < n, (
            f"reshape_stream(rank={rank}): tensor {tuple(t.shape)} has {n} stream dims, "
            f"max valid rank is {n - 1}."
        )

    rank_pos = n - 1 - rank
    D = stream_shape[rank_pos]
    assert D % chunk_size == 0 or rank == 0, (
        f"reshape_stream: shape[{rank_pos}]={D} not divisible by chunk_size={chunk_size}. "
        f"Automatic padding is only allowed when rank==0, got rank={rank}."
    )
    padded_D = ((D + chunk_size - 1) // chunk_size) * chunk_size

    if padded_D != D:
        pad_sizes = [0] * (2 * len(t.shape))
        pad_idx = 2 * (len(t.shape) - 1 - rank_pos)
        pad_sizes[pad_idx + 1] = padded_D - D
        t = F.pad(t, pad_sizes, value=0.0)
        stream_shape[rank_pos] = padded_D

    pre = stream_shape[:rank_pos]
    post = stream_shape[rank_pos + 1:]
    new_count = padded_D // chunk_size

    if add_outer_dim:
        new_shape = [1] + pre + [new_count, chunk_size] + post + [tile_r, tile_c]
        new_mask = (False,) * (1 + len(pre)) + (False, False) + tuple(mask[rank_pos + 1:])
        new_orig = (None,) * (1 + len(pre)) + (None, None) + tuple(orig[rank_pos + 1:])
    else:
        new_shape = pre + [new_count, chunk_size] + post + [tile_r, tile_c]
        # Split semantics: the new_count slot inherits dynamism (the produced
        # count depends on the original D); the chunk_size slot is always
        # static (caller supplied a Python int).
        was_dyn = mask[rank_pos]
        was_origin = orig[rank_pos]
        new_mask = (tuple(mask[:rank_pos])
                    + (was_dyn, False)
                    + tuple(mask[rank_pos + 1:]))
        new_orig = (tuple(orig[:rank_pos])
                    + (was_origin, None)
                    + tuple(orig[rank_pos + 1:]))

    result = t.reshape(new_shape)
    return StepTensor(result, stream_dtype=sd,
                      dyn_mask=new_mask, dyn_origins=new_orig)


def reshape_pad_stream(x, chunk_size, reshape_rank=0):
    return reshape_stream(x, chunk_size=chunk_size, rank=reshape_rank)


def retile_streamify(x, chunk, split_row=True):
    """Replace last stream dim D with D*num_chunks, shrinking the corresponding
    tile dim from (tile_r,tile_c) → (chunk, tile_c) [row] or (tile_r, chunk) [col].
    The last stream slot's dyn-ness is preserved (D * static_int stays dyn iff D was)."""
    sd, mask, orig = _step_meta(x, "retile_streamify")
    _assert_tile_kind(sd, "retile_streamify")
    assert len(mask) >= 1, (
        f"retile_streamify: input must have >=1 stream dim, got shape {tuple(x.tensor.shape)}"
    )
    t = x.tensor
    tile_r, tile_c = int(t.shape[-2]), int(t.shape[-1])
    stream_shape = t.shape[:-2]
    last = stream_shape[-1]
    pre = stream_shape[:-1]
    if split_row:
        actual_num_chunks = tile_r // chunk
        assert tile_r % chunk == 0, (
            f"retile_streamify: tile_r={tile_r} not divisible by chunk={chunk}"
        )
        reshaped = t.reshape(*pre, last, actual_num_chunks, chunk, tile_c)
        result = reshaped.reshape(*pre, last * actual_num_chunks, chunk, tile_c)
        new_tile = (chunk, tile_c)
    else:
        actual_num_chunks = tile_c // chunk
        assert tile_c % chunk == 0, (
            f"retile_streamify: tile_c={tile_c} not divisible by chunk={chunk}"
        )
        reshaped = t.reshape(*pre, last, tile_r, actual_num_chunks, chunk)
        perm = list(range(len(pre))) + [len(pre), len(pre) + 2, len(pre) + 1, len(pre) + 3]
        reshaped = reshaped.permute(perm)
        result = reshaped.reshape(*pre, last * actual_num_chunks, tile_r, chunk)
        new_tile = (tile_r, chunk)
    new_sd_cls = DynTile if isinstance(sd, DynTile) else Tile
    return StepTensor(result, stream_dtype=new_sd_cls(sd.tile_dtype, new_tile),
                      dyn_mask=mask, dyn_origins=orig)


def repeat_ref(x, ref):
    """Insert a new innermost stream dim whose size matches ref's last stream
    dim. The new slot inherits ref's last-stream-dim dyn-ness."""
    sd_x, mask_x, orig_x = _step_meta(x, "repeat_ref (input)")
    sd_r, mask_r, orig_r = _step_meta(ref, "repeat_ref (ref)")
    _assert_tile_kind(sd_x, "repeat_ref (input)")
    _assert_tile_kind(sd_r, "repeat_ref (ref)")
    ref_stream = list(ref.tensor.shape[:-2])
    inp_stream = list(x.tensor.shape[:-2])
    tile_dims = list(x.tensor.shape[-2:])
    assert inp_stream == ref_stream[:-1], (
        f"x stream shape must equal ref stream shape minus its trailing dim: "
        f"{inp_stream} vs {ref_stream[:-1]}"
    )
    assert mask_x == mask_r[:-1], (
        f"repeat_ref: input dyn_mask {mask_x} must equal ref's leading mask "
        f"{mask_r[:-1]}."
    )
    result = x.tensor.unsqueeze(-3)
    expand_shape = ref_stream + tile_dims
    result = result.expand(expand_shape).contiguous()
    return StepTensor(result, stream_dtype=sd_x,
                      dyn_mask=mask_x + (mask_r[-1],),
                      dyn_origins=orig_x + (orig_r[-1],))


def streamify(x, stride, out_shape_tiled):
    """Bufferize → re-stream with given stride pattern.

    Output stream_dtype is the buffer's inner Tile/DynTile, and the new stream
    shape appends `out_shape_tiled` (all static) onto the buffer's host stream.
    """
    sd, mask, orig = _step_meta(x, "streamify")
    assert isinstance(sd, Buffer), (
        f"streamify: input stream_dtype must be Buffer (call bufferize first), "
        f"got {sd!r}"
    )
    assert len(stride) == len(out_shape_tiled), (
        f"streamify: stride {tuple(stride)} and out_shape_tiled {tuple(out_shape_tiled)} "
        f"must have same length"
    )

    buffer_shape = sd.shape
    n_tiles = math.prod(buffer_shape)
    max_idx = sum((s - 1) * st for s, st in zip(out_shape_tiled, stride))
    assert max_idx < n_tiles, (
        f"streamify: stride {tuple(stride)} x out_shape_tiled {tuple(out_shape_tiled)} "
        f"exceeds buffer grid {buffer_shape} (max_idx={max_idx}, n_tiles={n_tiles})"
    )

    t = x.tensor
    buffer_rank = len(sd.shape)
    in_stream_rank = t.ndim - 2 - buffer_rank
    tile_r, tile_c = t.shape[-2], t.shape[-1]
    flat = t.reshape(*t.shape[:in_stream_rank], -1, tile_r, tile_c)

    ranges = [torch.arange(s) for s in out_shape_tiled]
    grids = torch.meshgrid(*ranges, indexing="ij")
    linear_idx = sum(g.long() * int(s) for g, s in zip(grids, stride))
    result = flat[..., linear_idx.long(), :, :]
    return StepTensor(
        result, stream_dtype=sd.buff_dtype,
        dyn_mask=mask + (False,) * len(out_shape_tiled),
        dyn_origins=orig + (None,) * len(out_shape_tiled),
    )


def bufferize(x, rank):
    """Absorb the trailing `rank` stream dims into a Buffer dtype's grid.

    The IR allows the FIRST buffer-grid dim to be dynamic but the rest must
    be static (see Buffer.__post_init__ in step_py/datatype.py). We enforce
    the same here: at most one absorbed dim may be dynamic, and only at the
    outermost position of the absorbed group.
    """
    sd, mask, orig = _step_meta(x, "bufferize")
    _assert_tile_kind(sd, "bufferize")
    assert rank >= 1, f"bufferize: rank must be >= 1, got {rank}"
    assert rank <= len(mask), (
        f"bufferize: rank {rank} exceeds input stream rank {len(mask)}"
    )
    absorb_start = len(mask) - rank
    for i in range(absorb_start, len(mask)):
        if mask[i] and i != absorb_start:
            raise AssertionError(
                f"bufferize: only the outermost absorbed stream dim may be "
                f"dynamic; dim {i} is dynamic (origin: {orig[i]}). Reorder "
                f"with flatten/promote so the dyn dim is the outermost in "
                f"the bufferized group."
            )
    buf_shape = tuple(int(x.tensor.shape[absorb_start + i]) for i in range(rank))
    return StepTensor(
        x.tensor, stream_dtype=Buffer(sd, buf_shape),
        dyn_mask=mask[:absorb_start], dyn_origins=orig[:absorb_start],
    )


def restream(x, stride, out_shape_tiled):
    """Composite: promote → retile (split both dims) → bufferize → streamify
    → accum_retile (col then row). Each primitive is typed, so dyn_mask flows
    through naturally."""
    sd, _, _ = _step_meta(x, "restream")
    _assert_tile_kind(sd, "restream")
    assert x.tensor.ndim >= 3, (
        f"restream: input must have >=1 stream dim + 2 tile dims, got ndim={x.tensor.ndim} "
        f"(shape={tuple(x.tensor.shape)})"
    )
    assert len(out_shape_tiled) >= 2, (
        f"restream: out_shape_tiled must end in (out_tile_r, out_tile_c) and "
        f"thus have length >= 2, got {tuple(out_shape_tiled)}"
    )
    assert len(stride) == len(out_shape_tiled), (
        f"restream: stride {tuple(stride)} and out_shape_tiled {tuple(out_shape_tiled)} "
        f"must have same length"
    )

    y = promote(x, rank=0)
    y = retile_streamify(y, chunk=1, split_row=True)
    y = retile_streamify(y, chunk=1, split_row=False)
    y = streamify(
        bufferize(y, rank=y.tensor.ndim - 2),
        stride=stride,
        out_shape_tiled=out_shape_tiled,
    )
    y = accum_retile_col(y, rank=1)
    y = accum_retile_row(y, rank=1)
    return y


def dyn_streamify(x, ref):
    """Bufferize → re-broadcast to ref's stream shape. Output dyn_mask copies ref's."""
    sd, _, _ = _step_meta(x, "dyn_streamify (input)")
    sd_r, mask_r, orig_r = _step_meta(ref, "dyn_streamify (ref)")
    assert isinstance(sd, Buffer), (
        f"dyn_streamify: input stream_dtype must be Buffer, got {sd!r}"
    )
    _assert_tile_kind(sd_r, "dyn_streamify (ref)")
    bufferized_rank = len(sd.shape)
    t = x.tensor
    ref_t = ref.tensor
    ref_stream_shape = ref_t.shape[:-2]
    buf_and_tile_dims = t.shape[-(2 + bufferized_rank):]
    expand_shape = list(ref_stream_shape) + list(buf_and_tile_dims)
    result = t.expand(expand_shape).contiguous()
    # Output: Tile/DynTile stream with stream shape = ref's stream shape. The
    # buffer grid dims become trailing stream dims on the output (all static
    # since Buffer.shape only allows static dims past the first).
    return StepTensor(
        result, stream_dtype=sd.buff_dtype,
        dyn_mask=mask_r + (False,) * bufferized_rank,
        dyn_origins=orig_r + (None,) * bufferized_rank,
    )

def broadcast(x, n):
    """n duplicate StepTensors. Each output's metadata mirrors `x`."""
    _step_meta(x, "broadcast")
    return [
        StepTensor(x.tensor.clone(), stream_dtype=x.stream_dtype,
                   dyn_mask=x.dyn_mask, dyn_origins=x.dyn_origins,
                   offsets=x.offsets)
        for _ in range(n)
    ]


def parallelize(x, n):
    """Cycle-level round-robin: consumer i gets tokens i, n+i, 2n+i, ...

    Output outermost dim inherits input's dyn-ness (slicing by stride keeps
    a dynamic outer if the input was dynamic — the quotient is still
    symbolic). Inner stream dims/tile are unchanged.
    """
    sd, mask, orig = _step_meta(x, "parallelize")
    _assert_tile_kind(sd, "parallelize")
    assert len(mask) >= 1, "parallelize: input must have >=1 stream dim"
    return [
        StepTensor(x.tensor[i::n].contiguous(),
                   stream_dtype=sd, dyn_mask=mask, dyn_origins=orig)
        for i in range(n)
    ]


def static_reassemble(inputs, target_stream_shape=None):
    """Inverse of parallelize: interleave tokens across inputs.

    output[k*n + i] = inputs[i][k]. All inputs must agree on stream_dtype
    AND dyn_mask (strict, like _assert_stream_match)."""
    n = len(inputs)
    assert n > 0, "static_reassemble: must have at least one input"
    sd0, mask0, orig0 = _step_meta(inputs[0], "static_reassemble (inputs[0])")
    _assert_tile_kind(sd0, "static_reassemble (inputs[0])")
    for i, p in enumerate(inputs):
        sd_i, mask_i, _ = _step_meta(p, f"static_reassemble (inputs[{i}])")
        assert sd_i == sd0, (
            f"static_reassemble: input {i} stream_dtype {sd_i!r} != {sd0!r}"
        )
        assert mask_i == mask0, (
            f"static_reassemble: input {i} dyn_mask {mask_i} != {mask0}"
        )

    S = inputs[0].tensor.shape[0]
    stacked = torch.stack([p.tensor for p in inputs], dim=1)  # (S, n, *rest)
    result = stacked.reshape(S * n, *inputs[0].tensor.shape[1:])
    if target_stream_shape is not None:
        tile_r, tile_c = result.shape[-2], result.shape[-1]
        target = tuple(target_stream_shape) + (tile_r, tile_c)
        if result.shape != target:
            result = result.reshape(target)
    return StepTensor(result, stream_dtype=sd0,
                      dyn_mask=mask0, dyn_origins=orig0)


def binary_map_accum(a, b, rank=1, weight_transposed=False, *, compute_bw=1):
    """matmul(a, b) followed by `rank` reductions along dim=-3.

    Output tile shape derived from matmul: (a_tile_r, b_tile_c) or
    (a_tile_r, b_tile_r) if transposed. Stream rank shrinks by `rank`.
    """
    assert compute_bw >= 1, f"binary_map_accum: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_map_accum")
    _assert_float(b, "binary_map_accum")
    _assert_stream_match(a, b, "binary_map_accum")
    assert rank > 0, f"binary_map_accum: rank must be > 0, got {rank}"
    sd_a, mask_a, orig_a = a.stream_dtype, a.dyn_mask, a.dyn_origins
    assert rank <= len(mask_a), (
        f"binary_map_accum: rank {rank} exceeds stream rank {len(mask_a)}"
    )
    if weight_transposed:
        mapped = torch.matmul(a.tensor, b.tensor.transpose(-2, -1))
        tile_shape = (sd_a.shape[0], b.stream_dtype.shape[0])
    else:
        mapped = torch.matmul(a.tensor, b.tensor)
        tile_shape = (sd_a.shape[0], b.stream_dtype.shape[1])
    for _ in range(rank):
        mapped = mapped.sum(dim=-3)
    return StepTensor(
        mapped,
        stream_dtype=Tile(sd_a.tile_dtype, tile_shape),
        dyn_mask=mask_a[:len(mask_a) - rank],
        dyn_origins=orig_a[:len(orig_a) - rank],
    )


def random_offchip_store(underlying, wdata, waddr, tile_row, tile_col, base_addr_byte=0, *, par_dispatch=1):
    assert par_dispatch >= 1, f"random_offchip_store: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], (
        f"random_offchip_store: underlying dtype must be float16 or float32, got {underlying.dtype}"
    )
    sd_w, mask_w, orig_w = _step_meta(wdata, "random_offchip_store (wdata)")
    sd_a, mask_a, _ = _step_meta(waddr, "random_offchip_store (waddr)")
    _assert_tile_kind(sd_w, "random_offchip_store (wdata)")
    _assert_tile_kind(sd_a, "random_offchip_store (waddr)")
    wdata_t = wdata.tensor
    waddr_t = waddr.tensor
    assert waddr_t.shape[-2:] == (1, 1), (
        f"random_offchip_store: waddr tile shape must be (1,1), got {tuple(waddr_t.shape[-2:])}"
    )
    assert wdata_t.shape[-2:] == (tile_row, tile_col), (
        f"random_offchip_store: wdata tile {tuple(wdata_t.shape[-2:])} != ({tile_row},{tile_col})"
    )
    assert wdata_t.shape[:-2] == waddr_t.shape[:-2], (
        f"random_offchip_store: wdata stream {tuple(wdata_t.shape[:-2])} != waddr stream {tuple(waddr_t.shape[:-2])}"
    )
    assert mask_w == mask_a, (
        f"random_offchip_store: wdata/waddr dyn_mask mismatch: {mask_w} vs {mask_a}"
    )
    R, C = underlying.shape[-2], underlying.shape[-1]
    assert R % tile_row == 0 and C % tile_col == 0, (
        f"random_offchip_store: ({R},{C}) not divisible by tile ({tile_row},{tile_col})"
    )
    # Mirror random_offchip_load's flat tile walk: batch dims (row-major) -> grid_r -> grid_c.
    # The Rust impl asserts 2D underlying, but the Python op layer (ops.py) builds tensor_shape_tiled
    # with leading batch dims (e.g. KV cache [batch, maxN, num_kv_heads, head_dim]), so we follow
    # the load-side semantics and accept N-D underlying.
    assert underlying.is_contiguous(), (
        "random_offchip_store: underlying must be contiguous so writes propagate through the view"
    )
    batch_shape = underlying.shape[:-2]
    B = 1
    for d in batch_shape:
        B *= d
    grid_r, grid_c = R // tile_row, C // tile_col
    tiles_per_batch = grid_r * grid_c
    flat_batch = underlying.view(B, R, C)
    addrs = waddr_t.reshape(-1).long().tolist()
    wflat = wdata_t.reshape(-1, tile_row, tile_col)
    for i, a in enumerate(addrs):
        b = a // tiles_per_batch
        within = a % tiles_per_batch
        gr, gc = within // grid_c, within % grid_c
        flat_batch[b, gr * tile_row:(gr + 1) * tile_row, gc * tile_col:(gc + 1) * tile_col] = wflat[i]
    stream_shape = waddr_t.shape[:-2]
    ack = torch.ones(*stream_shape, 1, 1, dtype=torch.float32)
    # IR: stream_dtype = Bool() (bare, not wrapped in Tile). DSL stores as a
    # (*stream, 1, 1) float tensor; we mirror IR's bare Bool here.
    return StepTensor(ack, stream_dtype=Bool(),
                      dyn_mask=mask_a, dyn_origins=orig_w)


def offchip_store(x, *, par_dispatch=1):
    """Sink. Returns a raw torch.Tensor (no stream contract on output) — the
    point of offchip_store is to write the stream to off-chip memory."""
    assert par_dispatch >= 1, f"offchip_store: par_dispatch must be >= 1, got {par_dispatch}"
    sd, _, _ = _step_meta(x, "offchip_store")
    _assert_tile_kind(sd, "offchip_store")
    # Note: the Rust IR also has a DynOffChipStore whose runtime body is byte-for-byte
    # identical to OffChipStore — they only differ at construction (DynOffChipStore reads
    # tensor_shape_tiled from a JSON file at startup instead of taking it as a Vec<usize>).
    # Since this DSL doesn't deal with on-disk shape files, dyn_offchip_store is omitted;
    # offchip_store covers the value-level behavior of both.
    t = x.tensor
    assert t.ndim >= 2, (
        f"offchip_store: input must be a tile stream (at least 2D for tile_r, tile_c), "
        f"got shape {tuple(t.shape)}."
    )
    tile_r, tile_c = t.shape[-2], t.shape[-1]

    # Strip leading 1 if present
    if t.shape[0] == 1:
        t = t[0]

    stream_shape = t.shape[:-2]

    if len(stream_shape) == 0:
        return t  # single tile

    if len(stream_shape) == 1:
        return t.reshape(stream_shape[0] * tile_r, tile_c)

    # 2-D+ stream_shape: last two stream_shape dims are row/col tile counts
    Tc = stream_shape[-1]
    ndim = len(t.shape)
    perm = list(range(ndim - 4)) + [ndim - 4, ndim - 2, ndim - 3, ndim - 1]
    t = t.permute(*perm).contiguous()

    total_rows = tile_r
    for d in stream_shape[:-1]:
        total_rows *= d
    return t.reshape(int(total_rows), int(Tc * tile_c))

DSL_FUNCTIONS = {
    # Source
    "offchip_load", "offchip_load_ref", "dyn_offchip_load",
    "select_gen", "metadata_gen", "expert_addr_gen",
    "cache_read_addr_gen", "random_offchip_load", "filter_last_tile",
    # Binary compute
    "binary_matmul", "binary_mul", "binary_add", "binary_div", "binary_is_equal",
    "binary_set_offset", "binary_row_wise_append", "binary_cache_write_addr_gen",
    # Fused compute
    "binary_map_accum",
    # Unary compute
    "unary_silu", "unary_square", "unary_exp", "unary_rsqrt", "unary_pow2",
    "unary_mul_imm", "unary_add_imm", "unary_sub_imm", "unary_rowwise_sum",
    "unary_mask_row", "unary_select_to_scalar", "unary_to_const_int",
    # Accumulation
    "accum_add", "accum_mul", "accum_max", "accum_retile_row", "accum_retile_col",
    "accum_signal_req_all_read",
    # Stream shape
    "promote", "promote_outer", "flatten", "reshape_stream", "reshape_pad_stream",
    "expand_ref", "repeat_ref", "repeat_static", "streamify", "dyn_streamify",
    "bufferize", "restream", "retile_streamify",
    # Multi-output
    "broadcast", "parallelize", "static_reassemble",
    # Routing
    "eager_merge", "flat_partition", "flat_reassemble",
    # Flatmap
    "flatmap_filter_row_streamify", "flatmap_counter",
    # Sink
    "offchip_store", "random_offchip_store",
}

# ---------------------------------------------------------------------------
# Shape trace (gated by env var STEP_DSL_TRACE=1).
# When enabled, every DSL_FUNCTIONS op prints input/output shapes to stdout
# so the orchestrator can capture and feed the trace back to the LLM.
# ---------------------------------------------------------------------------

import os as _step_dsl_os
import functools as _step_dsl_ft
import inspect as _step_dsl_isp

_STEP_DSL_TRACE = _step_dsl_os.environ.get("STEP_DSL_TRACE", "") == "1"


def _step_dsl_fmt(v):
    if isinstance(v, StepTensor):
        return f"{_step_dsl_fmt(v.tensor)}::{v.stream_dtype!r} dyn={v.dyn_mask}"
    if torch.is_tensor(v):
        s = tuple(v.shape)
        if len(s) >= 2:
            tile = f"tile({s[-2]},{s[-1]})"
            stream = s[:-2]
            return f"stream{tuple(stream)}×{tile}" if stream else tile
        return f"shape{s}"
    if isinstance(v, (list, tuple)) and v and all(
        isinstance(x, (torch.Tensor, StepTensor)) for x in v
    ):
        opener, closer = ("[", "]") if isinstance(v, list) else ("(", ")")
        return opener + ", ".join(_step_dsl_fmt(x) for x in v) + closer
    return repr(v)


def _step_dsl_log_shapes(_fn):
    if not _STEP_DSL_TRACE:
        return _fn
    _name = _fn.__name__
    _sig = _step_dsl_isp.signature(_fn)

    @_step_dsl_ft.wraps(_fn)
    def _wrapper(*args, **kwargs):
        bound = _sig.bind(*args, **kwargs)
        in_str = ", ".join(f"{k}={_step_dsl_fmt(v)}" for k, v in bound.arguments.items())
        print(f"[step_dsl] {_name} input shape(s): {in_str}", flush=True)
        result = _fn(*args, **kwargs)
        print(f"[step_dsl] {_name} output shape(s): {_step_dsl_fmt(result)}", flush=True)
        return result

    return _wrapper


if _STEP_DSL_TRACE:
    _step_dsl_g = globals()
    for _step_dsl_n in DSL_FUNCTIONS:
        assert _step_dsl_n in _step_dsl_g, (
            f"DSL_FUNCTIONS lists '{_step_dsl_n}' but it is not defined in step_dsl.py"
        )
        _step_dsl_g[_step_dsl_n] = _step_dsl_log_shapes(_step_dsl_g[_step_dsl_n])
    del _step_dsl_g, _step_dsl_n
