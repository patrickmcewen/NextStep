"""Memory-traffic-tracking companion to step_dsl.py.

Wraps every function in ``step_dsl.DSL_FUNCTIONS`` with a shim that records
per-op ``off_chip_traffic`` and ``on_chip_requirement`` (both ``count_fifos``
modes) — matching what ``step_tl/src/step_py/ops.py`` would compute on the
lowered IR. State lives on a ``Tracker`` that is created with
``step_dsl_memory.tracker()`` (a context manager) and read out via
``Tracker.records``, ``Tracker.total_off_chip``, and ``Tracker.total_on_chip``.

When no tracker is active, the shim is a transparent forwarder.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src import step_dsl

# ---------------------------------------------------------------------------
# Module config
# ---------------------------------------------------------------------------

MOCK_BF16: bool = True  # default; matches simulator scripts in step_tl/

# Re-export non-function symbols that DSL programs touch directly.
Buffered = step_dsl.Buffered

# ---------------------------------------------------------------------------
# Records and tracker
# ---------------------------------------------------------------------------


@dataclass
class OpRecord:
    op_name: str
    off_chip_bytes: int
    on_chip_bytes: int          # count_fifos=False
    on_chip_bytes_fifo: int     # count_fifos=True
    output_shape: tuple
    extra: dict = field(default_factory=dict)


class Tracker:
    def __init__(self, mock_bf16: bool = True):
        self.mock_bf16: bool = mock_bf16
        self.records: list[OpRecord] = []

    @property
    def total_off_chip(self) -> int:
        return sum(r.off_chip_bytes for r in self.records)

    @property
    def total_on_chip(self) -> int:
        return sum(r.on_chip_bytes for r in self.records)

    @property
    def total_on_chip_fifo(self) -> int:
        return sum(r.on_chip_bytes_fifo for r in self.records)


_ACTIVE: Tracker | None = None


class _TrackerScope:
    """Context manager body for ``tracker()``. Saves and restores ``_ACTIVE``
    so nested ``with`` blocks correctly stack and unwind, without using
    ``try/finally`` (the @contextmanager generator equivalent would).
    """

    def __init__(self, mock_bf16):
        self._tracker = Tracker(mock_bf16=mock_bf16)
        self._prev: Tracker | None = None

    def __enter__(self) -> "Tracker":
        global _ACTIVE
        self._prev = _ACTIVE
        _ACTIVE = self._tracker
        return self._tracker

    def __exit__(self, exc_type, exc, tb) -> None:
        global _ACTIVE
        _ACTIVE = self._prev


def tracker(mock_bf16: bool | None = None) -> _TrackerScope:
    return _TrackerScope(mock_bf16=MOCK_BF16 if mock_bf16 is None else mock_bf16)


# ---------------------------------------------------------------------------
# Helpers — n_byte and shape extraction
# ---------------------------------------------------------------------------


_DTYPE_TO_NAME = {
    torch.float32: "Float32",
    torch.float16: "Float16",
}


def _n_byte_for_name(name: str, mock_bf16: bool) -> int:
    """ops.py-style n_byte by IR datatype class name. Authoritative table."""
    if name == "Float16":
        return 2
    if name == "Float32":
        return 2 if mock_bf16 else 4
    if name == "Uint64":
        return 8
    raise AssertionError(f"_n_byte_for_name: unsupported dtype name {name!r}")


def _n_byte(torch_dtype, mock_bf16: bool) -> int:
    name = _DTYPE_TO_NAME.get(torch_dtype)
    assert name is not None, (
        f"_n_byte: unsupported torch dtype {torch_dtype}; "
        "DSL only allows float32/float16 inputs"
    )
    return _n_byte_for_name(name, mock_bf16)


def _tile_shape(t) -> tuple:
    assert t.ndim >= 2, f"_tile_shape: tensor must have >= 2 dims, got {tuple(t.shape)}"
    return (int(t.shape[-2]), int(t.shape[-1]))


def _tile_bytes(t, mock_bf16: bool) -> int:
    tr, tc = _tile_shape(t)
    return tr * tc * _n_byte(t.dtype, mock_bf16)


def _stream_total_elements(t) -> int:
    """prod(tensor.shape[:-2]); equals 1 for a bare tile."""
    assert t.ndim >= 2
    n = 1
    for d in t.shape[:-2]:
        n *= int(d)
    return n


def _stream_dtype_size_bytes(t, mock_bf16: bool) -> int:
    """Mirror ops.py: stream.stream_dtype.size_in_bytes() for a Tile dtype."""
    return _tile_bytes(t, mock_bf16)


# ---------------------------------------------------------------------------
# Per-op metric registry
# ---------------------------------------------------------------------------
#
# Each entry maps a DSL function name to a callable
#     (args, kwargs, output, mock_bf16) -> (off_chip, on_chip, on_chip_fifo, extra)
# Three byte values and an ``extra`` dict are returned (concrete ``int`` for
# each byte field; ``dict`` for ``extra``). ``extra`` carries op-specific
# facts useful for grouped reporting (tile shape, dtype, n_byte, …).
#
# Ops not yet present in this dict are forwarded transparently and do not
# record. This lets later tasks register families incrementally; the E2E
# parity test in the final task fails until every op family is keyed.

METRIC_FNS: dict[str, Callable[..., tuple[int, int, int, dict]]] = {}


# ---------------------------------------------------------------------------
# Wrap and export
# ---------------------------------------------------------------------------


def _make_wrapper(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if _ACTIVE is None:
            return out
        metric_fn = METRIC_FNS.get(name)
        if metric_fn is None:
            return out
        off, on_no_fifo, on_fifo, extra = metric_fn(args, kwargs, out, _ACTIVE.mock_bf16)
        _ACTIVE.records.append(OpRecord(
            op_name=name,
            off_chip_bytes=int(off),
            on_chip_bytes=int(on_no_fifo),
            on_chip_bytes_fifo=int(on_fifo),
            output_shape=_safe_output_shape(out),
            extra=extra,
        ))
        return out
    return wrapper


def _safe_output_shape(out: Any) -> tuple:
    """Return the shape tuple for whatever a DSL op returned.

    DSL ops return torch.Tensor, list[Tensor], Buffered, or _OffsetTile.
    Raises AssertionError for any other type so unexpected outputs surface
    immediately rather than silently producing an empty shape.
    """
    if isinstance(out, torch.Tensor):
        return tuple(out.shape)
    if isinstance(out, list):
        assert out and all(isinstance(x, torch.Tensor) for x in out), (
            f"_safe_output_shape: list output must be non-empty list of tensors, got {out!r}"
        )
        return tuple(out[0].shape)
    if isinstance(out, step_dsl.Buffered):
        return tuple(out.tensor.shape)
    if isinstance(out, step_dsl._OffsetTile):
        return tuple(out.data.shape)
    raise AssertionError(
        f"_safe_output_shape: unexpected output type {type(out).__name__}"
    )


# ---------------------------------------------------------------------------
# Per-op metric implementations — off-chip load/store family
# ---------------------------------------------------------------------------


def _metrics_offchip_load_like(args, kwargs, output, mock_bf16):
    """LinearOffChipLoad / LinearOffChipLoadRef / DynLinearOffChipLoad /
    RandomOffChipLoad share one formula:

      off_chip = stream.total_elements * tile_r * tile_c * n_byte
      on_chip  = tile_r * tile_c * n_byte   (both count_fifos modes)
    """
    tile_bytes = _tile_bytes(output, mock_bf16)
    n_stream = _stream_total_elements(output)
    extra = {"tile_shape": _tile_shape(output)}
    return n_stream * tile_bytes, tile_bytes, tile_bytes, extra


def _metrics_offchip_store(args, kwargs, output, mock_bf16):
    """OffChipStore: same formula as load, computed from the *input* stream
    (the data being written), not the flattened 2D output."""
    inp = args[0]
    tile_bytes = _tile_bytes(inp, mock_bf16)
    n_stream = _stream_total_elements(inp)
    extra = {"tile_shape": _tile_shape(inp)}
    return n_stream * tile_bytes, tile_bytes, tile_bytes, extra


def _metrics_random_offchip_store(args, kwargs, output, mock_bf16):
    """RandomOffChipStore: traffic over the ack stream * tile bytes.

    Tile bytes use the *underlying* tensor's dtype (positional arg 0), not
    the float32 'ones' ack tile that the DSL returns. Default buffer_depth
    is 1 in the DSL, so on-chip terms collapse to tile_bytes.
    """
    underlying = args[0]
    tile_row = args[3] if len(args) > 3 else kwargs["tile_row"]
    tile_col = args[4] if len(args) > 4 else kwargs["tile_col"]
    n_byte = _n_byte(underlying.dtype, mock_bf16)
    tile_bytes = int(tile_row) * int(tile_col) * n_byte
    n_stream = _stream_total_elements(output)  # ack stream
    extra = {"tile_shape": (int(tile_row), int(tile_col))}
    return n_stream * tile_bytes, tile_bytes, tile_bytes, extra


METRIC_FNS["offchip_load"]        = _metrics_offchip_load_like
METRIC_FNS["offchip_load_ref"]    = _metrics_offchip_load_like
METRIC_FNS["dyn_offchip_load"]    = _metrics_offchip_load_like
METRIC_FNS["random_offchip_load"] = _metrics_offchip_load_like
METRIC_FNS["offchip_store"]       = _metrics_offchip_store
METRIC_FNS["random_offchip_store"] = _metrics_random_offchip_store


# ---------------------------------------------------------------------------
# Per-op metric implementations — binary map family
# ---------------------------------------------------------------------------


def _tensor_of(x):
    """Underlying tensor for shape/dtype lookup. Handles Buffered and _OffsetTile."""
    if isinstance(x, step_dsl.Buffered):
        return x.tensor
    if isinstance(x, step_dsl._OffsetTile):
        return x.data
    return x


def _metrics_binary_map(args, kwargs, output, mock_bf16):
    """BinaryMap: off_chip = 0, on_chip(False) = 0, on_chip(True) = in1+in2+out."""
    in1 = _tensor_of(args[0])
    in2 = _tensor_of(args[1])
    out = _tensor_of(output)
    in1_b = _stream_dtype_size_bytes(in1, mock_bf16)
    in2_b = _stream_dtype_size_bytes(in2, mock_bf16)
    out_b = _stream_dtype_size_bytes(out, mock_bf16)
    return 0, 0, in1_b + in2_b + out_b, {"in1_bytes": in1_b, "in2_bytes": in2_b, "out_bytes": out_b}


for _bname in [
    "binary_matmul", "binary_mul", "binary_add", "binary_div", "binary_is_equal",
    "binary_set_offset", "binary_row_wise_append", "binary_cache_write_addr_gen",
]:
    METRIC_FNS[_bname] = _metrics_binary_map
del _bname


# ---------------------------------------------------------------------------
# Per-op metric implementations — unary map family
# ---------------------------------------------------------------------------


def _metrics_unary_map(args, kwargs, output, mock_bf16):
    """UnaryMap: off_chip = 0, on_chip(False) = 0, on_chip(True) = in_size + out_size."""
    inp = _tensor_of(args[0])
    out = _tensor_of(output)
    in_b = _stream_dtype_size_bytes(inp, mock_bf16)
    out_b = _stream_dtype_size_bytes(out, mock_bf16)
    return 0, 0, in_b + out_b, {"in_bytes": in_b, "out_bytes": out_b}


for _uname in [
    "unary_silu", "unary_square", "unary_exp", "unary_rsqrt", "unary_pow2",
    "unary_mul_imm", "unary_add_imm", "unary_sub_imm", "unary_rowwise_sum",
    "unary_mask_row", "unary_select_to_scalar", "unary_to_const_int",
]:
    METRIC_FNS[_uname] = _metrics_unary_map
del _uname


# ---------------------------------------------------------------------------
# Per-op metric implementations — accum family
# ---------------------------------------------------------------------------


def _metrics_accum_unary(args, kwargs, output, mock_bf16):
    """Single-input Accum family: on_chip(False) = out_size; on_chip(True) = out_size + in_size.

    Mirrors Accum.on_chip_requirement in ops.py:
      count_fifos=False -> accumulator_size
      count_fifos=True  -> accumulator_size + input_size
    """
    inp = _tensor_of(args[0])
    out = _tensor_of(output)
    out_b = _stream_dtype_size_bytes(out, mock_bf16)
    in_b = _stream_dtype_size_bytes(inp, mock_bf16)
    return 0, out_b, out_b + in_b, {"in_bytes": in_b, "out_bytes": out_b}


def _metrics_binary_map_accum(args, kwargs, output, mock_bf16):
    """BinaryMapAccum: on_chip(False) = out_size; on_chip(True) = in1_size * (n_inputs + 1).

    Mirrors BinaryMapAccum.on_chip_requirement in ops.py:
      count_fifos=False -> out stream_dtype size
      count_fifos=True  -> in_tile_size * (len(input_list) + 1) = in1_size * 3
    """
    in1 = _tensor_of(args[0])
    out = _tensor_of(output)
    out_b = _stream_dtype_size_bytes(out, mock_bf16)
    in1_b = _stream_dtype_size_bytes(in1, mock_bf16)
    return 0, out_b, 3 * in1_b, {"in_bytes": in1_b, "out_bytes": out_b}


for _aname in [
    "accum_add", "accum_mul", "accum_max",
    "accum_retile_row", "accum_retile_col",
    "accum_signal_req_all_read",
]:
    METRIC_FNS[_aname] = _metrics_accum_unary
del _aname

METRIC_FNS["binary_map_accum"] = _metrics_binary_map_accum


# ---------------------------------------------------------------------------
# Per-op metric implementations — stream-shape family
# ---------------------------------------------------------------------------
#
# Group A: on_chip(False)=0, on_chip(True)=2*stream_dtype_size
#   IR classes: Promote, Flatten, Reshape (reshape_stream), ReshapePadStream
#
# Group B: on_chip(False)=stream_dtype_size, on_chip(True)=2*stream_dtype_size
#   IR classes: PromoteOuter, ExpandRef, RepeatRef, RepeatStatic
#
# Group C: on_chip(False)=0, on_chip(True)=0
#   IR class: RetileStreamify
#
# Group D: on_chip(False)=buffer_size+tile_size, on_chip(True)=buffer_size+tile_size
#   IR classes: Bufferize, Streamify, DynStreamify
#   (count_fifos has no effect; same formula for both modes)
#
# All have off_chip_traffic=0 (assuming off_chip=False, the default).


def _metrics_stream_shape_zero_no_fifo(args, kwargs, output, mock_bf16):
    """Group A: Promote, Flatten, Reshape, ReshapePadStream.

    on_chip(False) = 0
    on_chip(True)  = 2 * stream_dtype_size_bytes(output)
    """
    out = _tensor_of(output)
    out_b = _stream_dtype_size_bytes(out, mock_bf16)
    return 0, 0, 2 * out_b, {"out_bytes": out_b}


def _metrics_stream_shape_one_no_fifo(args, kwargs, output, mock_bf16):
    """Group B: PromoteOuter, ExpandRef, RepeatRef, RepeatStatic.

    on_chip(False) = stream_dtype_size_bytes(output)
    on_chip(True)  = 2 * stream_dtype_size_bytes(output)
    """
    out = _tensor_of(output)
    out_b = _stream_dtype_size_bytes(out, mock_bf16)
    return 0, out_b, 2 * out_b, {"out_bytes": out_b}


def _metrics_retile_streamify(args, kwargs, output, mock_bf16):
    """RetileStreamify: on_chip_requirement always returns 0 (both modes)."""
    return 0, 0, 0, {}


def _metrics_bufferize(args, kwargs, output, mock_bf16):
    """Bufferize: on_chip = buffer_size + tile_size (identical for both count_fifos modes).

    IR formula:
      tile_size   = in_stream.stream_dtype.size_in_bytes()  (input tile bytes)
      buffer_size = output_buffer.stream_dtype.size_in_bytes()
                  = prod(buffer_grid) * tile_bytes

    DSL: bufferize(x, rank) returns Buffered(x, buffer_rank=rank).
      input tensor x has shape (*stream_dims, tile_r, tile_c)
      buffer_grid = x.shape[-2-rank : -2]  (the rank dims before the tile dims)
    """
    x = _tensor_of(args[0])          # input tensor
    buf_out = output                  # Buffered
    assert isinstance(buf_out, step_dsl.Buffered)
    tile_b = _tile_bytes(x, mock_bf16)
    buffer_grid = buf_out.buffer_shape  # e.g. (2,) for rank=1
    n_buffer_tiles = 1
    for d in buffer_grid:
        n_buffer_tiles *= int(d)
    buffer_b = n_buffer_tiles * tile_b
    total = buffer_b + tile_b
    return 0, total, total, {"tile_bytes": tile_b, "buffer_bytes": buffer_b}


def _metrics_streamify_family(args, kwargs, output, mock_bf16):
    """Streamify / DynStreamify: on_chip = buffer_size + tile_size (same for both modes).

    IR: input is Buffer, output stream_dtype is Tile.
      tile_size   = output.stream_dtype.size_in_bytes()  (same tile as buffer's buff_dtype)
      buffer_size = in_buffer.size_in_bytes()
                  = prod(buffer.shape) * tile_bytes

    DSL: streamify(buf, stride, out_shape_tiled) or dyn_streamify(buf, ...) where buf is Buffered.
    """
    buf = args[0]                     # Buffered
    assert isinstance(buf, step_dsl.Buffered)
    tile_b = _tile_bytes(buf.tensor, mock_bf16)
    buffer_grid = buf.buffer_shape
    n_buffer_tiles = 1
    for d in buffer_grid:
        n_buffer_tiles *= int(d)
    buffer_b = n_buffer_tiles * tile_b
    total = buffer_b + tile_b
    return 0, total, total, {"tile_bytes": tile_b, "buffer_bytes": buffer_b}


# Group A
for _sname in ["promote", "flatten", "reshape_stream", "reshape_pad_stream"]:
    METRIC_FNS[_sname] = _metrics_stream_shape_zero_no_fifo
del _sname

# Group B
for _sname in ["promote_outer", "expand_ref", "repeat_ref", "repeat_static"]:
    METRIC_FNS[_sname] = _metrics_stream_shape_one_no_fifo
del _sname

# Group C
METRIC_FNS["retile_streamify"] = _metrics_retile_streamify

# Group D
METRIC_FNS["bufferize"]    = _metrics_bufferize
METRIC_FNS["streamify"]    = _metrics_streamify_family
METRIC_FNS["dyn_streamify"] = _metrics_streamify_family


# ---------------------------------------------------------------------------
# Per-op metric implementations — routing / multi-output family
# ---------------------------------------------------------------------------
#
# IR formulas (count_fifos=False always returns 0 for all these ops):
#   Broadcast:        on_chip(T) = 0                   (no FIFOs needed)
#   Parallelize:      on_chip(T) = in_tile * (n+1)
#   StaticReassemble: on_chip(T) = in_tile * (n_inputs+1)
#   FlatPartition:    on_chip(T) = in_tile * (n+1)
#   FlatReassemble:   on_chip(T) = in_tile * (n_inputs+1)  (write_back_mu=False)
#   EagerMerge:       on_chip(T) = (in_tile + sel_dtype_size) * (n_inputs+1)
#                     where sel_dtype_size = MultiHot(total_n=n_inputs).size_in_bytes()
#                                         = n_inputs  (one byte per hot bit)
#   SelectGen:        always 0


def _metrics_broadcast(args, kwargs, output, mock_bf16):
    """Broadcast: off_chip=0, on_chip(F)=0, on_chip(T)=0."""
    return 0, 0, 0, {}


def _metrics_parallelize(args, kwargs, output, mock_bf16):
    """Parallelize: on_chip(T) = in_tile_size * (num_consumers + 1)."""
    x = _tensor_of(args[0])
    n = int(args[1]) if len(args) > 1 else int(kwargs["n"])
    in_tile_b = _stream_dtype_size_bytes(x, mock_bf16)
    return 0, 0, in_tile_b * (n + 1), {"in_bytes": in_tile_b, "n": n}


def _metrics_static_reassemble(args, kwargs, output, mock_bf16):
    """StaticReassemble: on_chip(T) = in_tile_size * (n_inputs + 1)."""
    inputs = args[0]
    assert isinstance(inputs, list) and inputs
    n = len(inputs)
    in_tile_b = _stream_dtype_size_bytes(_tensor_of(inputs[0]), mock_bf16)
    return 0, 0, in_tile_b * (n + 1), {"in_bytes": in_tile_b, "n_inputs": n}


def _metrics_flat_partition(args, kwargs, output, mock_bf16):
    """FlatPartition: on_chip(T) = in_tile_size * (num_consumers + 1)."""
    x = _tensor_of(args[0])
    n = int(args[2]) if len(args) > 2 else int(kwargs["n"])
    in_tile_b = _stream_dtype_size_bytes(x, mock_bf16)
    return 0, 0, in_tile_b * (n + 1), {"in_bytes": in_tile_b, "n": n}


def _metrics_flat_reassemble(args, kwargs, output, mock_bf16):
    """FlatReassemble (write_back_mu=False): on_chip(T) = in_tile_size * (n_inputs + 1)."""
    inputs = args[0]
    assert isinstance(inputs, list) and inputs
    n = len(inputs)
    in_tile_b = _stream_dtype_size_bytes(_tensor_of(inputs[0]), mock_bf16)
    return 0, 0, in_tile_b * (n + 1), {"in_bytes": in_tile_b, "n_inputs": n}


def _metrics_eager_merge(args, kwargs, output, mock_bf16):
    """EagerMerge: on_chip(T) = (in_tile_size + n_inputs) * (n_inputs + 1).

    sel_stream_dtype_size = MultiHot(total_n=n_inputs).size_in_bytes() = n_inputs bytes.
    """
    inputs = args[0]
    assert isinstance(inputs, list) and inputs
    n = len(inputs)
    in_tile_b = _stream_dtype_size_bytes(_tensor_of(inputs[0]), mock_bf16)
    sel_size = n  # MultiHot(total_n=n).size_in_bytes() == n
    return 0, 0, (in_tile_b + sel_size) * (n + 1), {"in_bytes": in_tile_b, "n_inputs": n}


def _metrics_select_gen(args, kwargs, output, mock_bf16):
    """SelectGen: always returns 0 for all metrics."""
    return 0, 0, 0, {}


METRIC_FNS["broadcast"]         = _metrics_broadcast
METRIC_FNS["parallelize"]       = _metrics_parallelize
METRIC_FNS["static_reassemble"] = _metrics_static_reassemble
METRIC_FNS["flat_partition"]    = _metrics_flat_partition
METRIC_FNS["flat_reassemble"]   = _metrics_flat_reassemble
METRIC_FNS["eager_merge"]       = _metrics_eager_merge
METRIC_FNS["select_gen"]        = _metrics_select_gen


# ---------------------------------------------------------------------------
# Per-op metric implementations — source-control family
# ---------------------------------------------------------------------------
#
# IR classes: MetadataGen, SelectGen (already registered above), CacheReadAddrGen,
#   FilterLastTile, ExpertAddrGen, FlatmapFilterRowStreamify, FlatmapCounter.
#
# All return on_chip_requirement = 0 for both count_fifos modes (verified in
# utility_ops.py and ops.py). off_chip_traffic is also 0 for all of them.


def _metrics_source_control_zero(args, kwargs, output, mock_bf16):
    """Metric for source-control ops whose IR on_chip_requirement is always 0.

    Covers: MetadataGen, CacheReadAddrGen, FilterLastTile, ExpertAddrGen,
            FlatmapFilterRowStreamify, FlatmapCounter.
    """
    return 0, 0, 0, {}


for _scname in [
    "metadata_gen",
    "expert_addr_gen",
    "cache_read_addr_gen",
    "filter_last_tile",
    "flatmap_filter_row_streamify",
    "flatmap_counter",
]:
    METRIC_FNS[_scname] = _metrics_source_control_zero
del _scname


for _name in step_dsl.DSL_FUNCTIONS:
    assert hasattr(step_dsl, _name), f"step_dsl missing function listed in DSL_FUNCTIONS: {_name}"
    globals()[_name] = _make_wrapper(_name, getattr(step_dsl, _name))

del _name
