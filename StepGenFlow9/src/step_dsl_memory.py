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


for _name in step_dsl.DSL_FUNCTIONS:
    assert hasattr(step_dsl, _name), f"step_dsl missing function listed in DSL_FUNCTIONS: {_name}"
    globals()[_name] = _make_wrapper(_name, getattr(step_dsl, _name))

del _name
