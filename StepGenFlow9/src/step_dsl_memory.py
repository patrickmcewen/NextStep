"""Memory-traffic-tracking companion to step_dsl.py.

Wraps every function in ``step_dsl.DSL_FUNCTIONS`` with a shim that records
per-op ``off_chip_traffic`` and ``on_chip_requirement`` (both ``count_fifos``
modes) — matching what ``step_tl/src/step_py/ops.py`` would compute on the
lowered IR. State lives on a ``Tracker`` that is created with
``step_dsl_memory.tracker()`` (a context manager) and read out via
``Tracker.records``, ``Tracker.total_off_chip``, ``Tracker.total_on_chip``,
and ``Tracker.report()``.

When no tracker is active, the shim is a transparent forwarder.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

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


@contextmanager
def tracker(mock_bf16: bool | None = None):
    global _ACTIVE
    prev = _ACTIVE
    _ACTIVE = Tracker(mock_bf16=MOCK_BF16 if mock_bf16 is None else mock_bf16)
    try:
        yield _ACTIVE
    finally:
        _ACTIVE = prev


# ---------------------------------------------------------------------------
# Per-op metric registry
# ---------------------------------------------------------------------------
#
# Each entry maps a DSL function name to a callable
#     (args, kwargs, output, mock_bf16) -> (off_chip, on_chip, on_chip_fifo, extra)
# All four byte values are concrete ``int``. ``extra`` is a dict of op-specific
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
    """Return a (best-effort) shape tuple for whatever a DSL op returned.

    DSL ops return torch.Tensor, list[Tensor], Buffered, or _OffsetTile.
    """
    import torch
    if isinstance(out, torch.Tensor):
        return tuple(out.shape)
    if isinstance(out, list) and out and all(isinstance(x, torch.Tensor) for x in out):
        return tuple(out[0].shape)
    if hasattr(out, "tensor"):  # Buffered
        return tuple(out.tensor.shape)
    if hasattr(out, "data"):    # _OffsetTile
        return tuple(out.data.shape)
    return ()


for _name in step_dsl.DSL_FUNCTIONS:
    assert hasattr(step_dsl, _name), f"step_dsl missing function listed in DSL_FUNCTIONS: {_name}"
    globals()[_name] = _make_wrapper(_name, getattr(step_dsl, _name))

del _name
