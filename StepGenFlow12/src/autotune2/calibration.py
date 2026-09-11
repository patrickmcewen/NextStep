"""Cross-run, cross-kernel calibration store for the simulation manager.

Every time the rust simulator runs, the analytical estimate for the
same composed source is also available (the analytical run is always
cheap, and rust modes still need it for on-chip bytes). Each such pair
is appended here as one ``CalibrationRecord``. Future PRs use the
accumulated records as evidence for the LLM decision agents — the
in-loop ``SimulationManager`` consults it to decide whether to spend
rust time on a variant, and the ``FinalPickAgent`` consults it when
choosing the single variant to ground-truth-evaluate.

Storage format
--------------
JSONL, one record per line, append-only. Each record is a single
``json.dumps(...) + "\\n"`` write that fits under POSIX's per-call
atomicity guarantee (well under ``PIPE_BUF``, typically 4096 bytes),
so multiple processes appending concurrently to the same file do not
interleave lines. The composed source itself is NOT stored inline —
records hold a path reference to ``composed_source.py`` inside the
producing run's checkpoint, both because composed sources frequently
exceed the atomic-write limit and because the curation agent reads
sources lazily and only for the small number of records it actually
selects.

Filtering
---------
Records carry an ``hw_config_hash`` so consumers can ignore evidence
gathered against a different hardware config (the analytical-vs-rust
relationship is hw-config-dependent). The store does not filter on
its own; callers iterate records and filter as they see fit.

For PR1 nothing writes to this store — only the type + I/O surface
exist so subsequent PRs can wire the rust-touching managers without
churning callers again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class CalibrationRecord:
    """One (analytical, rust) measurement pair for a single composed
    source.

    The composed source itself is referenced by ``composed_source_path``
    — typically the ``composed_source.py`` written by the producing
    turn's checkpoint dir (see ``search._write_turn_artifacts``). Keeping
    sources out-of-line keeps each JSONL line under the POSIX atomic-
    write limit and keeps the calibration file scannable.
    """

    node_path: str
    is_root: bool
    kernel: str
    preset: str
    composed_source_path: str
    analytical_cycles: int
    analytical_on_chip: int
    rust_cycles: int
    rust_dur_ms: float
    hw_config_hash: str
    compute_bw: int
    timestamp: str  # ISO 8601
    run_id: str  # identifies the autotune2 run that produced this record
    # Signed percentage error of the analytical estimate vs the rust
    # ground truth: 100 * (analytical_cycles - rust_cycles) / rust_cycles.
    # Negative means analytical underestimated. Stored on disk (not a
    # property) so offline analysis tools can grep/sort the JSONL
    # directly. The writer computes this from cycle counts at append
    # time. Units are *percent*, so e.g. -78.79 means analytical was
    # 78.79% below rust.
    error_pct: float

    @property
    def delta_pct(self) -> float:
        """Signed fractional skew (rust - analytical) / analytical.

        Positive means rust reports more cycles than the analytical
        model. Asserts analytical_cycles > 0 — a zero-cycle analytical
        estimate is a scorer bug, not something to silently divide by.
        """
        assert self.analytical_cycles > 0, (
            f"CalibrationRecord.delta_pct: analytical_cycles must be > 0, "
            f"got {self.analytical_cycles!r} for node {self.node_path!r}"
        )
        return (self.rust_cycles - self.analytical_cycles) / self.analytical_cycles


def hw_config_hash(hw_config: dict) -> str:
    """Stable short hash of a hardware config dict.

    Used as ``CalibrationRecord.hw_config_hash`` so consumers can
    ignore records gathered against a different hw config (the
    analytical-vs-rust relationship is hw-dependent — a memory-bound
    regime at one hw config may not hold at another).
    """
    blob = json.dumps(hw_config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class CalibrationStore:
    """Append-only JSONL store of ``CalibrationRecord``.

    Concurrent appends from the search fan-out (and from concurrent
    autotune2 runs sharing the same store) are safe because each call
    writes a single short line in a single ``write()`` syscall —
    POSIX guarantees atomicity for writes under ``PIPE_BUF`` (>= 4096
    bytes). Records keep the composed source out-of-line precisely to
    stay under that limit.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: CalibrationRecord) -> None:
        """Append one record. Creates parent dirs and file as needed.

        Each call writes exactly one line (``json.dumps(...) +
        "\\n"``) in a single ``write()`` so the on-disk file is always
        well-formed JSONL even under concurrent appends.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), sort_keys=True) + "\n"
        assert len(line.encode("utf-8")) < 4096, (
            f"CalibrationStore.append: serialized record is {len(line)} bytes, "
            f"exceeds the 4096-byte atomic-append limit. Concurrent appends "
            f"would interleave. Trim a field (composed_source_path is the "
            f"likely culprit) or switch to a locked writer."
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    def iter_records(
        self, *, hw_config_hash: str | None = None,
    ) -> Iterator[CalibrationRecord]:
        """Yield every record in the file in insertion order.

        When ``hw_config_hash`` is set, records with a different hash
        are filtered out — used by consumers that want only evidence
        gathered against the current hw config.
        """
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                record = CalibrationRecord(**data)
                if hw_config_hash is not None and record.hw_config_hash != hw_config_hash:
                    continue
                yield record


class CalibrationOverlayStore:
    """Read seed calibration records plus run-local records, append locally.

    This lets autotune2 use a shared StepDB calibration corpus as prior
    evidence without mutating it during every experiment. New rust
    measurements append only to ``append_store``; ``iter_records`` sees
    seed stores first, then records accumulated in the current run.
    """

    def __init__(
        self,
        *,
        seed_stores: list[CalibrationStore],
        append_store: CalibrationStore,
    ) -> None:
        self._seed_stores = list(seed_stores)
        self._append_store = append_store

    @property
    def path(self) -> Path:
        return self._append_store.path

    def append(self, record: CalibrationRecord) -> None:
        self._append_store.append(record)

    def iter_records(
        self, *, hw_config_hash: str | None = None,
    ) -> Iterator[CalibrationRecord]:
        for store in [*self._seed_stores, self._append_store]:
            yield from store.iter_records(hw_config_hash=hw_config_hash)
