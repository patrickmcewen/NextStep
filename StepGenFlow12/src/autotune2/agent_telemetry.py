"""Append-only telemetry log for the autotune2 LLM-backed agents.

Every per-variant decision the ``AgentManager`` makes and every end-of-run
pick the ``FinalPickAgent`` makes is recorded here as one JSONL line.
The point is post-hoc audit: catching prompt drift, identifying which
calls take longest, and seeing how often the fallback path fires —
without having to re-run the experiment to reproduce the agent's
behavior.

This file is the agent-decision analog of ``calibration.py``. They are
intentionally separate stores: ``calibration.jsonl`` records only the
``(analytical, rust)`` measurement pair and is written only when a rust
call actually fires; this store records every *decision* (including
analytical-only and budget-exhausted skips and fallbacks), even when no
rust call happens. The two stores reference the same content-addressed
composed-source sidecars via ``composed_source_hash`` — duplicating the
hash means you can join the two files without re-reading source bodies.

Storage format
--------------
JSONL, one record per line, append-only. Each ``append`` writes a
single ``json.dumps(...) + "\\n"`` in one ``write()`` syscall — POSIX
guarantees atomicity below ``PIPE_BUF`` (typically 4096 bytes), so
concurrent appends from the search fan-out stay well-formed. The
4096-byte ceiling is asserted at append time. Composed sources are
NOT stored inline; only the sha256 hash is recorded (the source body
already lives in the calibration ``sources_dir`` whenever a rust call
followed). Reason strings can be long, but the sim-decision /
final-pick agent prompts cap them to ~25 words, so the line stays
well under the limit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class AgentDecisionRecord:
    """One agent decision, ready to serialize as a JSONL line.

    ``stage`` distinguishes the two call sites that write to this store:
    ``"sim_decision"`` (in-loop ``AgentManager._decide``) and
    ``"final_pick"`` (end-of-run ``final_pick_agent``). Both share the
    same schema so a single grep over the file covers both surfaces;
    fields irrelevant to one stage default to empty / -1 sentinels.

    ``decision`` covers the union of states both stages can land in:
      - ``"rust"`` / ``"analytical"``: a successful sim-decision pick.
      - ``"analytical_budget_exhausted"``: budget hit zero so the
        sim-decision call was skipped entirely (no curation, no
        decision agent).
      - ``"fallback"``: an exception inside the agent path (RPC fail,
        parse fail, candidate-fetch crash) — the outer ``except``
        caught and degraded.  ``reason`` carries the exception summary.
      - ``"pareto_short_circuit"``: ``final_pick_agent`` saw a
        one-entry Pareto and promoted directly without an agent
        round-trip.
      - ``"picked"``: ``final_pick_agent`` ran successfully and picked
        ``picked_variant_index``.

    Timing
    ------
    ``curation_dur_ms`` and ``decision_dur_ms`` are reported separately
    because the two calls usually hit different model profiles (cheaper
    curation, primary decision) and we want to spot regressions on
    either independently. ``-1.0`` means "call did not happen" (cold-
    start curation skip, or a fallback that aborted before that point).
    """

    stage: str  # "sim_decision" | "final_pick"
    run_id: str
    kernel: str
    preset: str
    hw_config_hash: str
    node_path: str
    is_root: bool
    timestamp: str  # ISO 8601 UTC

    composed_source_hash: str = ""  # sha256 of composed source (matches calibration sources_dir)
    variant_kind: str = ""  # "baseline" | "variant"; sim_decision only
    attempt_index: int = -1
    turn_index: int = -1

    decision: str = ""
    reason: str = ""
    curated_record_ids: list[str] = field(default_factory=list)
    num_candidates_available: int = 0

    curation_dur_ms: float = -1.0
    decision_dur_ms: float = -1.0

    # final_pick only — index into the Pareto-front list the agent
    # saw. -1 for sim_decision or for final_pick fallbacks that did
    # not produce an agent index.
    picked_variant_index: int = -1
    num_pareto_entries: int = 0


class AgentDecisionStore:
    """Append-only JSONL store of ``AgentDecisionRecord``.

    Same atomic-append discipline as ``CalibrationStore``: each
    ``append`` writes one ``json.dumps(...) + "\\n"`` in one
    ``write()`` so concurrent writers from the search fan-out don't
    interleave lines. The 4096-byte ceiling is asserted; if a record
    ever exceeds it, the offending field (almost certainly ``reason``)
    needs to be truncated upstream rather than this writer silently
    chunking.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: AgentDecisionRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), sort_keys=True) + "\n"
        assert len(line.encode("utf-8")) < 4096, (
            f"AgentDecisionStore.append: serialized record is {len(line)} "
            f"bytes, exceeds the 4096-byte atomic-append limit. Concurrent "
            f"appends would interleave. Trim 'reason' or 'curated_record_ids' "
            f"upstream."
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)

    def iter_records(self) -> Iterator[AgentDecisionRecord]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield AgentDecisionRecord(**data)
