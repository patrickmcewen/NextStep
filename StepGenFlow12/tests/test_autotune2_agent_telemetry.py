"""Unit tests for the autotune2 agent-decision telemetry store (PR6).

Covers the JSONL store behaviour in isolation:

  - ``AgentDecisionStore.append`` writes one well-formed JSON line per
    record, creates parent dirs lazily, and survives concurrent appends
    (each call is one ``write()`` so POSIX atomicity holds).
  - ``iter_records`` round-trips appended records.
  - The 4096-byte atomic-append ceiling is asserted at append time.

End-to-end wiring of the store through ``AgentManager`` and
``final_pick_agent`` is covered by ``test_autotune2_sim_manager.py``
and ``test_autotune2_final_pick.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.autotune2.agent_telemetry import (
    AgentDecisionRecord,
    AgentDecisionStore,
)


def _make_record(**overrides) -> AgentDecisionRecord:
    base = dict(
        stage="sim_decision",
        run_id="r1",
        kernel="gemm",
        preset="small",
        hw_config_hash="abcd1234abcd1234",
        node_path="root",
        is_root=True,
        timestamp="2026-05-21T00:00:00+00:00",
        composed_source_hash="0" * 64,
        variant_kind="variant",
        attempt_index=0,
        turn_index=0,
        decision="rust",
        reason="evidence supports rust",
        curated_record_ids=["rec00", "rec01"],
        num_candidates_available=5,
        curation_dur_ms=123.4,
        decision_dur_ms=456.7,
    )
    base.update(overrides)
    return AgentDecisionRecord(**base)


def test_append_writes_one_jsonl_line(tmp_path: Path):
    store = AgentDecisionStore(path=tmp_path / "agent_decisions.jsonl")
    store.append(_make_record())
    raw = (tmp_path / "agent_decisions.jsonl").read_text()
    assert raw.endswith("\n")
    lines = [l for l in raw.splitlines() if l]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["decision"] == "rust"
    assert parsed["curated_record_ids"] == ["rec00", "rec01"]
    assert parsed["stage"] == "sim_decision"


def test_append_creates_parent_dir(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "agent_decisions.jsonl"
    store = AgentDecisionStore(path=nested)
    store.append(_make_record())
    assert nested.exists()


def test_iter_records_round_trips(tmp_path: Path):
    store = AgentDecisionStore(path=tmp_path / "td.jsonl")
    store.append(_make_record(decision="rust"))
    store.append(_make_record(decision="analytical", reason="cheap enough"))
    store.append(_make_record(
        stage="final_pick", decision="picked",
        picked_variant_index=2, num_pareto_entries=4,
    ))
    rows = list(store.iter_records())
    assert len(rows) == 3
    assert [r.decision for r in rows] == ["rust", "analytical", "picked"]
    assert rows[2].stage == "final_pick"
    assert rows[2].picked_variant_index == 2
    assert rows[2].num_pareto_entries == 4


def test_iter_records_empty_when_file_missing(tmp_path: Path):
    store = AgentDecisionStore(path=tmp_path / "nope.jsonl")
    assert list(store.iter_records()) == []


def test_append_rejects_oversized_record(tmp_path: Path):
    """Records whose serialized line crosses the 4096-byte atomic-append
    limit must be rejected at append time — silent acceptance would let
    concurrent writers interleave lines.
    """
    store = AgentDecisionStore(path=tmp_path / "big.jsonl")
    huge_reason = "X" * 5000
    with pytest.raises(AssertionError, match="atomic-append limit"):
        store.append(_make_record(reason=huge_reason))


def test_concurrent_appends_do_not_interleave(tmp_path: Path):
    """Many concurrent appends from coroutines must produce N well-formed
    JSONL lines — the writer's ``open('a')+write(line)`` per call relies
    on POSIX < PIPE_BUF atomicity, so a regression here (e.g. switching
    to two write calls per record) would surface as torn lines.
    """
    store = AgentDecisionStore(path=tmp_path / "race.jsonl")

    async def writer(i: int):
        store.append(_make_record(reason=f"r{i}"))

    async def run_all():
        await asyncio.gather(*(writer(i) for i in range(64)))

    asyncio.run(run_all())

    rows = list(store.iter_records())
    assert len(rows) == 64
    reasons = sorted(r.reason for r in rows)
    assert reasons == sorted(f"r{i}" for i in range(64))
