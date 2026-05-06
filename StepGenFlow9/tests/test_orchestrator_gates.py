"""Unit + integration tests for orchestrator gate helpers and _run_pass_loop.

These tests cover both `--check-order=correctness-first` (existing behavior;
parity contract) and `--check-order=compliance-first` (new ordering).
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src import orchestrator as orch_mod


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_log_capture():
    msgs = []
    def log(m):
        msgs.append(m)
    return log, msgs


def test_gate_correctness_pass(tmp_path, monkeypatch):
    """match=True returns _GateResult(None, 'PASS', 0) and writes correctness_result.txt."""
    def fake_check(code, kernel, dims, tensors):
        return "match=True\nmax_diff=0.0"

    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl", fake_check)

    log, _ = _make_log_capture()
    res, shape_trace = _run(orch_mod._gate_correctness(
        code="pass", kernel_name="k", dims={}, tensors={},
        executor="dsl", turn_dir=tmp_path, log=log,
    ))

    assert res.feedback is None
    assert res.status == "PASS"
    assert res.tokens == 0
    assert shape_trace == ""
    assert (tmp_path / "correctness_result.txt").read_text() == "match=True\nmax_diff=0.0"


def test_gate_correctness_mismatch(tmp_path, monkeypatch):
    def fake_check(code, kernel, dims, tensors):
        return "match=False\nmax_diff=0.5"

    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl", fake_check)

    log, _ = _make_log_capture()
    res, _ = _run(orch_mod._gate_correctness(
        code="pass", kernel_name="k", dims={}, tensors={},
        executor="dsl", turn_dir=tmp_path, log=log,
    ))
    assert res.feedback == "## Correctness check result\nmatch=False\nmax_diff=0.5"
    assert res.status == "FAIL: match=False"
    assert res.tokens == 0


def test_gate_correctness_exception(tmp_path, monkeypatch):
    def fake_check(code, kernel, dims, tensors):
        raise ValueError("kaboom")

    monkeypatch.setitem(orch_mod._CORRECTNESS_CHECKERS, "dsl", fake_check)

    log, _ = _make_log_capture()
    res, _ = _run(orch_mod._gate_correctness(
        code="pass", kernel_name="k", dims={}, tensors={},
        executor="dsl", turn_dir=tmp_path, log=log,
    ))
    assert res.feedback.startswith("## Error running code\n")
    assert "ValueError: kaboom" in res.feedback
    assert res.status.startswith("FAIL:")
    written = (tmp_path / "correctness_result.txt").read_text()
    assert written.startswith("ERROR:\n")
