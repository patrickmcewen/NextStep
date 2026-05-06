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
