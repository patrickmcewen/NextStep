import asyncio
import json
from pathlib import Path

import pytest

from src import orchestrator as orch_module
from src.orchestrator import _load_autotune_progress, _run_outer_autotune


def test_load_autotune_progress_returns_dict_when_file_exists(tmp_path: Path):
    (tmp_path / "progress.json").write_text(json.dumps({
        "baseline_cycles": 1000, "best_cycles": 900,
        "turn": 3, "last_status": "NEW_BEST",
    }))
    out = _load_autotune_progress(tmp_path)
    assert out == {
        "baseline_cycles": 1000, "best_cycles": 900,
        "turn": 3, "last_status": "NEW_BEST",
    }


def test_load_autotune_progress_returns_empty_dict_when_missing(tmp_path: Path):
    assert _load_autotune_progress(tmp_path) == {}


def test_load_autotune_progress_returns_empty_dict_when_dir_missing(tmp_path: Path):
    assert _load_autotune_progress(tmp_path / "does-not-exist") == {}


def _stub_run_autotune_ok(returned: dict):
    async def stub(**kwargs):
        return returned
    return stub


def _stub_run_autotune_raises(exc: Exception):
    async def stub(**kwargs):
        raise exc
    return stub


def test_run_outer_autotune_success_path(tmp_path: Path, monkeypatch):
    fake_result = {
        "success": True,
        "kernel": "gemm", "preset": "small",
        "baseline_cycles": 1000, "best_cycles": 800, "speedup": 1.25,
        "turns": 4, "resume_from": "x", "checkpoint_dir": str(tmp_path / "autotune"),
    }
    monkeypatch.setattr(orch_module, "run_autotune", _stub_run_autotune_ok(fake_result))

    logged = []
    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {"hw_config": {}, "constraints": {"max_total_compute_bw": 1}}, "max_turns": 2, "agent_variant": "general"},
        log=logged.append,
        tag="[outer_0]",
    ))
    assert out["status"] == "ok"
    assert out["baseline_cycles"] == 1000
    assert out["best_cycles"] == 800
    assert out["speedup"] == 1.25


def test_run_outer_autotune_traps_exception_with_progress(tmp_path: Path, monkeypatch):
    autotune_kernel_dir = tmp_path / "autotune" / "gemm"
    autotune_kernel_dir.mkdir(parents=True)
    (autotune_kernel_dir / "progress.json").write_text(json.dumps({
        "baseline_cycles": 1000, "best_cycles": 850, "turn": 2, "last_status": "NEW_BEST",
    }))
    monkeypatch.setattr(orch_module, "run_autotune",
                        _stub_run_autotune_raises(RuntimeError("kaboom")))

    logged = []
    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {}, "max_turns": None, "agent_variant": "general"},
        log=logged.append,
        tag="[outer_0]",
    ))
    assert out["status"] == "error"
    assert "RuntimeError" in out["error"]
    assert out["baseline_cycles"] == 1000
    assert out["best_cycles"] == 850
    assert out["speedup"] == pytest.approx(1000 / 850)
    assert out["checkpoint_dir"] == str(tmp_path / "autotune")
    assert any("autotune FAILED" in m for m in logged)


def test_run_outer_autotune_traps_exception_without_progress(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(orch_module, "run_autotune",
                        _stub_run_autotune_raises(RuntimeError("kaboom")))

    logged = []
    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {}, "max_turns": None, "agent_variant": "general"},
        log=logged.append,
        tag="[outer_0]",
    ))
    assert out["status"] == "error"
    assert out["baseline_cycles"] is None
    assert out["best_cycles"] is None
    assert out["speedup"] is None
