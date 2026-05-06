import asyncio
import json
from pathlib import Path

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


def test_run_outer_autotune_single_pass_success(tmp_path: Path, monkeypatch):
    """One pass, feasible result -> status='ok' with one pass entry."""
    fake_result = {
        "success": True,
        "kernel": "gemm", "preset": "small",
        "baseline_cycles": 1000, "best_cycles": 800, "speedup": 1.25,
        "baseline_on_chip_bytes": 100, "best_on_chip_bytes": 100,
        "baseline_off_chip_bytes": 0, "best_off_chip_bytes": 0,
        "baseline_feasible": True, "feasible": True, "feasibility": None,
        "turns": 4, "resume_from": "x",
        "checkpoint_dir": str(tmp_path / "autotune" / "pass_0_general" / "gemm"),
        "agent_variant": "general",
    }
    monkeypatch.setattr(orch_module, "run_autotune", _stub_run_autotune_ok(fake_result))

    logged = []
    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {"hw_config": {}, "constraints": {"max_total_compute_bw": 1}},
                          "passes": [{"agent": "general", "max_turns": 4}]},
        log=logged.append,
        tag="[outer_0]",
    ))
    assert out["status"] == "ok"
    assert out["halt_reason"] is None
    assert len(out["passes"]) == 1
    assert out["passes"][0]["best_cycles"] == 800
    assert out["overall"]["baseline_cycles"] == 1000
    assert out["overall"]["best_cycles"] == 800
    assert out["overall"]["feasible"] is True


def test_run_outer_autotune_chain_two_passes_feeds_best_forward(
    tmp_path: Path, monkeypatch
):
    """Pass 1's resume_from must be pass 0's best.py path."""
    captured_resume_args = []

    async def stub(**kwargs):
        captured_resume_args.append(kwargs["resume_from"])
        idx = len(captured_resume_args) - 1
        return {
            "success": True, "kernel": "gemm", "preset": "small",
            "baseline_cycles": 1000 - idx*100, "best_cycles": 900 - idx*200,
            "speedup": 1.0, "baseline_on_chip_bytes": 0, "best_on_chip_bytes": 0,
            "baseline_off_chip_bytes": 0, "best_off_chip_bytes": 0,
            "baseline_feasible": True, "feasible": True, "feasibility": None,
            "turns": 2, "resume_from": kwargs["resume_from"],
            "checkpoint_dir": str(Path(kwargs["checkpoint_dir"]) / "gemm"),
            "agent_variant": kwargs["agent_variant"],
        }
    monkeypatch.setattr(orch_module, "run_autotune", stub)

    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {"hw_config": {}, "constraints": {"max_total_compute_bw": 1}},
                          "passes": [
                              {"agent": "general", "max_turns": 2},
                              {"agent": "parallel", "max_turns": 2},
                          ]},
        log=[].append,
        tag="[outer_0]",
    ))
    assert out["status"] == "ok"
    assert len(out["passes"]) == 2
    # Pass 0 was given the outer dir; pass 1 was given pass 0's best.py.
    assert captured_resume_args[0] == str(tmp_path)
    assert captured_resume_args[1].endswith("pass_0_general/gemm/best.py")
    # Overall = baseline of pass 0 -> best of pass 1.
    assert out["overall"]["baseline_cycles"] == 1000
    assert out["overall"]["best_cycles"] == 700


def test_run_outer_autotune_halts_chain_on_infeasible(tmp_path: Path, monkeypatch):
    """If a pass returns feasible=False, subsequent passes don't run."""
    call_count = {"n": 0}

    async def stub(**kwargs):
        call_count["n"] += 1
        return {
            "success": True, "kernel": "gemm", "preset": "small",
            "baseline_cycles": 1000, "best_cycles": 1000, "speedup": 1.0,
            "baseline_on_chip_bytes": 400, "best_on_chip_bytes": 350,
            "baseline_off_chip_bytes": 0, "best_off_chip_bytes": 0,
            "baseline_feasible": False, "feasible": False,
            "feasibility": kwargs.get("feasibility"),
            "turns": 2, "resume_from": kwargs["resume_from"],
            "checkpoint_dir": str(Path(kwargs["checkpoint_dir"]) / "gemm"),
            "agent_variant": kwargs["agent_variant"],
        }
    monkeypatch.setattr(orch_module, "run_autotune", stub)

    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {"hw_config": {}, "constraints": {"max_total_compute_bw": 1}},
                          "passes": [
                              {"agent": "memory", "max_turns": 2,
                               "feasibility": {"on_chip_bytes": 256}},
                              {"agent": "general", "max_turns": 2},
                          ]},
        log=[].append,
        tag="[outer_0]",
    ))
    assert call_count["n"] == 1   # second pass skipped
    assert out["status"] == "halted"
    assert out["halt_reason"] == "infeasible"
    assert out["halted_pass_index"] == 0
    assert out["overall"]["feasible"] is False


def test_run_outer_autotune_pass_exception_recovers_progress(
    tmp_path: Path, monkeypatch
):
    autotune_kernel_dir = tmp_path / "autotune" / "pass_0_general" / "gemm"
    autotune_kernel_dir.mkdir(parents=True)
    (autotune_kernel_dir / "progress.json").write_text(json.dumps({
        "baseline_cycles": 1000, "best_cycles": 850,
        "turn": 2, "last_status": "NEW_BEST",
        "best_on_chip_bytes": 100, "best_off_chip_bytes": 0,
        "last_on_chip_bytes": 100, "last_off_chip_bytes": 0,
    }))
    monkeypatch.setattr(orch_module, "run_autotune",
                        _stub_run_autotune_raises(RuntimeError("kaboom")))

    logged = []
    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {}, "passes": [{"agent": "general", "max_turns": 2}]},
        log=logged.append,
        tag="[outer_0]",
    ))
    assert out["status"] == "error"
    assert "RuntimeError" in out["error"]
    assert out["halted_pass_index"] == 0
    # Overall surfaces the recovered partial best.
    assert out["overall"]["baseline_cycles"] == 1000
    assert out["overall"]["best_cycles"] == 850
    assert any("autotune FAILED" in m for m in logged)


def test_run_outer_autotune_mid_chain_infeasible(tmp_path: Path, monkeypatch):
    """First pass feasible, second pass infeasible: chain halts at idx 1, and
    overall pulls baseline from pass 0 / best from pass 1."""
    call_count = {"n": 0}

    async def stub(**kwargs):
        i = call_count["n"]
        call_count["n"] += 1
        return {
            "success": True, "kernel": "gemm", "preset": "small",
            "baseline_cycles": 1000 if i == 0 else 800,
            "best_cycles": 800 if i == 0 else 800,
            "speedup": 1.0,
            "baseline_on_chip_bytes": 0, "best_on_chip_bytes": 0,
            "baseline_off_chip_bytes": 0, "best_off_chip_bytes": 0,
            "baseline_feasible": True,
            "feasible": (i == 0),  # pass 0 ok, pass 1 fails feasibility
            "feasibility": kwargs.get("feasibility"),
            "turns": 2, "resume_from": kwargs["resume_from"],
            "checkpoint_dir": str(Path(kwargs["checkpoint_dir"]) / "gemm"),
            "agent_variant": kwargs["agent_variant"],
        }
    monkeypatch.setattr(orch_module, "run_autotune", stub)

    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {"hw_config": {}, "constraints": {"max_total_compute_bw": 1}},
                          "passes": [
                              {"agent": "general", "max_turns": 2},
                              {"agent": "memory", "max_turns": 2,
                               "feasibility": {"on_chip_bytes": 256}},
                              {"agent": "parallel", "max_turns": 2},
                          ]},
        log=[].append,
        tag="[outer_0]",
    ))
    assert call_count["n"] == 2  # pass 2 (parallel) skipped
    assert out["status"] == "halted"
    assert out["halted_pass_index"] == 1
    assert out["overall"]["baseline_cycles"] == 1000  # from pass 0
    assert out["overall"]["best_cycles"] == 800       # from pass 1
    assert out["overall"]["feasible"] is False


def test_run_outer_autotune_mid_chain_exception(tmp_path: Path, monkeypatch):
    """Pass 0 succeeds, pass 1 raises: aggregation reads pass 0's baseline,
    not the error partial's."""
    call_count = {"n": 0}

    async def stub(**kwargs):
        i = call_count["n"]
        call_count["n"] += 1
        if i == 1:
            raise RuntimeError("pass1 boom")
        return {
            "success": True, "kernel": "gemm", "preset": "small",
            "baseline_cycles": 1000, "best_cycles": 800, "speedup": 1.25,
            "baseline_on_chip_bytes": 0, "best_on_chip_bytes": 0,
            "baseline_off_chip_bytes": 0, "best_off_chip_bytes": 0,
            "baseline_feasible": True, "feasible": True, "feasibility": None,
            "turns": 2, "resume_from": kwargs["resume_from"],
            "checkpoint_dir": str(Path(kwargs["checkpoint_dir"]) / "gemm"),
            "agent_variant": kwargs["agent_variant"],
        }
    monkeypatch.setattr(orch_module, "run_autotune", stub)

    out = asyncio.run(_run_outer_autotune(
        outer_dir=tmp_path,
        kernel_name="gemm",
        preset="small",
        llm_config={"profile": "x"},
        autotune_options={"config": {}, "passes": [
            {"agent": "general", "max_turns": 2},
            {"agent": "parallel", "max_turns": 2},
        ]},
        log=[].append,
        tag="[outer_0]",
    ))
    assert out["status"] == "error"
    assert out["halted_pass_index"] == 1
    assert len(out["passes"]) == 2
    assert out["passes"][0]["status"] == "ok"
    assert out["passes"][1]["status"] == "error"
    # Overall reads pass 0's baseline (real) and pass 1's recovered best (None
    # since no progress.json was written for pass 1).
    assert out["overall"]["baseline_cycles"] == 1000
    assert out["overall"]["best_cycles"] is None
    # Schema-symmetry check: error entry has the same key shape as success.
    assert set(out["passes"][1].keys()) >= {
        "feasible", "baseline_feasible", "speedup",
        "baseline_on_chip_bytes", "baseline_off_chip_bytes",
    }
