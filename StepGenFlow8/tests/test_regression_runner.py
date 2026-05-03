import asyncio
import sys
from pathlib import Path

import pytest

from src.regression_planning import Job
from src.regression_runner import JobResult, build_run_py_command, read_per_outer, run_jobs, run_subprocess


def test_build_run_py_command_minimal():
    job = Job("gemm", "small")
    cmd = build_run_py_command(
        job,
        python_exe=sys.executable,
        run_py_path=Path("run.py"),
        model="gpt-oss-120b",
        config=None,
        max_outer=None,
        max_turns=None,
        pipeline=None,
        translator=None,
        checkpoint_dir=None,
        autotune=False,
        autotune_config=None,
        autotune_max_turns=None,
        autotune_agent=None,
    )
    assert cmd == [sys.executable, "run.py", "gemm", "small", "--model", "gpt-oss-120b"]


def test_build_run_py_command_full_passthrough():
    job = Job("gemm", "small")
    cmd = build_run_py_command(
        job,
        python_exe="py",
        run_py_path=Path("run.py"),
        model="m",
        config="cfg.json",
        max_outer=2,
        max_turns=5,
        pipeline="direct",
        translator="llm",
        checkpoint_dir=Path("/tmp/ckpt"),
        autotune=False,
        autotune_config=None,
        autotune_max_turns=None,
        autotune_agent=None,
    )
    assert cmd == [
        "py", "run.py", "gemm", "small",
        "--model", "m",
        "--config", "cfg.json",
        "--max-outer", "2",
        "--max-turns", "5",
        "--pipeline", "direct",
        "--translator", "llm",
        "--checkpoint-dir", "/tmp/ckpt",
    ]


def test_build_run_py_command_with_autotune_flags():
    job = Job("gemm", "small")
    cmd = build_run_py_command(
        job,
        python_exe="py",
        run_py_path=Path("run.py"),
        model="m",
        config=None,
        max_outer=None,
        max_turns=None,
        pipeline=None,
        translator=None,
        checkpoint_dir=None,
        autotune=True,
        autotune_config="my_at.json",
        autotune_max_turns=4,
        autotune_agent="parallel",
    )
    assert cmd == [
        "py", "run.py", "gemm", "small",
        "--model", "m",
        "--autotune",
        "--autotune-config", "my_at.json",
        "--autotune-max-turns", "4",
        "--autotune-agent", "parallel",
    ]


def test_build_run_py_command_autotune_off_omits_flags():
    job = Job("gemm", "small")
    cmd = build_run_py_command(
        job,
        python_exe="py",
        run_py_path=Path("run.py"),
        model="m",
        config=None,
        max_outer=None,
        max_turns=None,
        pipeline=None,
        translator=None,
        checkpoint_dir=None,
        autotune=False,
        autotune_config=None,
        autotune_max_turns=None,
        autotune_agent=None,
    )
    assert "--autotune" not in cmd
    assert "--autotune-config" not in cmd
    assert "--autotune-max-turns" not in cmd
    assert "--autotune-agent" not in cmd


def _exit_with_cmd(code: int) -> list[str]:
    return [sys.executable, "-c", f"import sys; print('hi'); sys.exit({code})"]


def test_run_subprocess_pass(tmp_path: Path):
    log_path = tmp_path / "job.log"
    exit_code, duration = asyncio.run(
        run_subprocess(_exit_with_cmd(0), log_path, cwd=tmp_path)
    )
    assert exit_code == 0
    assert duration >= 0
    assert "hi" in log_path.read_text()


def test_run_subprocess_fail(tmp_path: Path):
    log_path = tmp_path / "job.log"
    exit_code, _ = asyncio.run(
        run_subprocess(_exit_with_cmd(7), log_path, cwd=tmp_path)
    )
    assert exit_code == 7


def test_run_jobs_caps_concurrency(tmp_path: Path):
    jobs = [Job(f"k{i}", "p") for i in range(8)]

    async def _drive():
        inflight = 0
        peak = 0
        lock = asyncio.Lock()

        async def stub(job: Job, log_path: Path) -> tuple[int, float, int, int, dict | None]:
            nonlocal inflight, peak
            async with lock:
                inflight += 1
                peak = max(peak, inflight)
            await asyncio.sleep(0.05)
            async with lock:
                inflight -= 1
            return 0, 0.05, 1, 1, None

        results = await run_jobs(jobs, max_parallel=3, jobs_dir=tmp_path, run_one=stub)
        return results, peak

    results, peak = asyncio.run(_drive())
    assert peak <= 3
    assert len(results) == 8
    assert all(r.status == "pass" for r in results)
    assert {r.job.kernel for r in results} == {f"k{i}" for i in range(8)}
    assert all(r.outer_passed == 1 for r in results)
    assert all(r.outer_total == 1 for r in results)


def test_run_jobs_records_failures(tmp_path: Path):
    jobs = [Job("a", "p"), Job("b", "p")]

    async def stub(job: Job, log_path: Path) -> tuple[int, float, int, int, dict | None]:
        if job.kernel == "a":
            return 0, 0.01, 1, 1, None
        return 1, 0.01, 0, 3, None

    results = asyncio.run(
        run_jobs(jobs, max_parallel=2, jobs_dir=tmp_path, run_one=stub)
    )
    by_kernel = {r.job.kernel: r for r in results}
    assert by_kernel["a"].status == "pass"
    assert by_kernel["b"].status == "fail"
    assert by_kernel["b"].exit_code == 1
    assert by_kernel["b"].outer_passed == 0
    assert by_kernel["b"].outer_total == 3


import json

from src.regression_runner import init_output_dir, write_summary


def test_init_output_dir_creates_timestamped_subtree(tmp_path: Path):
    out = init_output_dir(tmp_path)
    assert out.parent == tmp_path
    assert (out / "jobs").is_dir()
    # name is YYYYMMDD-HHMMSS
    name = out.name
    assert len(name) == 15 and name[8] == "-"


def test_write_summary_aggregates_per_benchmark_and_overall(tmp_path: Path):
    results = [
        JobResult(Job("gemm", "small"),  status="pass", exit_code=0, duration_s=1.0, outer_passed=3, outer_total=3),
        JobResult(Job("gemm", "square"), status="fail", exit_code=1, duration_s=2.0, outer_passed=1, outer_total=3),
        JobResult(Job("silu", "small"),  status="pass", exit_code=0, duration_s=0.5, outer_passed=2, outer_total=2),
    ]
    summary_path = tmp_path / "summary.json"
    write_summary(
        summary_path,
        results=results,
        started_at="2026-04-27T14:00:00Z",
        finished_at="2026-04-27T14:05:00Z",
        wall_seconds=300.0,
        max_parallel=2,
        model="gpt-oss-120b",
        preset_mode="all_presets",
    )
    data = json.loads(summary_path.read_text())
    assert data["overall"] == {"passed": 2, "total": 3, "fraction": pytest.approx(2 / 3)}
    assert data["benchmarks"]["gemm"]["passed"] == 1
    assert data["benchmarks"]["gemm"]["total"] == 2
    assert data["benchmarks"]["gemm"]["fraction"] == pytest.approx(0.5)
    assert data["benchmarks"]["gemm"]["presets"]["small"]["status"] == "pass"
    assert data["benchmarks"]["gemm"]["presets"]["square"]["exit_code"] == 1
    assert data["benchmarks"]["gemm"]["presets"]["small"]["outer_passed"] == 3
    assert data["benchmarks"]["gemm"]["presets"]["small"]["outer_total"] == 3
    assert data["benchmarks"]["gemm"]["presets"]["square"]["outer_passed"] == 1
    assert data["benchmarks"]["gemm"]["presets"]["square"]["outer_total"] == 3
    assert data["model"] == "gpt-oss-120b"
    assert data["outer_overall"] == {"passed": 6, "total": 8}


import logging as _logging

from src.regression_runner import setup_logging


def test_setup_logging_writes_to_file_and_returns_logger(tmp_path: Path):
    log_path = tmp_path / "regression.log"
    logger = setup_logging(log_path)
    logger.info("hello-world-marker")
    for h in logger.handlers:
        h.flush()
    assert "hello-world-marker" in log_path.read_text()
    # Cleanup so other tests don't inherit handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def test_read_per_outer_returns_passed_and_total(tmp_path: Path):
    (tmp_path / "result.json").write_text(json.dumps({
        "success": True,
        "per_outer": [
            {"outer": 0, "success": False},
            {"outer": 1, "success": True},
            {"outer": 2, "success": True},
        ],
    }))
    assert read_per_outer(tmp_path) == (2, 3)


def test_read_per_outer_missing_file(tmp_path: Path):
    assert read_per_outer(tmp_path) == (0, 0)


def test_read_per_outer_missing_per_outer_key(tmp_path: Path):
    (tmp_path / "result.json").write_text(json.dumps({"success": False}))
    assert read_per_outer(tmp_path) == (0, 0)


from src.regression_runner import read_autotune


def _write_result_json(tmp_path: Path, per_outer: list[dict], **extra) -> None:
    payload = {"success": True, "per_outer": per_outer, **extra}
    (tmp_path / "result.json").write_text(json.dumps(payload))


def test_read_autotune_returns_none_when_no_result_json(tmp_path: Path):
    assert read_autotune(tmp_path) is None


def test_read_autotune_returns_none_when_no_per_outer(tmp_path: Path):
    (tmp_path / "result.json").write_text(json.dumps({"success": False}))
    assert read_autotune(tmp_path) is None


def test_read_autotune_returns_none_when_no_outer_has_autotune(tmp_path: Path):
    _write_result_json(tmp_path, [
        {"outer": 0, "success": True, "autotune": None},
        {"outer": 1, "success": False, "autotune": None},
    ])
    assert read_autotune(tmp_path) is None


def test_read_autotune_returns_none_when_only_errors(tmp_path: Path):
    _write_result_json(tmp_path, [
        {"outer": 0, "success": True, "autotune": {
            "status": "error", "error": "boom",
            "baseline_cycles": 1000, "best_cycles": 950, "speedup": 1000/950,
            "checkpoint_dir": "x",
        }},
    ])
    assert read_autotune(tmp_path) is None


def test_read_autotune_picks_best_outer_with_lowest_best_cycles(tmp_path: Path):
    _write_result_json(tmp_path, [
        {"outer": 0, "success": True, "autotune": {
            "status": "ok", "baseline_cycles": 1000, "best_cycles": 900,
            "speedup": 1000/900, "checkpoint_dir": "x"}},
        {"outer": 1, "success": True, "autotune": {
            "status": "ok", "baseline_cycles": 1000, "best_cycles": 850,
            "speedup": 1000/850, "checkpoint_dir": "y"}},
        {"outer": 2, "success": True, "autotune": {
            "status": "error", "error": "z",
            "baseline_cycles": 1000, "best_cycles": 920,
            "speedup": 1000/920, "checkpoint_dir": "z"}},
        {"outer": 3, "success": False, "autotune": None},
    ])
    out = read_autotune(tmp_path)
    assert out is not None
    assert out["best_outer"] == 1
    assert out["baseline_cycles"] == 1000
    assert out["best_cycles"] == 850
    assert out["speedup"] == pytest.approx(1000 / 850)
    assert len(out["per_outer"]) == 4
    statuses = [e["status"] for e in out["per_outer"]]
    assert statuses == ["ok", "ok", "error", "missing"]
