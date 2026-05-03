"""Async subprocess runner + summary writer for the regression script."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from src.regression_planning import Job

_log = logging.getLogger(__name__)

RunOne = Callable[[Job, Path], Awaitable[tuple[int, float, int, int, dict | None]]]


@dataclass(frozen=True)
class JobResult:
    job: Job
    status: str          # "pass" or "fail"
    exit_code: int
    duration_s: float
    outer_passed: int
    outer_total: int
    autotune: dict | None = None


def build_run_py_command(
    job: Job,
    *,
    python_exe: str,
    run_py_path: Path,
    model: str,
    config: str | None,
    max_outer: int | None,
    max_turns: int | None,
    pipeline: str | None,
    translator: str | None,
    checkpoint_dir: Path | None,
    autotune: bool,
    autotune_config: str | None,
    autotune_max_turns: int | None,
    autotune_agent: str | None,
) -> list[str]:
    cmd = [python_exe, str(run_py_path), job.kernel, job.preset, "--model", model]
    if config is not None:
        cmd += ["--config", config]
    if max_outer is not None:
        cmd += ["--max-outer", str(max_outer)]
    if max_turns is not None:
        cmd += ["--max-turns", str(max_turns)]
    if pipeline is not None:
        cmd += ["--pipeline", pipeline]
    if translator is not None:
        cmd += ["--translator", translator]
    if checkpoint_dir is not None:
        cmd += ["--checkpoint-dir", str(checkpoint_dir)]
    if autotune:
        cmd += ["--autotune"]
        if autotune_config is not None:
            cmd += ["--autotune-config", autotune_config]
        if autotune_max_turns is not None:
            cmd += ["--autotune-max-turns", str(autotune_max_turns)]
        if autotune_agent is not None:
            cmd += ["--autotune-agent", autotune_agent]
    return cmd


def read_per_outer(checkpoint_dir: Path) -> tuple[int, int]:
    """Read `<checkpoint_dir>/result.json` and return (outer_passed, outer_total).

    Falls back to (0, 0) if the file is missing or doesn't include per_outer.
    Caller treats (0, 0) as "no info available" and may substitute exit-code
    based defaults.
    """
    result_path = checkpoint_dir / "result.json"
    if not result_path.exists():
        return 0, 0
    data = json.loads(result_path.read_text())
    per_outer = data.get("per_outer")
    if not per_outer:
        return 0, 0
    passed = sum(1 for entry in per_outer if entry.get("success"))
    return passed, len(per_outer)


def read_autotune(checkpoint_dir: Path) -> dict | None:
    """Return best-across-outers autotune summary, or None if no data.

    Reads <checkpoint_dir>/result.json's per_outer entries, picks the entry
    with status=='ok' that has the lowest best_cycles, and returns:
      {
        "best_outer": int,
        "baseline_cycles": int,
        "best_cycles": int,
        "speedup": float,
        "per_outer": [{outer, status, baseline_cycles, best_cycles, speedup}, ...],
      }
    Returns None when no outer has status=='ok' (autotune disabled, no outer
    succeeded, or every outer's autotune crashed).
    """
    result_path = checkpoint_dir / "result.json"
    if not result_path.exists():
        return None
    data = json.loads(result_path.read_text())
    per_outer = data.get("per_outer")
    if not per_outer:
        return None

    summarized = []
    for entry in per_outer:
        at = entry.get("autotune")
        if at is None:
            summarized.append({
                "outer": entry.get("outer"),
                "status": "missing",
                "baseline_cycles": None,
                "best_cycles": None,
                "speedup": None,
            })
        else:
            summarized.append({
                "outer": entry.get("outer"),
                "status": at.get("status"),
                "baseline_cycles": at.get("baseline_cycles"),
                "best_cycles": at.get("best_cycles"),
                "speedup": at.get("speedup"),
            })

    ok_entries = [e for e in summarized
                  if e["status"] == "ok" and e["best_cycles"] is not None]
    if not ok_entries:
        return None

    best = min(ok_entries, key=lambda e: e["best_cycles"])
    return {
        "best_outer": best["outer"],
        "baseline_cycles": best["baseline_cycles"],
        "best_cycles": best["best_cycles"],
        "speedup": best["speedup"],
        "per_outer": summarized,
    }


async def run_subprocess(cmd: list[str], log_path: Path, cwd: Path) -> tuple[int, float]:
    """Run `cmd` as a subprocess; merge stdout+stderr into `log_path`.

    Returns (exit_code, duration_seconds). Does not raise on non-zero exit.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with open(log_path, "wb") as log_f:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=log_f,
            stderr=asyncio.subprocess.STDOUT,
        )
        exit_code = await proc.wait()
    return exit_code, time.monotonic() - start


async def run_jobs(
    jobs: list[Job],
    *,
    max_parallel: int,
    jobs_dir: Path,
    run_one: RunOne,
) -> list[JobResult]:
    """Run all jobs with at most `max_parallel` in flight at once.

    `run_one(job, log_path)` returns `(exit_code, duration_s)`. Caller injects
    this so tests can stub subprocesses; production wires it to a closure that
    builds the run.py command and calls run_subprocess.
    """
    assert max_parallel >= 1, "max_parallel must be >= 1"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max_parallel)
    total = len(jobs)
    completed = 0
    passed = 0

    async def _run(job: Job) -> JobResult:
        nonlocal completed, passed
        log_path = jobs_dir / f"{job.kernel}__{job.preset}.log"
        async with sem:
            _log.info("START %s/%s", job.kernel, job.preset)
            exit_code, duration, outer_passed, outer_total, autotune = \
                await run_one(job, log_path)
        status = "pass" if exit_code == 0 else "fail"
        completed += 1
        if status == "pass":
            passed += 1
            _log.info(
                "PASS %s/%s (%.1fs, outer %d/%d)",
                job.kernel, job.preset, duration, outer_passed, outer_total,
            )
        else:
            _log.warning(
                "FAIL %s/%s (%.1fs, exit=%d, outer %d/%d)",
                job.kernel, job.preset, duration, exit_code, outer_passed, outer_total,
            )
        _log.info("[%d/%d done, %d passed]", completed, total, passed)
        return JobResult(
            job=job, status=status, exit_code=exit_code, duration_s=duration,
            outer_passed=outer_passed, outer_total=outer_total, autotune=autotune,
        )

    return await asyncio.gather(*(_run(j) for j in jobs))


def init_output_dir(results_root: Path) -> Path:
    """Create `results_root/<YYYYMMDD-HHMMSS>/jobs/` and return the run dir."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = results_root / stamp
    (out / "jobs").mkdir(parents=True, exist_ok=False)
    return out


def write_summary(
    path: Path,
    *,
    results: list[JobResult],
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    max_parallel: int,
    model: str,
    preset_mode: str,
) -> None:
    by_kernel: dict[str, list[JobResult]] = defaultdict(list)
    for r in results:
        by_kernel[r.job.kernel].append(r)

    benchmarks = {}
    for kernel, group in sorted(by_kernel.items()):
        passed = sum(1 for r in group if r.status == "pass")
        total = len(group)
        benchmarks[kernel] = {
            "passed": passed,
            "total": total,
            "fraction": passed / total,
            "presets": {
                r.job.preset: {
                    "status": r.status,
                    "duration_s": round(r.duration_s, 3),
                    "exit_code": r.exit_code,
                    "outer_passed": r.outer_passed,
                    "outer_total": r.outer_total,
                }
                for r in sorted(group, key=lambda r: r.job.preset)
            },
        }

    overall_passed = sum(1 for r in results if r.status == "pass")
    overall_total = len(results)
    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(wall_seconds, 3),
        "max_parallel": max_parallel,
        "model": model,
        "preset_mode": preset_mode,
        "overall": {
            "passed": overall_passed,
            "total": overall_total,
            "fraction": (overall_passed / overall_total) if overall_total else 0.0,
        },
        "outer_overall": {
            "passed": sum(r.outer_passed for r in results),
            "total": sum(r.outer_total for r in results),
        },
        "benchmarks": benchmarks,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure a dedicated logger that tees to stdout and `log_path`.

    Returns the runner's logger. Idempotent per `log_path`: handlers attached
    to a previous call on the same path are not duplicated.
    """
    logger = logging.getLogger("regression")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    # Avoid duplicate handlers on repeat calls (matters in tests).
    existing_files = {getattr(h, "baseFilename", None) for h in logger.handlers}
    if str(log_path) not in existing_files:
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    # Also funnel module loggers (regression_planning, regression_runner) here.
    for name in ("src.regression_planning", "src.regression_runner"):
        sub = logging.getLogger(name)
        sub.setLevel(logging.INFO)
        sub.handlers.clear()
        sub.propagate = False
        for h in logger.handlers:
            sub.addHandler(h)
    return logger
