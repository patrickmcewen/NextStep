#!/usr/bin/env python3
"""Batch runner for run_autotune2.py over many outer-dir checkpoints.

The YAML config supplies checkpoint ``outer_*/`` paths plus the
``run_autotune2.py`` runtime args that should be explicitly forwarded.
Unspecified autotune2 args are not emitted, so ``run_autotune2.py`` keeps
its own defaults.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.log_redirect import redirect_stdio_to, terminal_print
from src.process_group import setup_process_group

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "autotune2_batch.yaml"
DEFAULT_RUN_AUTOTUNE2 = REPO_ROOT / "run_autotune2.py"
DEFAULT_RESULTS_ROOT = Path("/workspace/autotune2_batch_results")

_log = logging.getLogger("autotune2_batch")


@dataclass(frozen=True)
class Autotune2Job:
    outer_dir: Path
    label: str
    autotune2_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchConfig:
    jobs: list[Autotune2Job]
    max_parallel: int = 1
    results_root: Path = DEFAULT_RESULTS_ROOT
    run_autotune2: Path = DEFAULT_RUN_AUTOTUNE2
    autotune2_args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchResult:
    job: Autotune2Job
    status: str
    exit_code: int
    duration_s: float
    log_path: Path
    command: list[str]


def _flag_name(key: str) -> str:
    name = key[2:] if key.startswith("--") else key
    name = name.replace("_", "-")
    assert name, "CLI arg name cannot be empty"
    return f"--{name}"


def flatten_cli_args(args: dict[str, Any]) -> list[str]:
    """Convert a YAML mapping into argv tokens.

    Rules:
      - ``null`` values are omitted, preserving run_autotune2.py defaults.
      - ``true`` emits ``--flag``.
      - ``false`` emits ``--no-flag`` for argparse BooleanOptionalAction args.
      - scalar values emit ``--flag value``.
      - lists repeat the same flag once per list item.
    """
    assert isinstance(args, dict), "autotune2_args must be a YAML mapping"
    tokens: list[str] = []
    for key, value in args.items():
        flag = _flag_name(str(key))
        if value is None:
            continue
        if isinstance(value, bool):
            tokens.append(flag if value else f"--no-{flag[2:]}")
            continue
        if isinstance(value, list):
            for item in value:
                assert item is not None, f"{key}: list values cannot be null"
                assert not isinstance(item, (dict, list)), (
                    f"{key}: nested list/dict values are not supported"
                )
                tokens.extend([flag, str(item)])
            continue
        assert not isinstance(value, dict), (
            f"{key}: nested mappings are not supported in autotune2_args"
        )
        tokens.extend([flag, str(value)])
    return tokens


def build_autotune2_command(
    *,
    outer_dir: Path,
    python_exe: str,
    run_autotune2: Path,
    autotune2_args: dict[str, Any],
) -> list[str]:
    return [
        python_exe,
        str(run_autotune2),
        str(outer_dir),
        *flatten_cli_args(autotune2_args),
    ]


def init_output_dir(results_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = results_root / stamp
    (out / "jobs").mkdir(parents=True, exist_ok=False)
    return out


async def run_subprocess(cmd: list[str], log_path: Path, cwd: Path) -> tuple[int, float]:
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


def _job_label(outer_dir: Path) -> str:
    parts = outer_dir.parts[-3:] if len(outer_dir.parts) >= 3 else outer_dir.parts
    raw = "__".join(parts)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _load_job(raw: Any, inherited_args: dict[str, Any]) -> Autotune2Job:
    if isinstance(raw, (str, Path)):
        outer_dir = Path(raw)
        return Autotune2Job(outer_dir=outer_dir, label=_job_label(outer_dir))
    assert isinstance(raw, dict), (
        "each jobs entry must be either a path string or a mapping"
    )
    assert "outer_dir" in raw, "jobs entries must include outer_dir"
    outer_dir = Path(raw["outer_dir"])
    label = str(raw.get("label") or _job_label(outer_dir))
    job_args = raw.get("autotune2_args", {})
    assert isinstance(job_args, dict), "jobs[].autotune2_args must be a mapping"
    merged_args = {**inherited_args, **job_args}
    return Autotune2Job(
        outer_dir=outer_dir,
        label=label,
        autotune2_args=merged_args,
    )


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:i]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in ("null", "Null", "NULL", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _split_key_value(text: str) -> tuple[str, str]:
    key, sep, value = text.partition(":")
    assert sep, f"expected key: value YAML line, got {text!r}"
    key = key.strip()
    assert key, f"empty YAML key in line {text!r}"
    return key, value.strip()


def _simple_yaml_load(text: str) -> Any:
    """Small YAML subset parser used when PyYAML is unavailable.

    Supports the mapping/list/scalar shape used by ``autotune2_batch.yaml``.
    PyYAML remains the preferred parser when installed.
    """
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        without_comment = _strip_yaml_comment(raw).rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip(" "))
        lines.append((indent, without_comment.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        assert index < len(lines), "unexpected end of YAML"
        if lines[index][1].startswith("- "):
            out: list[Any] = []
            while index < len(lines):
                line_indent, text_line = lines[index]
                if line_indent != indent or not text_line.startswith("- "):
                    break
                item_text = text_line[2:].strip()
                index += 1
                if not item_text:
                    item, index = parse_block(index, indent + 2)
                    out.append(item)
                    continue
                if ":" in item_text:
                    key, value = _split_key_value(item_text)
                    item = {key: _parse_scalar(value) if value else None}
                    if index < len(lines) and lines[index][0] > indent:
                        child, index = parse_block(index, lines[index][0])
                        assert isinstance(child, dict), (
                            "list item continuation must be a mapping"
                        )
                        item.update(child)
                    out.append(item)
                else:
                    out.append(_parse_scalar(item_text))
            return out, index

        out: dict[str, Any] = {}
        while index < len(lines):
            line_indent, text_line = lines[index]
            if line_indent != indent or text_line.startswith("- "):
                break
            key, value = _split_key_value(text_line)
            index += 1
            if value:
                out[key] = _parse_scalar(value)
            else:
                assert index < len(lines), f"missing value for YAML key {key!r}"
                child, index = parse_block(index, lines[index][0])
                out[key] = child
        return out, index

    if not lines:
        return None
    parsed, final_index = parse_block(0, lines[0][0])
    assert final_index == len(lines), "failed to parse complete YAML document"
    return parsed


def _load_yaml(path: Path) -> Any:
    text = path.read_text()
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _simple_yaml_load(text)
    return yaml.safe_load(text)


def load_batch_config(path: Path) -> BatchConfig:
    assert path.exists(), f"batch config not found: {path}"
    data = _load_yaml(path)
    assert isinstance(data, dict), "batch config must be a YAML mapping"

    global_args = data.get("autotune2_args", {})
    assert isinstance(global_args, dict), "autotune2_args must be a mapping"

    raw_jobs = data.get("jobs")
    if raw_jobs is None:
        raw_jobs = data.get("outer_dirs")
    assert isinstance(raw_jobs, list) and raw_jobs, (
        "batch config must include a non-empty jobs or outer_dirs list"
    )
    jobs = [_load_job(raw, global_args) for raw in raw_jobs]
    for job in jobs:
        assert job.outer_dir.is_dir(), f"outer_dir not found: {job.outer_dir}"

    max_parallel = int(data.get("max_parallel", 1))
    assert max_parallel >= 1, "max_parallel must be >= 1"
    results_root = Path(data.get("results_root", DEFAULT_RESULTS_ROOT))
    run_autotune2 = Path(data.get("run_autotune2", DEFAULT_RUN_AUTOTUNE2))

    return BatchConfig(
        jobs=jobs,
        max_parallel=max_parallel,
        results_root=results_root,
        run_autotune2=run_autotune2,
        autotune2_args=global_args,
    )


def _setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("autotune2_batch")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if not any(getattr(h, "baseFilename", None) == str(log_path)
               for h in logger.handlers):
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if sys.stdout.isatty() and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


async def _run_jobs(
    *,
    config: BatchConfig,
    out_dir: Path,
    python_exe: str,
) -> list[BatchResult]:
    sem = asyncio.Semaphore(config.max_parallel)
    total = len(config.jobs)
    completed = 0
    passed = 0

    async def run_one(job: Autotune2Job) -> BatchResult:
        nonlocal completed, passed
        log_path = out_dir / "jobs" / f"{job.label}.log"
        command = build_autotune2_command(
            outer_dir=job.outer_dir,
            python_exe=python_exe,
            run_autotune2=config.run_autotune2,
            autotune2_args=job.autotune2_args,
        )
        async with sem:
            _log.info("START %s", job.label)
            exit_code, duration = await run_subprocess(
                command, log_path, cwd=REPO_ROOT,
            )
        status = "pass" if exit_code == 0 else "fail"
        completed += 1
        if status == "pass":
            passed += 1
            _log.info("PASS %s (%.1fs)", job.label, duration)
        else:
            _log.warning(
                "FAIL %s (%.1fs, exit=%d)", job.label, duration, exit_code,
            )
        _log.info("[%d/%d done, %d passed]", completed, total, passed)
        return BatchResult(
            job=job,
            status=status,
            exit_code=exit_code,
            duration_s=duration,
            log_path=log_path,
            command=command,
        )

    return await asyncio.gather(*(run_one(job) for job in config.jobs))


def _write_summary(
    path: Path,
    *,
    results: list[BatchResult],
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    max_parallel: int,
) -> None:
    payload = {
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": round(wall_seconds, 3),
        "max_parallel": max_parallel,
        "overall": {
            "passed": sum(1 for r in results if r.status in ("pass", "dry-run")),
            "total": len(results),
        },
        "jobs": [
            {
                "label": r.job.label,
                "outer_dir": str(r.job.outer_dir),
                "status": r.status,
                "exit_code": r.exit_code,
                "duration_s": round(r.duration_s, 3),
                "log_path": str(r.log_path),
                "command": r.command,
            }
            for r in results
        ],
    }
    total = payload["overall"]["total"]
    passed = payload["overall"]["passed"]
    payload["overall"]["fraction"] = passed / total if total else 0.0
    path.write_text(json.dumps(payload, indent=2))


def _write_run_config(
    path: Path,
    *,
    config_path: Path,
    config: BatchConfig,
    python_exe: str,
    dry_run: bool,
) -> None:
    payload = {
        "argv": sys.argv,
        "config_path": str(config_path),
        "python_exe": python_exe,
        "dry_run": dry_run,
        "max_parallel": config.max_parallel,
        "results_root": str(config.results_root),
        "run_autotune2": str(config.run_autotune2),
        "autotune2_args": config.autotune2_args,
        "jobs": [
            {
                "label": job.label,
                "outer_dir": str(job.outer_dir),
                "autotune2_args": job.autotune2_args,
            }
            for job in config.jobs
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


async def _amain(args: argparse.Namespace) -> int:
    config = load_batch_config(args.config)
    out_dir = init_output_dir(config.results_root)
    log_path = out_dir / "autotune2_batch.log"
    redirected = redirect_stdio_to(log_path)
    if redirected:
        terminal_print(f"autotune2 batch log -> {log_path}")
    log = _setup_logging(log_path)
    python_exe = args.python_exe or sys.executable

    _write_run_config(
        out_dir / "config.json",
        config_path=args.config,
        config=config,
        python_exe=python_exe,
        dry_run=args.dry_run,
    )
    log.info("batch output: %s", out_dir)
    log.info("jobs: %d, max_parallel: %d", len(config.jobs), config.max_parallel)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.monotonic()
    if args.dry_run:
        results = [
            BatchResult(
                job=job,
                status="dry-run",
                exit_code=0,
                duration_s=0.0,
                log_path=out_dir / "jobs" / f"{job.label}.log",
                command=build_autotune2_command(
                    outer_dir=job.outer_dir,
                    python_exe=python_exe,
                    run_autotune2=config.run_autotune2,
                    autotune2_args=job.autotune2_args,
                ),
            )
            for job in config.jobs
        ]
    else:
        results = await _run_jobs(
            config=config,
            out_dir=out_dir,
            python_exe=python_exe,
        )
    wall = time.monotonic() - t0
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_summary(
        out_dir / "summary.json",
        results=results,
        started_at=started_at,
        finished_at=finished_at,
        wall_seconds=wall,
        max_parallel=config.max_parallel,
    )

    passed = sum(1 for r in results if r.status in ("pass", "dry-run"))
    log.info("DONE: %d/%d passed in %.1fs", passed, len(results), wall)
    if redirected:
        terminal_print(
            f"autotune2 batch done — {passed}/{len(results)} passed "
            f"in {wall:.1f}s (output: {out_dir})"
        )
    return 0 if passed == len(results) else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Batch YAML path (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--python-exe", default=None,
        help="Python executable used for child run_autotune2.py processes "
             "(default: this interpreter).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Write config/summary and commands without launching children.",
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        setup_process_group()
    except PermissionError:
        pass
    return asyncio.run(_amain(_parse_args(sys.argv[1:])))


if __name__ == "__main__":
    sys.exit(main())
