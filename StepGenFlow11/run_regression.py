# run_regression.py
"""Regression runner: drive run.py over many (kernel, preset) jobs in parallel."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.regression_planning import load_bench_config, plan_jobs
from src.regression_runner import (
    build_run_py_command,
    init_output_dir,
    read_autotune,
    read_per_outer,
    read_total_tokens,
    run_jobs,
    run_subprocess,
    setup_logging,
    write_summary,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCH_CONFIG = Path("/workspace/DEIOpt/StepDB/bench_config.yaml")
DEFAULT_RUN_PY = REPO_ROOT / "run.py"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preset-config", type=Path, default=None,
                      help="YAML mapping kernel -> preset (or list of presets).")
    mode.add_argument("--all-presets", action="store_true",
                      help="Run every preset of every kernel in bench_config.yaml.")
    mode.add_argument("--subset", default=None, metavar="NAME",
                      help="Named group from a multi-group subset YAML file.")

    p.add_argument("--subset-file", type=Path, default=REPO_ROOT / "regression_subsets.yaml",
                   help="Path to the subset YAML file (used with --subset).")
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--results-root", type=Path, default=Path("/workspace/regression_results"))
    p.add_argument("--bench-config", type=Path, default=DEFAULT_BENCH_CONFIG,
                   help="Path to StepDB bench_config.yaml (override for tests).")
    p.add_argument("--run-py", type=Path, default=DEFAULT_RUN_PY,
                   help="Path to run.py (override for tests).")

    # Pass-through to run.py
    p.add_argument("--model", default="gpt-oss-120b")
    p.add_argument("--config", default=None)
    p.add_argument("--max-outer", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--pipeline", default=None, choices=[None, "standard", "direct", "direct_no_functional"])
    p.add_argument("--translator", default=None, choices=[None, "auto", "llm"])
    p.add_argument("--bundle-dir", default=None, metavar="PATH",
                   help="Bundle directory passed through to each per-job run.py invocation.")

    # Autotune pass-through to run.py
    p.add_argument("--autotune", action="store_true",
                   help="Enable per-outer autotune in each run.py invocation.")
    p.add_argument("--autotune-config", default="autotune_config.json",
                   help="Pass-through path to autotune_config JSON.")
    p.add_argument("--autotune-max-turns", type=int, default=None,
                   help="Pass-through override for autotune max_turns.")
    p.add_argument("--autotune-agent", default=None, choices=[None, "general", "parallel"],
                   help="Pass-through autotune agent variant.")
    p.add_argument("--check-order", default=None,
                   choices=[None, "correctness-first", "compliance-first"],
                   help="Pass-through gate ordering for each per-job run.py invocation.")
    return p.parse_args(argv)


def _load_preset_config(path: Path | None) -> dict | None:
    if path is None:
        return None
    assert path.exists(), f"--preset-config path not found: {path}"
    with open(path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "--preset-config must be a YAML mapping"
    return data


def _load_subset(subset_file: Path, subset_name: str) -> dict:
    assert subset_file.exists(), f"--subset-file path not found: {subset_file}"
    with open(subset_file) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"--subset-file must be a YAML mapping: {subset_file}"
    assert subset_name in data, (
        f"subset {subset_name!r} not found in {subset_file}; "
        f"available: {sorted(data.keys())}"
    )
    subset = data[subset_name]
    assert isinstance(subset, dict) and subset, (
        f"subset {subset_name!r} must be a non-empty mapping (kernel -> preset(s))"
    )
    return subset


async def _amain(args: argparse.Namespace) -> int:
    bench_config = load_bench_config(args.bench_config)
    if args.subset:
        preset_config = _load_subset(args.subset_file, args.subset)
        all_presets = False
    else:
        preset_config = _load_preset_config(args.preset_config)
        all_presets = args.all_presets
    jobs = plan_jobs(bench_config, preset_config=preset_config, all_presets=all_presets)

    out_dir = init_output_dir(args.results_root)
    checkpoints_root = out_dir / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "regression.log")

    if args.subset:
        preset_mode = f"subset:{args.subset}"
    elif args.all_presets:
        preset_mode = "all_presets"
    else:
        preset_mode = "preset_config"
    log.info("regression run output: %s", out_dir)
    log.info("jobs: %d, max_parallel: %d, mode: %s", len(jobs), args.max_parallel, preset_mode)

    (out_dir / "config.json").write_text(json.dumps({
        "argv": sys.argv,
        "preset_mode": preset_mode,
        "max_parallel": args.max_parallel,
        "model": args.model,
        "config": args.config,
        "max_outer": args.max_outer,
        "max_turns": args.max_turns,
        "pipeline": args.pipeline,
        "translator": args.translator,
        "check_order": args.check_order,
        "autotune": args.autotune,
        "autotune_config": args.autotune_config,
        "autotune_max_turns": args.autotune_max_turns,
        "autotune_agent": args.autotune_agent,
        "bench_config": str(args.bench_config),
        "run_py": str(args.run_py),
        "subset_file": str(args.subset_file),
        "jobs": [{"kernel": j.kernel, "preset": j.preset} for j in jobs],
    }, indent=2))

    async def run_one(job, log_path):
        job_ckpt_dir = checkpoints_root / f"{job.kernel}__{job.preset}"
        cmd = build_run_py_command(
            job,
            python_exe=sys.executable,
            run_py_path=args.run_py,
            model=args.model,
            config=args.config,
            max_outer=args.max_outer,
            max_turns=args.max_turns,
            pipeline=args.pipeline,
            translator=args.translator,
            checkpoint_dir=job_ckpt_dir,
            bundle_dir=args.bundle_dir,
            autotune=args.autotune,
            autotune_config=args.autotune_config,
            autotune_max_turns=args.autotune_max_turns,
            autotune_agent=args.autotune_agent,
            check_order=args.check_order,
        )
        exit_code, duration = await run_subprocess(cmd, log_path, cwd=REPO_ROOT)
        outer_passed, outer_total = read_per_outer(job_ckpt_dir)
        total_tokens = read_total_tokens(job_ckpt_dir)
        autotune = read_autotune(job_ckpt_dir)
        return exit_code, duration, outer_passed, outer_total, total_tokens, autotune

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = asyncio.get_event_loop().time()
    results = await run_jobs(
        jobs,
        max_parallel=args.max_parallel,
        jobs_dir=out_dir / "jobs",
        run_one=run_one,
    )
    wall = asyncio.get_event_loop().time() - t0
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    write_summary(
        out_dir / "summary.json",
        results=results,
        started_at=started_at,
        finished_at=finished_at,
        wall_seconds=wall,
        max_parallel=args.max_parallel,
        model=args.model,
        preset_mode=preset_mode,
    )

    overall_passed = sum(1 for r in results if r.status == "pass")
    log.info(
        "DONE: %d/%d passed (%.1f%%) in %.1fs",
        overall_passed, len(results),
        100.0 * overall_passed / len(results) if results else 0.0,
        wall,
    )
    return 0


def main() -> int:
    args = _parse_args(sys.argv[1:])
    assert args.max_parallel >= 1, "--max-parallel must be >= 1"
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
