"""Chain run_regression.py -> run.py, feeding passing regression outputs as few-shot.

Workflow:
  1. Run run_regression.py with the user-supplied options. The regression
     writes its run dir under --results-root as <YYYYMMDD-HHMMSS>/.
  2. Read summary.json from that run dir to discover which (kernel, preset)
     pairs passed.
  3. For each passing benchmark, locate
        <run-dir>/checkpoints/<kernel>__<preset>/*/<kernel>/outer_*/dsl_code.py
     (using the highest outer_N if more than one exists).
  4. Invoke run.py on the downstream <kernel> <preset> with each path supplied
     via --few-shot.

The wrapper does not duplicate argparse from run.py / run_regression.py.
Forward extra options with --reg-args / --run-args (shell-quoted, parsed via
shlex), e.g.:

  python run_chained.py rope qwen_b64 \\
      --reg-args="--subset just_the_subparts --model gpt-oss-120b" \\
      --run-args="--model gpt-oss-120b --max-outer 4 --check-order always-both"

Use --skip-regression DIR to reuse an existing regression run without re-running
it (handy when iterating on the downstream call).
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
REGRESSION_SCRIPT = REPO_ROOT / "run_regression.py"
RUN_SCRIPT = REPO_ROOT / "run.py"
DEFAULT_RESULTS_ROOT = Path("/workspace/regression_results")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("kernel", help="Kernel for the downstream run.py call.")
    p.add_argument("preset", help="Preset for the downstream run.py call.")
    p.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help="Directory where run_regression.py writes its timestamped run "
             "dirs. Must match the --results-root forwarded inside --reg-args "
             "(if you override it there).",
    )
    p.add_argument(
        "--skip-regression", type=Path, default=None, metavar="RUN_DIR",
        help="Skip step 1 and reuse this existing regression run dir "
             "(e.g. /workspace/regression_results/20260512-062335).",
    )
    p.add_argument(
        "--require-all-pass", action="store_true",
        help="Abort before invoking run.py if any benchmark in the regression "
             "failed. Default: warn and proceed with the passing subset.",
    )
    p.add_argument(
        "--reg-args", default="",
        help="Extra args forwarded to run_regression.py, shell-quoted "
             "(parsed with shlex). Ignored when --skip-regression is set.",
    )
    p.add_argument(
        "--run-args", default="",
        help="Extra args forwarded to run.py, shell-quoted (parsed with shlex). "
             "kernel/preset and --few-shot are appended by this wrapper.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved few-shot paths and the run.py command, then exit.",
    )
    return p.parse_args(argv)


def _snapshot_run_dirs(results_root: Path) -> set[Path]:
    if not results_root.is_dir():
        return set()
    return {d for d in results_root.iterdir() if d.is_dir()}


def _run_regression(reg_extra: list[str], results_root: Path) -> Path:
    """Invoke run_regression.py and return the freshly-created run dir."""
    assert REGRESSION_SCRIPT.is_file(), f"missing {REGRESSION_SCRIPT}"
    before = _snapshot_run_dirs(results_root)
    cmd = [sys.executable, str(REGRESSION_SCRIPT), *reg_extra]
    print(f"[chained] launching regression: {shlex.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    assert rc == 0, f"run_regression.py exited with {rc}"

    after = _snapshot_run_dirs(results_root)
    new_dirs = sorted(after - before)
    assert len(new_dirs) == 1, (
        f"expected exactly one new run dir under {results_root}, "
        f"found {len(new_dirs)}: {new_dirs}"
    )
    return new_dirs[0]


def _collect_few_shot_paths(run_dir: Path, require_all_pass: bool) -> list[Path]:
    """Read summary.json and return one dsl_code.py per passing benchmark."""
    summary_path = run_dir / "summary.json"
    assert summary_path.is_file(), f"missing summary at {summary_path}"
    summary = json.loads(summary_path.read_text())

    benchmarks = summary.get("benchmarks") or {}
    assert benchmarks, f"no 'benchmarks' section in {summary_path}"

    passing: list[tuple[str, str]] = []
    failing: list[tuple[str, str]] = []
    for kernel, kdata in benchmarks.items():
        for preset, pdata in (kdata.get("presets") or {}).items():
            if pdata.get("status") == "pass":
                passing.append((kernel, preset))
            else:
                failing.append((kernel, preset))

    if failing:
        msg = f"[chained] {len(failing)} benchmark(s) did not pass: {failing}"
        if require_all_pass:
            raise SystemExit(msg + " (aborting because --require-all-pass was set)")
        print(msg + " — proceeding with the passing subset.", flush=True)

    assert passing, f"no passing benchmarks in {summary_path}; nothing to few-shot from"

    checkpoints_root = run_dir / "checkpoints"
    paths: list[Path] = []
    for kernel, preset in passing:
        bench_root = checkpoints_root / f"{kernel}__{preset}"
        candidates = sorted(bench_root.glob(f"*/{kernel}/outer_*/dsl_code.py"))
        assert candidates, (
            f"benchmark {kernel}/{preset} marked pass but no dsl_code.py found "
            f"under {bench_root}"
        )
        # Highest outer_N (last in sort order) is the one that produced the
        # final verified DSL.
        paths.append(candidates[-1])
    return paths


def _run_downstream(
    kernel: str,
    preset: str,
    run_extra: list[str],
    few_shot_paths: list[Path],
    dry_run: bool,
) -> int:
    assert RUN_SCRIPT.is_file(), f"missing {RUN_SCRIPT}"
    cmd = [
        sys.executable, str(RUN_SCRIPT),
        kernel, preset,
        *run_extra,
        "--few-shot", *[str(p) for p in few_shot_paths],
    ]
    print(f"[chained] launching downstream run: {shlex.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    reg_extra = shlex.split(args.reg_args)
    run_extra = shlex.split(args.run_args)

    if args.skip_regression is not None:
        run_dir = args.skip_regression
        assert run_dir.is_dir(), f"--skip-regression dir not found: {run_dir}"
        print(f"[chained] reusing existing regression run: {run_dir}", flush=True)
    else:
        run_dir = _run_regression(reg_extra, args.results_root)
        print(f"[chained] regression run dir: {run_dir}", flush=True)

    few_shot_paths = _collect_few_shot_paths(run_dir, args.require_all_pass)
    print(f"[chained] {len(few_shot_paths)} few-shot example(s):", flush=True)
    for p in few_shot_paths:
        print(f"  - {p}", flush=True)

    return _run_downstream(args.kernel, args.preset, run_extra, few_shot_paths, args.dry_run)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
