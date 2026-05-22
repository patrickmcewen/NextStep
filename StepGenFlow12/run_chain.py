#!/usr/bin/env python3
"""Chain run.py -> run_autotune2.py with co-located outputs.

Layout produced::

    <checkpoint-base>/<kernel>__<preset>/
        <run-py-ts>/                       # run.py output
            config.json
            <kernel>/outer_0/
            <kernel>/outer_1/
            ...
        <autotune-ts-0>/                   # autotune2 snapshot of outer_0
            config.json
            <kernel>/outer_0/
            autotune2.log
            autotune2_summary.json
        <autotune-ts-1>/                   # autotune2 snapshot of outer_1
        ...

run.py runs first. After it succeeds, one run_autotune2.py subprocess is
spawned per outer_N in parallel; each writes its snapshot as a sibling of
the run.py output under the same kernel_grouping dir.

Argument forwarding:
  * Shared args (--model, --config, --check-order, --max-tile) are passed
    to both binaries.
  * run.py-only args have no prefix.
  * autotune2-only args are prefixed --at-* so they cannot collide with
    run.py's (and so this script's --help is self-documenting).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_PY = HERE / "run.py"
AUTOTUNE_PY = HERE / "run_autotune2.py"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- top-level / chain-specific ---------------------------------------
    p.add_argument("kernel", help="Kernel name from StepDB (forwarded to both).")
    p.add_argument("preset", help="Preset name (forwarded to both).")
    p.add_argument(
        "--checkpoint-base", default="/workspace/checkpoints/chained",
        help="Parent directory for the kernel_grouping dir. The chain "
             "creates <checkpoint-base>/<kernel>__<preset>/ and aims both "
             "run.py and the autotune2 subprocesses at it so all outputs "
             "land as siblings. Default: /workspace/checkpoints/chained.",
    )
    p.add_argument(
        "--skip-autotune", action="store_true",
        help="Run run.py only; do not spawn autotune2 subprocesses.",
    )
    p.add_argument(
        "--skip-run", action="store_true",
        help="Skip run.py and run autotune2 against an existing run.py "
             "output dir. Requires --run-ts-dir to point at the existing "
             "<run-py-ts> dir.",
    )
    p.add_argument(
        "--run-ts-dir", default=None,
        help="Path to an existing run.py output (a <ts> dir under some "
             "kernel_grouping dir). When set with --skip-run, autotune2 is "
             "spawned for its outer_* dirs and snapshots are written into "
             "that dir's parent. Mutually exclusive with running run.py.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned subprocess commands and exit without "
             "running anything.",
    )

    # --- shared args (forwarded to both run.py and autotune2) -------------
    p.add_argument("--model", default="gpt-oss-120b",
                   help="LLM profile name (configs/<name>.json). Forwarded "
                        "to both run.py and run_autotune2.py.")
    p.add_argument("--config", default=None,
                   help="Explicit LLM config JSON path (overrides --model). "
                        "Forwarded to both.")
    p.add_argument(
        "--check-order", default="correctness-first",
        choices=("correctness-first", "compliance-first", "always-both"),
        help="Gate cascade order. Forwarded to both.",
    )
    p.add_argument(
        "--max-tile", type=int, default=None, metavar="N",
        help="Pass-1 + autotune2 max-tile bound. Forwarded to both (so "
             "autotune2 inherits the same bound the chain pinned for "
             "pass-1). Off by default.",
    )

    # --- run.py-only args -------------------------------------------------
    p.add_argument("--max-outer", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--results-dir", default="/workspace/results")
    p.add_argument("--experience-dir", default="experience")
    p.add_argument("--pipeline", default="standard",
                   choices=("standard", "direct", "direct_no_functional"))
    p.add_argument("--translator", default="auto", choices=("llm", "auto"))
    p.add_argument("--resume", default=None, metavar="PATH")
    p.add_argument("--resume-planner", default=None, metavar="OUTER_DIR")
    p.add_argument("--resume-after-pass1", default=None, metavar="OUTER_DIR")
    p.add_argument("--few-shot", nargs="+", default=None, metavar="PATH")
    p.add_argument("--bundle-dir", default=None, metavar="PATH")
    p.add_argument("--no-plan", action="store_true")
    p.add_argument("--max-replans", type=int, default=1)
    p.add_argument("--node-attempts", type=int, default=4)
    p.add_argument("--max-plan-depth", type=int, default=5)
    p.add_argument("--non-root-sequential", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--stateless-refactor", action="store_true")

    # --- autotune2-only args (--at-* prefix) ------------------------------
    p.add_argument(
        "--at-autotune-config",
        default="/workspace/NextStep/StepGenFlow12/autotune_configs.yaml",
        help="Path to autotune config JSON/YAML (hw_config, max_on_chip_memory, "
             "attempt_budgets, optional passes:[...]). Default: "
             "/workspace/NextStep/StepGenFlow12/autotune_configs.yaml.",
    )
    p.add_argument(
        "--at-autotune-config-name",
        default="autotune_config_2",
        help="Named config to resolve when --at-autotune-config is YAML.",
    )
    p.add_argument("--at-max-turns-per-attempt", type=int, default=16)
    p.add_argument("--at-root-pick", default="final_pick",
                   choices=("final_pick", "top_k", "agent"))
    p.add_argument("--at-top-k", type=int, default=3)
    p.add_argument("--at-compute-bw", type=int, default=100_000)
    p.add_argument("--at-include-sources", action="store_true")
    p.add_argument("--at-fewshot", default="tile_shrink",
                   choices=("tile_shrink", "parallel"))
    p.add_argument("--at-sim-mode", default="analytical",
                   choices=("analytical", "rust", "deterministic-split", "agent"))
    p.add_argument("--at-sim-calibration-path", default=None)
    p.add_argument("--at-curation-max-candidates", type=int, default=50)
    p.add_argument("--at-curation-k", type=int, default=4)
    p.add_argument("--at-rust-functional-check",
                   action=argparse.BooleanOptionalAction, default=False)

    return p


def _build_run_cmd(args: argparse.Namespace, grouping: Path) -> list[str]:
    """Translate parsed args into a run.py argv."""
    cmd = [sys.executable, str(RUN_PY), args.kernel, args.preset]
    cmd += ["--checkpoint-dir", str(grouping)]

    # shared
    cmd += ["--model", args.model]
    if args.config is not None:
        cmd += ["--config", args.config]
    cmd += ["--check-order", args.check_order]
    if args.max_tile is not None:
        cmd += ["--max-tile", str(args.max_tile)]

    # run.py-only
    cmd += ["--max-outer", str(args.max_outer)]
    cmd += ["--max-turns", str(args.max_turns)]
    cmd += ["--results-dir", args.results_dir]
    cmd += ["--experience-dir", args.experience_dir]
    cmd += ["--pipeline", args.pipeline]
    cmd += ["--translator", args.translator]
    if args.resume is not None:
        cmd += ["--resume", args.resume]
    if args.resume_planner is not None:
        cmd += ["--resume-planner", args.resume_planner]
    if args.resume_after_pass1 is not None:
        cmd += ["--resume-after-pass1", args.resume_after_pass1]
    if args.few_shot:
        cmd += ["--few-shot", *args.few_shot]
    if args.bundle_dir is not None:
        cmd += ["--bundle-dir", args.bundle_dir]
    if args.no_plan:
        cmd += ["--no-plan"]
    cmd += ["--max-replans", str(args.max_replans)]
    cmd += ["--node-attempts", str(args.node_attempts)]
    cmd += ["--max-plan-depth", str(args.max_plan_depth)]
    cmd += ["--non-root-sequential" if args.non_root_sequential
            else "--no-non-root-sequential"]
    if args.no_judge:
        cmd += ["--no-judge"]
    if args.stateless_refactor:
        cmd += ["--stateless-refactor"]
    return cmd


def _build_autotune_cmd(
    args: argparse.Namespace, grouping: Path, outer_dir: Path,
) -> list[str]:
    """Translate parsed args into a run_autotune2.py argv for one outer_N."""
    cmd = [sys.executable, str(AUTOTUNE_PY), str(outer_dir)]
    cmd += ["--checkpoint-dir", str(grouping)]
    cmd += ["--kernel", args.kernel, "--preset", args.preset]

    # shared
    cmd += ["--model", args.model]
    if args.config is not None:
        cmd += ["--config", args.config]
    cmd += ["--check-order", args.check_order]
    if args.max_tile is not None:
        cmd += ["--max-tile", str(args.max_tile)]

    # autotune2-only
    cmd += ["--autotune-config", args.at_autotune_config]
    cmd += ["--autotune-config-name", args.at_autotune_config_name]
    cmd += ["--max-turns-per-attempt", str(args.at_max_turns_per_attempt)]
    cmd += ["--root-pick", args.at_root_pick]
    cmd += ["--top-k", str(args.at_top_k)]
    cmd += ["--compute-bw", str(args.at_compute_bw)]
    if args.at_include_sources:
        cmd += ["--include-sources"]
    cmd += ["--fewshot", args.at_fewshot]
    cmd += ["--sim-mode", args.at_sim_mode]
    if args.at_sim_calibration_path is not None:
        cmd += ["--sim-calibration-path", args.at_sim_calibration_path]
    cmd += ["--curation-max-candidates", str(args.at_curation_max_candidates)]
    cmd += ["--curation-k", str(args.at_curation_k)]
    cmd += ["--rust-functional-check" if args.at_rust_functional_check
            else "--no-rust-functional-check"]
    return cmd


def _identify_new_ts_dir(grouping: Path, before: set[str]) -> Path:
    """Return the single new <ts> child added under `grouping` since `before`."""
    after = {p.name for p in grouping.iterdir() if p.is_dir()}
    new_dirs = sorted(after - before)
    assert len(new_dirs) == 1, (
        f"expected exactly one new dir under {grouping} after run.py, "
        f"got {new_dirs}. Other writers may be racing on this base dir."
    )
    return grouping / new_dirs[0]


def main() -> int:
    args = build_parser().parse_args()

    assert not (args.skip_run and not args.run_ts_dir), (
        "--skip-run requires --run-ts-dir pointing at an existing run.py "
        "output (a <ts> dir under some kernel_grouping)."
    )
    assert not (args.skip_run and args.skip_autotune), (
        "--skip-run and --skip-autotune together leave nothing to do."
    )

    grouping = Path(args.checkpoint_base) / f"{args.kernel}__{args.preset}"
    grouping.mkdir(parents=True, exist_ok=True)

    # ----- run.py phase ---------------------------------------------------
    if args.skip_run:
        run_ts_dir = Path(args.run_ts_dir).resolve()
        assert run_ts_dir.is_dir(), f"--run-ts-dir not found: {run_ts_dir}"
        # Use the existing dir's parent as the grouping for autotune2
        # snapshots; that keeps siblings co-located with the source run.
        grouping = run_ts_dir.parent
        print(f"chain: --skip-run; using existing run output at {run_ts_dir}")
        print(f"chain: autotune2 snapshots will be written under {grouping}")
    else:
        run_cmd = _build_run_cmd(args, grouping)
        print(f"chain: kernel_grouping = {grouping}")
        print(f"chain: launching run.py")
        print(f"  $ {shlex.join(run_cmd)}")
        if args.dry_run:
            run_ts_dir = grouping / "<dry-run-ts>"
        else:
            before = {p.name for p in grouping.iterdir() if p.is_dir()}
            rc = subprocess.call(run_cmd)
            if rc != 0:
                print(f"chain: run.py exited rc={rc}; skipping autotune phase")
                return rc
            run_ts_dir = _identify_new_ts_dir(grouping, before)
            print(f"chain: run.py output -> {run_ts_dir}")

    if args.skip_autotune:
        print("chain: --skip-autotune set; done")
        return 0

    # ----- autotune2 phase ------------------------------------------------
    kernel_dir = run_ts_dir / args.kernel
    if args.dry_run and not args.skip_run:
        # We don't actually know the outer dirs yet; show one stub command.
        stub_outer = kernel_dir / "outer_<N>"
        at_cmd = _build_autotune_cmd(args, grouping, stub_outer)
        print(f"chain: would spawn one autotune2 per outer_N, e.g.:")
        print(f"  $ {shlex.join(at_cmd)}")
        return 0

    assert kernel_dir.is_dir(), (
        f"expected kernel dir at {kernel_dir} (run.py output layout is "
        f"<ts>/<kernel>/outer_N/; did run.py finish?)"
    )
    outers = sorted(
        p for p in kernel_dir.iterdir()
        if p.is_dir() and p.name.startswith("outer_")
    )
    assert outers, f"no outer_* dirs found under {kernel_dir}"
    print(f"chain: found {len(outers)} outer_* dirs: "
          f"{[o.name for o in outers]}; spawning autotune2 in parallel")

    procs: list[tuple[str, subprocess.Popen, Path]] = []
    for outer in outers:
        at_cmd = _build_autotune_cmd(args, grouping, outer)
        # Per-outer chain-level log under the grouping dir. The autotune
        # subprocess's own log_redirect no-ops (its stdout is piped here),
        # so this captures everything it would have written.
        log_path = grouping / f"chain_autotune_{outer.name}.log"
        print(f"chain: [{outer.name}] -> {log_path}")
        print(f"  $ {shlex.join(at_cmd)}")
        if args.dry_run:
            continue
        log_fh = open(log_path, "w", buffering=1)
        proc = subprocess.Popen(
            at_cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        )
        procs.append((outer.name, proc, log_path))

    if args.dry_run:
        return 0

    failures: list[tuple[str, int]] = []
    for name, proc, log_path in procs:
        rc = proc.wait()
        proc.stdout.close() if proc.stdout is not None else None
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"chain: [{name}] {status} — log: {log_path}")
        if rc != 0:
            failures.append((name, rc))

    if failures:
        print(f"chain: {len(failures)}/{len(outers)} autotune2 runs failed: "
              f"{failures}")
        return 1
    print(f"chain: all {len(outers)} autotune2 runs OK; outputs under "
          f"{grouping}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
