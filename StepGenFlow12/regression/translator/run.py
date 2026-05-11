"""Replay translator regression fixtures via `--resume-after-pass1`.

Each fixture under `fixtures/<name>/` carries a pruned snapshot of an
outer_* directory plus a `metadata.yaml` with the kernel/preset to invoke.
We shell out to the project's `run.py` per fixture; pass = exit 0.

Usage:
    python regression/translator/run.py                 # run all fixtures
    python regression/translator/run.py FIXTURE_NAME    # run specific one(s)
    python regression/translator/run.py --list          # show available fixtures
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
FIXTURES_DIR = HERE / "fixtures"
RUNS_DIR = HERE / "runs"
RUN_PY = REPO_ROOT / "run.py"


def _load_metadata(fixture: Path) -> dict:
    meta_path = fixture / "metadata.json"
    assert meta_path.is_file(), (
        f"fixture {fixture.name!r}: missing metadata.json at {meta_path}"
    )
    with meta_path.open() as f:
        meta = json.load(f)
    assert isinstance(meta, dict), (
        f"fixture {fixture.name!r}: metadata.json must be a JSON object"
    )
    for k in ("kernel", "preset"):
        assert k in meta, f"fixture {fixture.name!r}: metadata.json missing {k!r}"
    return meta


def _discover_fixtures(names: list[str] | None) -> list[Path]:
    assert FIXTURES_DIR.is_dir(), f"no fixtures dir at {FIXTURES_DIR}"
    all_fixtures = sorted(p for p in FIXTURES_DIR.iterdir() if p.is_dir())
    if not names:
        return all_fixtures
    by_name = {p.name: p for p in all_fixtures}
    missing = [n for n in names if n not in by_name]
    assert not missing, (
        f"unknown fixture(s): {missing}; available: {sorted(by_name)}"
    )
    return [by_name[n] for n in names]


def _run_one(fixture: Path, run_dir: Path) -> dict:
    meta = _load_metadata(fixture)
    out_dir = run_dir / fixture.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    ckpt_dir = out_dir / "checkpoints"

    cmd = [
        sys.executable, str(RUN_PY),
        meta["kernel"], meta["preset"],
        "--resume-after-pass1", str(fixture.resolve()),
        "--checkpoint-dir", str(ckpt_dir),
        "--max-outer", "1",
    ]
    t0 = time.monotonic()
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    duration = time.monotonic() - t0
    return {
        "fixture": fixture.name,
        "kernel": meta["kernel"],
        "preset": meta["preset"],
        "status": "pass" if proc.returncode == 0 else "fail",
        "exit_code": proc.returncode,
        "duration_s": round(duration, 1),
        "log": str(log_path.relative_to(REPO_ROOT)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fixtures", nargs="*",
                   help="Specific fixture name(s); omit to run all.")
    p.add_argument("--list", action="store_true",
                   help="List fixtures (name / kernel / preset / notes) and exit.")
    args = p.parse_args()

    if args.list:
        for f in _discover_fixtures(None):
            m = _load_metadata(f)
            print(f"{f.name}\t{m['kernel']}\t{m['preset']}\t{m.get('notes', '')}")
        return 0

    fixtures = _discover_fixtures(args.fixtures or None)
    assert fixtures, f"no fixtures found in {FIXTURES_DIR}"

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"running {len(fixtures)} fixture(s) -> {run_dir}")

    results = []
    for f in fixtures:
        print(f"  [{f.name}] ...", flush=True)
        r = _run_one(f, run_dir)
        results.append(r)
        print(f"    -> {r['status']} (exit={r['exit_code']}, "
              f"{r['duration_s']}s, log={r['log']})")

    (run_dir / "summary.json").write_text(json.dumps({"results": results}, indent=2))

    passed = sum(1 for r in results if r["status"] == "pass")
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
