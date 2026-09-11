"""Snapshot an outer_* dir into the translator regression suite.

Copies only the files the orchestrator actually consumes under
`--resume-after-pass1`:
  * plan/iteration_*/tree.json
  * plan/iteration_*/<node>/reference.py
  * plan/iteration_*/<node>/refactored.py        (when present)
  * pass1/iteration_*/.../refactor_final/turn_*/status.txt
  * pass1/iteration_*/.../refactor_final/turn_*/extracted_code.py
Plus the source `dsl_code.py` as a human-readable reference.

Pruning per fixture is ~700x (213MB -> ~300KB) so snapshots are
cheap enough to live alongside the translator source.

Usage:
    python add_fixture.py NAME \
        --kernel KERNEL --preset PRESET \
        --source /workspace/checkpoints/.../outer_X \
        [--notes "..."] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_ESSENTIAL_FILES = frozenset({
    "tree.json", "reference.py", "refactored.py",
    "status.txt", "extracted_code.py",
})


def _snapshot(src_outer: Path, dest: Path) -> None:
    for sub in ("plan", "pass1"):
        src_sub = src_outer / sub
        assert src_sub.is_dir(), f"source missing {sub}/: {src_outer}"
        for path in src_sub.rglob("*"):
            if path.is_file() and path.name in _ESSENTIAL_FILES:
                rel = path.relative_to(src_outer)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
    dsl = src_outer / "dsl_code.py"
    if dsl.is_file():
        shutil.copy(dsl, dest / "dsl_code.py")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", help="Fixture name (used as directory under fixtures/).")
    p.add_argument("--kernel", required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--source", required=True, type=Path,
                   help="Source outer_* directory.")
    p.add_argument("--notes", default="",
                   help="Free-text description; surfaces in `run.py --list`.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing fixture with the same name.")
    args = p.parse_args()

    src = args.source.resolve()
    assert src.is_dir(), f"--source must be a directory: {src}"

    dest = FIXTURES_DIR / args.name
    if dest.exists():
        assert args.overwrite, (
            f"fixture {args.name!r} already exists at {dest} (pass --overwrite)"
        )
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    _snapshot(src, dest)

    meta = {
        "kernel": args.kernel,
        "preset": args.preset,
        "source": str(src),
        "notes": args.notes,
    }
    with (dest / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    size_kb = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1024
    print(f"snapshot -> {dest} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
