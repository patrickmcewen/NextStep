"""Migrate an autotune2 calibration.jsonl into the StepDB global store.

Each input record points at a composed_source.py inside the producing
run's checkpoint dir. That source dir is not stable — once a regression-
results sweep gets cleaned up, the records become dead weight (the
curation agent silently skips records whose composed_source_path no
longer exists). This script copies the referenced sources into a
sources dir co-located with the global JSONL and rewrites the path on
the way through, so the global store survives independent of the
producing runs.

Usage
-----
    python migrate_calibration.py <input.jsonl> [<input.jsonl> ...]
        [--store /workspace/NextStep/StepDB/calibration.jsonl]

Then point autotune2 at the global store:

    python run_autotune2.py ... --sim-calibration-path \\
        /workspace/NextStep/StepDB/calibration.jsonl

De-duplication
--------------
A record is keyed by (run_id, timestamp, node_path), which the rust-
evaluator code path produces uniquely per evaluation. Re-running the
migration over the same input is therefore a no-op.

Schema coupling
---------------
The required-field list below mirrors ``CalibrationRecord`` in
``StepGenFlow12/src/autotune2/calibration.py``. We mirror rather than
import so StepDB stays decoupled from StepGenFlow12 — if the schema
gains/loses a field, this script will assert loudly and you update both
sides together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

STEPDB_ROOT = Path(__file__).resolve().parent
DEFAULT_STORE = STEPDB_ROOT / "calibration.jsonl"

# Mirror of CalibrationRecord fields. Keep in sync with
# StepGenFlow12/src/autotune2/calibration.py.
REQUIRED_FIELDS = frozenset({
    "node_path", "is_root", "kernel", "preset", "composed_source_path",
    "analytical_cycles", "analytical_on_chip", "rust_cycles", "rust_dur_ms",
    "hw_config_hash", "compute_bw", "timestamp", "run_id", "error_pct",
})


def _existing_keys(store_path: Path) -> set[tuple[str, str, str]]:
    """Read all (run_id, timestamp, node_path) keys already in the store."""
    if not store_path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    with store_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            keys.add((r["run_id"], r["timestamp"], r["node_path"]))
    return keys


def migrate_file(
    input_path: Path, store_path: Path, sources_dir: Path,
    existing: set[tuple[str, str, str]],
) -> tuple[int, int]:
    """Migrate one input JSONL into the global store.

    Returns ``(migrated, skipped)``. ``existing`` is mutated in place so
    multi-file runs de-dup against records added earlier in the same
    invocation, not only against the on-disk store.
    """
    assert input_path.exists(), f"input does not exist: {input_path}"
    migrated = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as f_in, \
         store_path.open("a", encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = REQUIRED_FIELDS - rec.keys()
            assert not missing, (
                f"{input_path}:{line_no}: record missing required fields "
                f"{sorted(missing)} — schema drift vs CalibrationRecord. "
                f"Update REQUIRED_FIELDS in this script."
            )
            extra = rec.keys() - REQUIRED_FIELDS
            assert not extra, (
                f"{input_path}:{line_no}: record has unknown fields "
                f"{sorted(extra)} — schema drift vs CalibrationRecord. "
                f"Update REQUIRED_FIELDS in this script."
            )

            key = (rec["run_id"], rec["timestamp"], rec["node_path"])
            if key in existing:
                skipped += 1
                continue

            src = Path(rec["composed_source_path"])
            assert src.exists(), (
                f"{input_path}:{line_no}: composed_source_path is gone: "
                f"{src}. The producing run's checkpoint was likely cleaned "
                f"up before migration."
            )
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            assert src.stem == digest, (
                f"{input_path}:{line_no}: filename stem {src.stem!r} does "
                f"not match sha256 of contents {digest!r}. Record was not "
                f"produced by autotune2's _write_calibration_source."
            )

            dst = sources_dir / f"{digest}.py"
            if not dst.exists():
                shutil.copyfile(src, dst)

            rec["composed_source_path"] = str(dst)
            out_line = json.dumps(rec, sort_keys=True) + "\n"
            assert len(out_line.encode("utf-8")) < 4096, (
                f"{input_path}:{line_no}: rewritten record is "
                f"{len(out_line)} bytes, exceeds the 4096-byte atomic-"
                f"append limit. CalibrationStore.append would reject it."
            )
            f_out.write(out_line)
            existing.add(key)
            migrated += 1
    return migrated, skipped


def main() -> None:
    p = argparse.ArgumentParser(
        description="Migrate autotune2 calibration.jsonl into the StepDB "
                    "global store, copying composed sources alongside."
    )
    p.add_argument(
        "inputs", nargs="+", type=Path,
        help="One or more existing calibration.jsonl files to migrate.",
    )
    p.add_argument(
        "--store", type=Path, default=DEFAULT_STORE,
        help=f"Destination JSONL (default: {DEFAULT_STORE}). Sources are "
             f"copied into <store>.parent/calibration_sources/.",
    )
    args = p.parse_args()

    store_path: Path = args.store
    sources_dir = store_path.parent / "calibration_sources"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_keys(store_path)
    total_mig = 0
    total_skip = 0
    for inp in args.inputs:
        mig, skip = migrate_file(inp, store_path, sources_dir, existing)
        print(f"{inp}: migrated {mig}, skipped {skip}")
        total_mig += mig
        total_skip += skip
    print(f"--- total: migrated {total_mig}, skipped {total_skip}")
    print(f"--- store: {store_path}")
    print(f"--- sources: {sources_dir}")


if __name__ == "__main__":
    main()
