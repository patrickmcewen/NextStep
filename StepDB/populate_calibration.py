"""Populate the StepDB calibration store from validate_timing runs.

For each (kernel, preset) pair this script builds the STeP graph from
the seed ``step_impl.py``, runs the analytical timing model, runs the
rust cycle-accurate simulator, and appends one ``CalibrationRecord``
JSONL line to the global calibration store. The seed ``step_impl.py``
is copied into ``<store>.parent/calibration_sources/<sha256>.py`` so
the record stays valid even if the seed kernel is later edited.

Usage
-----
    python populate_calibration.py gemm small
    python populate_calibration.py gemm                  # all gemm presets
    python populate_calibration.py --all-small -j 8
    python populate_calibration.py --all -j 8
    python populate_calibration.py gemm small \\
        --store /workspace/NextStep/StepDB/calibration.jsonl \\
        --autotune-config /workspace/NextStep/StepGenFlow12/autotune_config.json \\
        --compute-bw 100000

Source form caveat
------------------
Records produced by autotune2 store *DSL* sources (the LLM-generated
``tiled_reference``). Seed ``step_impl.py`` is in *STeP IR* form. This
script writes the IR source as-is, so the curation agent will see a
mix of DSL and IR sources in the global store. The cycle pair is still
ground truth; the LLM can distinguish source form by signature
(``def build_graph`` vs ``def tiled_reference``). If you need IR→DSL
conversion build that separately.

hw_config_hash
--------------
``CalibrationRecord.hw_config_hash`` filters records on the consumer
side; it must match the hash autotune2 produces from its own
``hw_config`` block. Defaults to the 5-key HBM dict autotune2 reads
out of ``autotune_config.json`` (yielding ``4b414022bff4a4f4`` for the
in-tree configs) — *not* ``DEFAULT_HW_CONFIG`` which carries two extra
keys and hashes differently. Pass ``--autotune-config`` to point at
the same JSON your autotune2 runs use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import sympy
import yaml

STEPDB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STEPDB_DIR))

from validate_timing import (  # noqa: E402
    build_graph_from_impl,
    compute_memory_totals,
    load_config,
    normalize_compute_bw,
    run_analytical_model,
    run_simulator,
)
from timing_and_emulator.timing import DEFAULT_HW_CONFIG  # noqa: E402

DEFAULT_STORE = STEPDB_DIR / "calibration.jsonl"
DEFAULT_AUTOTUNE_CONFIG = (
    STEPDB_DIR.parent / "StepGenFlow12" / "autotune_configs.yaml"
)
DEFAULT_AUTOTUNE_CONFIG_NAME = "autotune_config_2"

# Mirrors CalibrationRecord field set in
# StepGenFlow12/src/autotune2/calibration.py. Kept in sync with
# migrate_calibration.py — see that script's docstring.
REQUIRED_FIELDS = frozenset({
    "node_path", "is_root", "kernel", "preset", "composed_source_path",
    "analytical_cycles", "analytical_on_chip", "rust_cycles", "rust_dur_ms",
    "hw_config_hash", "compute_bw", "timestamp", "run_id", "error_pct",
})


def _hw_config_hash(hw_config: dict) -> str:
    """Mirror of ``calibration.hw_config_hash`` — stable 16-hex digest."""
    blob = json.dumps(hw_config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _merge_recursive(base: dict, override: dict) -> dict:
    """Deep-merge ``override`` into ``base``. Pure data, no mutation."""
    from copy import deepcopy
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_recursive(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_yaml_config(configs: dict, name: str) -> dict:
    """Resolve one named config from a YAML ``configs:`` block, applying
    ``base`` entries recursively. Mirrors
    ``StepGenFlow12/src/autotune_config_loader.resolve_config`` so the
    script reads the same hw_config autotune2 would see; we don't import
    it to keep StepDB decoupled from StepGenFlow12."""
    assert name in configs, (
        f"autotune config {name!r} not found; available: {sorted(configs)}"
    )
    raw = configs[name]
    bases = raw.get("base", [])
    if isinstance(bases, str):
        bases = [bases]
    resolved: dict = {}
    for base_name in bases:
        assert base_name != name, (
            f"autotune config {name!r}: config cannot inherit from itself"
        )
        resolved = _merge_recursive(resolved, _resolve_yaml_config(configs, base_name))
    override = {k: v for k, v in raw.items() if k != "base"}
    return _merge_recursive(resolved, override)


def _load_hw_config(
    autotune_config_path: Path | None, config_name: str | None,
) -> dict:
    """Load and return the ``hw_config`` block.

    Supports both the legacy single-config JSON shape and the new
    inheritance YAML shape with a named entry. When loading YAML, the
    caller must pass ``config_name`` (or set a ``default:`` key in the
    YAML), since the file holds multiple configs.
    """
    if autotune_config_path is None:
        return dict(DEFAULT_HW_CONFIG)
    assert autotune_config_path.exists(), (
        f"--autotune-config not found: {autotune_config_path}"
    )
    text = autotune_config_path.read_text()
    if autotune_config_path.suffix == ".json":
        data = json.loads(text)
    else:
        assert autotune_config_path.suffix in (".yaml", ".yml"), (
            f"--autotune-config must be .json, .yaml, or .yml: "
            f"{autotune_config_path}"
        )
        yaml_root = yaml.safe_load(text)
        configs = yaml_root.get("configs", yaml_root)
        name = config_name or yaml_root.get("default")
        assert isinstance(name, str) and name, (
            f"{autotune_config_path}: pass --autotune-config-name or set "
            f"a 'default:' key in the YAML"
        )
        data = _resolve_yaml_config(configs, name)
    assert "hw_config" in data, (
        f"{autotune_config_path}: missing 'hw_config' key (resolved "
        f"config: {sorted(data)})"
    )
    return data["hw_config"]


def _existing_combos(
    store_path: Path,
) -> set[tuple[str, str, str, int]]:
    """Read (kernel, preset, hw_config_hash, compute_bw) tuples already
    present in the store. Used to resume-skip pairs whose calibration
    point already exists, so a re-run after a crash doesn't re-compute
    the work that already landed on disk."""
    if not store_path.exists():
        return set()
    combos: set[tuple[str, str, str, int]] = set()
    with store_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            combos.add((
                r["kernel"], r["preset"],
                r["hw_config_hash"], int(r["compute_bw"]),
            ))
    return combos


def _copy_source(step_impl_path: Path, sources_dir: Path) -> tuple[Path, str]:
    """Copy step_impl.py into sources_dir under its sha256-named filename.

    Returns (destination_path, sha256_digest).
    """
    digest = hashlib.sha256(step_impl_path.read_bytes()).hexdigest()
    sources_dir.mkdir(parents=True, exist_ok=True)
    dst = sources_dir / f"{digest}.py"
    if not dst.exists():
        shutil.copyfile(step_impl_path, dst)
    return dst, digest


def calibrate_one(
    *, kernel: str, preset: str, config: dict, hw_config: dict,
    compute_bw: int, sources_dir: Path,
) -> dict:
    """Build → analytical → rust for one (kernel, preset). Returns the
    JSONL record dict (not yet written)."""
    dims = dict(config[kernel]["presets"][preset])
    graph, output_op = build_graph_from_impl(kernel, dims, config)

    # Match autotune2's compute-budget rescaling so analytical & rust
    # numbers are comparable to records the autotuner appends. Mirrors
    # run_autotune2.py --compute-bw / compose._rescale_compute_bw.
    normalize_compute_bw(graph, compute_bw)

    predicted, detail = run_analytical_model(graph, hw_config=hw_config)
    mem = compute_memory_totals(detail, hw_config.get(
        "pmu_buffer_bytes", DEFAULT_HW_CONFIG["pmu_buffer_bytes"]
    ))

    work_dir = STEPDB_DIR / "seed_kernels" / kernel / f"_work_calib_{preset}"
    t0 = time.perf_counter()
    actual = run_simulator(graph, output_op, str(work_dir))
    rust_dur_ms = (time.perf_counter() - t0) * 1000.0

    step_impl = STEPDB_DIR / config[kernel]["step_impl"]
    src_dst, _ = _copy_source(step_impl, sources_dir)

    rust_int = int(actual)
    assert rust_int > 0, (
        f"calibrate_one: rust_cycles must be > 0 for error_pct to be "
        f"meaningful, got {rust_int!r} for {kernel}/{preset}"
    )
    return {
        "node_path": "root",
        "is_root": True,
        "kernel": kernel,
        "preset": preset,
        "composed_source_path": str(src_dst),
        "analytical_cycles": int(predicted),
        "analytical_on_chip": int(mem["on_chip_bytes"]),
        "rust_cycles": rust_int,
        "rust_dur_ms": float(rust_dur_ms),
        "hw_config_hash": _hw_config_hash(hw_config),
        "compute_bw": int(compute_bw),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": _RUN_ID,
        "error_pct": 100.0 * (int(predicted) - rust_int) / rust_int,
    }


_RUN_ID = f"populate_calibration_{int(time.time())}"


def _append_record(store_path: Path, record: dict) -> None:
    assert REQUIRED_FIELDS == record.keys(), (
        f"record schema drift vs CalibrationRecord: extra="
        f"{sorted(record.keys() - REQUIRED_FIELDS)}, missing="
        f"{sorted(REQUIRED_FIELDS - record.keys())}"
    )
    line = json.dumps(record, sort_keys=True) + "\n"
    assert len(line.encode("utf-8")) < 4096, (
        f"populate_calibration: record line is {len(line)} bytes, "
        f"exceeds the 4096-byte atomic-append limit."
    )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with store_path.open("a", encoding="utf-8") as f:
        f.write(line)


def _build_job_list(config: dict, args) -> list[tuple[str, str]]:
    seed_kernels = [k for k, v in config.items() if v.get("origin") == "seed"]
    SMALL_PRESETS = {"small", "tiny", "square"}
    if args.kernel and args.preset:
        return [(args.kernel, args.preset)]
    if args.kernel:
        assert args.kernel in config, f"Unknown kernel: {args.kernel}"
        return [(args.kernel, p) for p in config[args.kernel]["presets"]]
    if args.all:
        return [(k, p) for k in seed_kernels for p in config[k]["presets"]]
    if args.all_small:
        return [
            (k, p) for k in seed_kernels for p in config[k]["presets"]
            if p in SMALL_PRESETS
        ]
    return [
        (k, list(config[k]["presets"].keys())[0]) for k in seed_kernels
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kernel", nargs="?", help="Kernel name")
    p.add_argument("preset", nargs="?", help="Preset name")
    p.add_argument(
        "--all", action="store_true",
        help="All seed kernels, every preset.",
    )
    p.add_argument(
        "--all-small", action="store_true",
        help="All seed kernels, presets in {small,tiny,square} only.",
    )
    p.add_argument(
        "-j", "--jobs", type=int, default=1,
        help="Parallel workers (default: 1 = serial).",
    )
    p.add_argument(
        "--store", type=Path, default=DEFAULT_STORE,
        help=f"Destination JSONL (default: {DEFAULT_STORE}). Sources are "
             f"copied into <store>.parent/calibration_sources/.",
    )
    p.add_argument(
        "--autotune-config", type=Path, default=DEFAULT_AUTOTUNE_CONFIG,
        help=f"JSON or YAML file whose 'hw_config' block is hashed into "
             f"CalibrationRecord.hw_config_hash. Must match what your "
             f"autotune2 runs use, or the records will be filtered out. "
             f"Default: {DEFAULT_AUTOTUNE_CONFIG}.",
    )
    p.add_argument(
        "--autotune-config-name", default=DEFAULT_AUTOTUNE_CONFIG_NAME,
        help=f"Named config to resolve when --autotune-config is a YAML "
             f"file with a 'configs:' block (ignored for JSON). "
             f"Default: {DEFAULT_AUTOTUNE_CONFIG_NAME!r}.",
    )
    p.add_argument(
        "--compute-bw", type=int, default=100000,
        help="Total compute_bw budget rescaled across compute ops "
             "(matches autotune2's --compute-bw). Default: 100000.",
    )
    args = p.parse_args()

    config = load_config()
    hw_config = _load_hw_config(args.autotune_config, args.autotune_config_name)
    sources_dir = args.store.parent / "calibration_sources"
    args.store.parent.mkdir(parents=True, exist_ok=True)

    # Resume-skip: any (kernel, preset, hw_hash, compute_bw) combo
    # already in the store is considered done. A second invocation
    # after a crash picks up where the first left off rather than
    # re-running the work that already landed on disk.
    hw_hash = _hw_config_hash(hw_config)
    existing = _existing_combos(args.store)

    raw_jobs = _build_job_list(config, args)
    jobs = [
        (k, pr) for k, pr in raw_jobs
        if (k, pr, hw_hash, args.compute_bw) not in existing
    ]
    resumed_skip = len(raw_jobs) - len(jobs)

    print(
        f"Calibrating {len(jobs)} (kernel, preset) pair(s) with "
        f"{args.jobs} worker(s).\n"
        f"  store:    {args.store}\n"
        f"  sources:  {sources_dir}\n"
        f"  hw_hash:  {hw_hash}\n"
        f"  compute_bw: {args.compute_bw}\n"
        f"  already in store: {resumed_skip} (will not re-run)\n"
    )

    skipped: list[tuple[str, str, str]] = []
    appended = [0]  # mutable counter usable from worker threads
    print_lock = threading.Lock()

    def _worker(kernel: str, preset: str) -> dict:
        """Compute and persist one calibration point.

        The append happens inside the worker so each record lands on
        disk atomically (single ``write()`` < 4096 bytes — POSIX-atomic)
        as soon as the rust sim returns. Concurrent workers are safe
        against each other for the same reason ``CalibrationStore`` is.
        """
        rec = calibrate_one(
            kernel=kernel, preset=preset, config=config,
            hw_config=hw_config, compute_bw=args.compute_bw,
            sources_dir=sources_dir,
        )
        _append_record(args.store, rec)
        return rec

    def _log_result(idx: int, total: int, k: str, pr: str, rec: dict) -> None:
        err = (
            100.0 * (rec["analytical_cycles"] - rec["rust_cycles"])
            / max(rec["rust_cycles"], 1)
        )
        print(
            f"  [{idx}/{total}] {k:30s} {pr:15s}  "
            f"pred={rec['analytical_cycles']:>10d}  "
            f"actual={rec['rust_cycles']:>10d}  "
            f"err={err:>6.1f}%  "
            f"rust_dur={rec['rust_dur_ms']:.0f}ms"
        )

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            future_to_job = {
                pool.submit(_worker, k, p): (k, p) for k, p in jobs
            }
            done = 0
            for fut in as_completed(future_to_job):
                k, pr = future_to_job[fut]
                done += 1
                try:
                    rec = fut.result()
                except Exception as e:
                    skipped.append((k, pr, str(e)))
                    with print_lock:
                        print(f"  [{done}/{len(jobs)}] {k}/{pr} SKIPPED: {e}")
                    continue
                appended[0] += 1
                with print_lock:
                    _log_result(done, len(jobs), k, pr, rec)
    else:
        for i, (k, pr) in enumerate(jobs, start=1):
            try:
                rec = _worker(k, pr)
            except Exception as e:
                skipped.append((k, pr, str(e)))
                print(f"  [{i}/{len(jobs)}] {k}/{pr} SKIPPED: {e}")
                continue
            appended[0] += 1
            _log_result(i, len(jobs), k, pr, rec)

    print(
        f"\n--- appended {appended[0]} record(s), skipped {len(skipped)}, "
        f"resumed-past {resumed_skip}"
    )
    print(f"--- store:   {args.store}")
    print(f"--- sources: {sources_dir}")
    if skipped:
        print("--- skipped detail:")
        for k, pr, reason in skipped:
            print(f"    {k}/{pr}: {reason[:120]}")


if __name__ == "__main__":
    main()
