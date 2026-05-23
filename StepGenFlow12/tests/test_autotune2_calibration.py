"""Unit tests for autotune2 calibration store composition."""

from __future__ import annotations

from pathlib import Path

from src.autotune2.calibration import (
    CalibrationOverlayStore,
    CalibrationRecord,
    CalibrationStore,
)


def _record(**overrides) -> CalibrationRecord:
    base = dict(
        node_path="root",
        is_root=True,
        kernel="gemm",
        preset="small",
        composed_source_path="/tmp/source.py",
        analytical_cycles=100,
        analytical_on_chip=200,
        rust_cycles=150,
        rust_dur_ms=1.5,
        hw_config_hash="hw0",
        compute_bw=100_000,
        timestamp="2026-05-22T00:00:00+00:00",
        run_id="seed",
        error_pct=(100 - 150) / 150,
    )
    base.update(overrides)
    return CalibrationRecord(**base)


def test_overlay_appends_only_to_append_store(tmp_path: Path):
    seed_path = tmp_path / "seed" / "calibration.jsonl"
    run_path = tmp_path / "run" / "calibration.jsonl"
    seed = CalibrationStore(seed_path)
    run = CalibrationStore(run_path)
    seed.append(_record(run_id="seed"))

    overlay = CalibrationOverlayStore(seed_stores=[seed], append_store=run)
    overlay.append(_record(run_id="run"))

    assert [r.run_id for r in seed.iter_records()] == ["seed"]
    assert [r.run_id for r in run.iter_records()] == ["run"]
    assert [r.run_id for r in overlay.iter_records()] == ["seed", "run"]


def test_overlay_iter_records_filters_seed_and_append_stores(tmp_path: Path):
    seed = CalibrationStore(tmp_path / "seed.jsonl")
    run = CalibrationStore(tmp_path / "run.jsonl")
    seed.append(_record(run_id="seed-keep", hw_config_hash="hw0"))
    seed.append(_record(run_id="seed-drop", hw_config_hash="other"))
    run.append(_record(run_id="run-keep", hw_config_hash="hw0"))
    run.append(_record(run_id="run-drop", hw_config_hash="other"))

    overlay = CalibrationOverlayStore(seed_stores=[seed], append_store=run)

    assert [
        r.run_id for r in overlay.iter_records(hw_config_hash="hw0")
    ] == ["seed-keep", "run-keep"]
