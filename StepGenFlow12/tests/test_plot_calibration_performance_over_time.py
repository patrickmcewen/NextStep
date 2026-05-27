from __future__ import annotations

import json
import hashlib
from pathlib import Path

from plot_calibration_performance_over_time import (
    build_figure,
    default_baseline_score_path,
    default_output_path,
    load_baseline_score,
    load_points,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_points_filters_node_and_sorts_by_timestamp(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    _write_jsonl(
        calibration_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "timestamp": "2026-05-27T10:40:00+00:00",
                "analytical_cycles": 200,
                "rust_cycles": 260,
            },
            {
                "node_path": "root",
                "kernel": "end_to_end",
                "timestamp": "2026-05-27T10:20:00+00:00",
                "analytical_cycles": 999,
                "rust_cycles": 999,
            },
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "timestamp": "2026-05-27T10:10:00+00:00",
                "analytical_cycles": 100,
                "rust_cycles": 130,
            },
        ],
    )

    kernel, points = load_points(calibration_path, "root/moe_block")

    assert kernel == "end_to_end"
    assert [p.analytical_cycles for p in points] == [100, 200]
    assert [p.rust_cycles for p in points] == [130, 260]


def test_default_output_path_uses_node_name(tmp_path: Path):
    calibration_path = tmp_path / "autotune2" / "calibration.jsonl"

    assert default_output_path(calibration_path, "root/moe_block") == (
        tmp_path / "autotune2" / "calibration_performance_over_time_root_moe_block.png"
    )


def test_default_baseline_score_path_uses_node_path(tmp_path: Path):
    calibration_path = tmp_path / "autotune2" / "calibration.jsonl"

    assert default_baseline_score_path(calibration_path, "root/moe_block") == (
        tmp_path / "autotune2" / "root" / "moe_block" / "pass_0_general" / "pass1_baseline_score.json"
    )


def test_load_baseline_score_reads_cycles_and_on_chip(tmp_path: Path):
    baseline_path = tmp_path / "pass1_baseline_score.json"
    baseline_path.write_text(
        json.dumps({"cycles": 123, "on_chip": 456, "provenance": "pass1_baseline"}),
        encoding="utf-8",
    )

    assert load_baseline_score(baseline_path) == (123, 456)


def test_build_figure_plots_rust_and_analytical_lines(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    _write_jsonl(
        calibration_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "timestamp": "2026-05-27T10:10:00+00:00",
                "analytical_cycles": 100,
                "rust_cycles": 130,
            },
        ],
    )
    kernel, points = load_points(calibration_path, "root/moe_block")

    fig = build_figure(points, "root/moe_block", kernel, baseline_score=(90, 1_000))
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]

    assert "Analytical cycles" in labels
    assert "Rust cycles" in labels
    assert "Baseline score" in labels
    assert ":" in [line.get_linestyle() for line in ax.get_lines()]
    assert ax.get_xlabel() == "Time"
    assert ax.get_ylabel() == "Performance (cycles)"
    assert ax.get_yscale() == "log"
    assert "end_to_end" in ax.get_title()
    assert "root/moe_block" in ax.get_title()


def test_build_figure_can_plot_rust_only(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    _write_jsonl(
        calibration_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "timestamp": "2026-05-27T10:10:00+00:00",
                "analytical_cycles": 100,
                "rust_cycles": 130,
            },
        ],
    )
    kernel, points = load_points(calibration_path, "root/moe_block")

    fig = build_figure(points, "root/moe_block", kernel, rust_only=True)
    labels = [line.get_label() for line in fig.axes[0].get_lines()]

    assert labels == ["Rust cycles"]


def test_load_points_adds_under_budget_analytical_decision_score(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    decision_path = tmp_path / "agent_decisions.jsonl"
    turn_dir = (
        tmp_path
        / "root"
        / "moe_block"
        / "pass_0_general"
        / "baseline_0_attempt_0_b200"
        / "session_0"
        / "turn_0"
    )
    turn_dir.mkdir(parents=True)
    composed_source = "def composed():\n    return 1\n"
    source_hash = hashlib.sha256(composed_source.encode("utf-8")).hexdigest()
    (turn_dir / "composed_source.py").write_text(composed_source, encoding="utf-8")
    (turn_dir / "score.json").write_text(
        json.dumps({"entries": [{"cycles": 123, "on_chip": 150, "provenance": "p"}]}),
        encoding="utf-8",
    )
    _write_jsonl(
        calibration_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "timestamp": "2026-05-27T10:10:00+00:00",
                "analytical_cycles": 100,
                "rust_cycles": 130,
            },
        ],
    )
    _write_jsonl(
        decision_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "decision": "analytical",
                "hw_config_hash": "hw0",
                "composed_source_hash": source_hash,
                "timestamp": "2026-05-27T10:20:00+00:00",
            },
        ],
    )

    _kernel, points = load_points(
        calibration_path, "root/moe_block", agent_decisions_path=decision_path
    )

    assert [point.analytical_cycles for point in points] == [100, 123]
    assert [point.rust_cycles for point in points] == [130, None]


def test_load_points_skips_over_budget_analytical_decision_score(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    decision_path = tmp_path / "agent_decisions.jsonl"
    turn_dir = (
        tmp_path
        / "root"
        / "moe_block"
        / "pass_0_general"
        / "baseline_0_attempt_0_b200"
        / "session_0"
        / "turn_0"
    )
    turn_dir.mkdir(parents=True)
    composed_source = "def composed():\n    return 1\n"
    source_hash = hashlib.sha256(composed_source.encode("utf-8")).hexdigest()
    (turn_dir / "composed_source.py").write_text(composed_source, encoding="utf-8")
    (turn_dir / "score.json").write_text(
        json.dumps({"entries": [{"cycles": 123, "on_chip": 250, "provenance": "p"}]}),
        encoding="utf-8",
    )
    _write_jsonl(
        calibration_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "timestamp": "2026-05-27T10:10:00+00:00",
                "analytical_cycles": 100,
                "rust_cycles": 130,
            },
        ],
    )
    _write_jsonl(
        decision_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "decision": "analytical",
                "hw_config_hash": "hw0",
                "composed_source_hash": source_hash,
                "timestamp": "2026-05-27T10:20:00+00:00",
            },
        ],
    )
    kernel, points = load_points(
        calibration_path, "root/moe_block", agent_decisions_path=decision_path
    )

    assert [point.analytical_cycles for point in points] == [100]
    assert [point.rust_cycles for point in points] == [130]
