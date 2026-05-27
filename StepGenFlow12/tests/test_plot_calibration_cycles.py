from __future__ import annotations

import json
from pathlib import Path

from plot_calibration_cycles import (
    build_memory_figure,
    build_figure,
    default_memory_output_path,
    default_output_path,
    default_baseline_score_path,
    load_baseline_score,
    load_memory_points,
    load_plot_data,
    load_points,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_points_filters_exact_node_path(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    _write_jsonl(
        calibration_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "analytical_cycles": 100,
                "analytical_on_chip": 1_000,
                "rust_cycles": 125,
            },
            {
                "node_path": "root",
                "kernel": "end_to_end",
                "analytical_cycles": 200,
                "analytical_on_chip": 2_000,
                "rust_cycles": 250,
            },
        ],
    )

    assert load_points(calibration_path, "root/moe_block") == [(100, 125)]


def test_default_output_path_uses_node_name(tmp_path: Path):
    calibration_path = tmp_path / "autotune2" / "calibration.jsonl"

    assert default_output_path(calibration_path, "root/moe_block") == (
        tmp_path / "autotune2" / "calibration_cycles_root_moe_block.png"
    )


def test_default_memory_output_path_uses_node_name(tmp_path: Path):
    calibration_path = tmp_path / "autotune2" / "calibration.jsonl"

    assert default_memory_output_path(calibration_path, "root/moe_block") == (
        tmp_path / "autotune2" / "calibration_cycles_vs_on_chip_root_moe_block.png"
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


def test_build_figure_uses_log_axes_and_larger_fonts():
    fig = build_figure([(100, 125), (10_000, 12_500)], "root/moe_block", "end_to_end")
    ax = fig.axes[0]

    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    assert ax.title.get_fontsize() >= 18
    assert ax.xaxis.label.get_fontsize() >= 15
    assert ax.yaxis.label.get_fontsize() >= 15


def test_load_plot_data_includes_kernel_in_title(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    _write_jsonl(
        calibration_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "analytical_cycles": 100,
                "analytical_on_chip": 1_000,
                "rust_cycles": 125,
            },
        ],
    )

    kernel, points = load_plot_data(calibration_path, "root/moe_block")
    fig = build_figure(points, "root/moe_block", kernel)

    assert "end_to_end" in fig.axes[0].get_title()
    assert "root/moe_block" in fig.axes[0].get_title()


def test_load_memory_points_filters_exact_node_path(tmp_path: Path):
    calibration_path = tmp_path / "calibration.jsonl"
    _write_jsonl(
        calibration_path,
        [
            {
                "node_path": "root/moe_block",
                "kernel": "end_to_end",
                "analytical_cycles": 100,
                "analytical_on_chip": 1_000,
                "rust_cycles": 125,
            },
            {
                "node_path": "root",
                "kernel": "end_to_end",
                "analytical_cycles": 200,
                "analytical_on_chip": 2_000,
                "rust_cycles": 250,
            },
        ],
    )

    assert load_memory_points(calibration_path, "root/moe_block") == [(1_000, 100, 125)]


def test_build_memory_figure_labels_series_and_draws_mismatch_connectors():
    fig = build_memory_figure(
        [(1_000, 100, 125), (10_000, 1_000, 1_250)],
        "root/moe_block",
        "end_to_end",
        baseline_score=(500, 5_000),
    )
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    dotted_lines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    collection_labels = [collection.get_label() for collection in ax.collections]

    assert "Analytical cycles" in labels
    assert "Rust cycles" in labels
    assert len(dotted_lines) == 2
    assert "Baseline score" in collection_labels
    assert ax.get_xlabel() == "Analytical on-chip memory"
    assert ax.get_ylabel() == "Cycles"
    assert list(dotted_lines[0].get_xdata()) == [1_000, 1_000]
    assert list(dotted_lines[0].get_ydata()) == [100, 125]
