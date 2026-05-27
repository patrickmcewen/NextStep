from __future__ import annotations

import argparse
import re
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PerformancePoint:
    timestamp: datetime
    analytical_cycles: int | None
    rust_cycles: int | None


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _memory_budget_from_turn_dir(turn_dir: Path) -> int:
    for parent in [turn_dir, *turn_dir.parents]:
        match = re.search(r"_b(\d+)$", parent.name)
        if match:
            return int(match.group(1))
    raise AssertionError(f"could not find memory budget in path: {turn_dir}")


def _composed_source_paths_by_hash(root_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    import hashlib

    for source_path in root_dir.rglob("composed_source.py"):
        source_hash = hashlib.sha256(source_path.read_text(encoding="utf-8").encode()).hexdigest()
        paths[source_hash] = source_path
    return paths


def load_agent_analytical_points(
    agent_decisions_path: Path, node_path: str
) -> list[PerformancePoint]:
    points: list[PerformancePoint] = []
    source_paths = _composed_source_paths_by_hash(agent_decisions_path.parent)
    with agent_decisions_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            assert "node_path" in record, f"line {line_no}: missing node_path"
            assert "decision" in record, f"line {line_no}: missing decision"
            assert "composed_source_hash" in record, f"line {line_no}: missing composed_source_hash"
            assert "timestamp" in record, f"line {line_no}: missing timestamp"
            if record["node_path"] == node_path and record["decision"] == "analytical":
                source_path = source_paths.get(str(record["composed_source_hash"]))
                if source_path is None:
                    continue
                turn_dir = source_path.parent
                score_path = turn_dir / "score.json"
                if not score_path.exists():
                    continue
                score = json.loads(score_path.read_text(encoding="utf-8"))
                assert "entries" in score, f"{score_path}: missing entries"
                assert len(score["entries"]) == 1, f"{score_path}: expected exactly one score entry"
                entry = score["entries"][0]
                assert "cycles" in entry, f"{score_path}: missing cycles"
                assert "on_chip" in entry, f"{score_path}: missing on_chip"
                if int(entry["on_chip"]) <= _memory_budget_from_turn_dir(turn_dir):
                    points.append(
                        PerformancePoint(
                            timestamp=datetime.fromisoformat(record["timestamp"]),
                            analytical_cycles=int(entry["cycles"]),
                            rust_cycles=None,
                        )
                    )
    return points


def load_points(
    calibration_path: Path,
    node_path: str,
    *,
    agent_decisions_path: Path | None = None,
) -> tuple[str, list[PerformancePoint]]:
    kernel: str | None = None
    points: list[PerformancePoint] = []
    with calibration_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            assert "node_path" in record, f"line {line_no}: missing node_path"
            assert "kernel" in record, f"line {line_no}: missing kernel"
            assert "timestamp" in record, f"line {line_no}: missing timestamp"
            assert "analytical_cycles" in record, f"line {line_no}: missing analytical_cycles"
            assert "rust_cycles" in record, f"line {line_no}: missing rust_cycles"
            if record["node_path"] == node_path:
                if kernel is None:
                    kernel = str(record["kernel"])
                assert kernel == record["kernel"], (
                    f"line {line_no}: mixed kernels for node_path={node_path!r}: "
                    f"{kernel!r} and {record['kernel']!r}"
                )
                points.append(
                    PerformancePoint(
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        analytical_cycles=int(record["analytical_cycles"]),
                        rust_cycles=int(record["rust_cycles"]),
                    )
                )
    assert points, f"no calibration records found for node_path={node_path!r}"
    assert kernel is not None
    if agent_decisions_path is not None:
        points.extend(load_agent_analytical_points(agent_decisions_path, node_path))
    return kernel, sorted(points, key=lambda point: point.timestamp)


def default_output_path(calibration_path: Path, node_path: str) -> Path:
    node_slug = node_path.replace("/", "_")
    return calibration_path.parent / f"calibration_performance_over_time_{node_slug}.png"


def default_baseline_score_path(calibration_path: Path, node_path: str) -> Path:
    return calibration_path.parent.joinpath(
        *node_path.split("/"), "pass_0_general", "pass1_baseline_score.json"
    )


def load_baseline_score(baseline_path: Path) -> tuple[int, int]:
    score = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert "cycles" in score, f"{baseline_path}: missing cycles"
    assert "on_chip" in score, f"{baseline_path}: missing on_chip"
    return int(score["cycles"]), int(score["on_chip"])


def build_figure(
    points: list[PerformancePoint],
    node_path: str,
    kernel: str,
    *,
    rust_only: bool = False,
    baseline_score: tuple[int, int] | None = None,
):
    plt = _setup_matplotlib()

    analytical_points = [point for point in points if point.analytical_cycles is not None]
    rust_points = [point for point in points if point.rust_cycles is not None]
    analytical_cycles = [point.analytical_cycles for point in analytical_points]
    rust_cycles = [point.rust_cycles for point in rust_points]
    assert min([*analytical_cycles, *rust_cycles]) > 0, "log plot requires positive cycles"

    fig, ax = plt.subplots(figsize=(9, 6))
    if not rust_only:
        ax.plot(
            [point.timestamp for point in analytical_points],
            analytical_cycles,
            marker="o",
            label="Analytical cycles",
        )
    ax.plot([point.timestamp for point in rust_points], rust_cycles, marker="o", label="Rust cycles")
    if baseline_score is not None:
        baseline_cycles, _baseline_on_chip = baseline_score
        ax.axhline(
            baseline_cycles,
            linestyle=":",
            color="black",
            linewidth=1.5,
            label="Baseline score",
        )
    ax.set_yscale("log")
    ax.set_title(f"Performance over time: {kernel} / {node_path}", fontsize=18)
    ax.set_xlabel("Time", fontsize=15)
    ax.set_ylabel("Performance (cycles)", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=13)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_points(
    points: list[PerformancePoint],
    node_path: str,
    kernel: str,
    output_path: Path,
    *,
    rust_only: bool = False,
    baseline_score: tuple[int, int] | None = None,
) -> None:
    fig = build_figure(
        points,
        node_path,
        kernel,
        rust_only=rust_only,
        baseline_score=baseline_score,
    )
    fig.savefig(output_path, dpi=160)
    plt = _setup_matplotlib()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot analytical and rust cycle performance over calibration time."
    )
    parser.add_argument("calibration_jsonl", type=Path)
    parser.add_argument("node_path")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path. Defaults next to calibration.jsonl using the node path in the filename.",
    )
    parser.add_argument(
        "--rust-only",
        action="store_true",
        help="Plot only rust_cycles, omitting the analytical_cycles line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.calibration_jsonl, args.node_path)
    agent_decisions_path = args.calibration_jsonl.parent / "agent_decisions.jsonl"
    baseline_path = default_baseline_score_path(args.calibration_jsonl, args.node_path)
    kernel, points = load_points(
        args.calibration_jsonl,
        args.node_path,
        agent_decisions_path=agent_decisions_path if agent_decisions_path.exists() else None,
    )
    baseline_score = load_baseline_score(baseline_path) if baseline_path.exists() else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_points(
        points,
        args.node_path,
        kernel,
        output_path,
        rust_only=args.rust_only,
        baseline_score=baseline_score,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
