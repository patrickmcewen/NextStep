from __future__ import annotations

import argparse
import json
from pathlib import Path


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_points(calibration_path: Path, node_path: str) -> list[tuple[int, int]]:
    return load_plot_data(calibration_path, node_path)[1]


def load_plot_data(calibration_path: Path, node_path: str) -> tuple[str, list[tuple[int, int]]]:
    kernel: str | None = None
    points: list[tuple[int, int]] = []
    with calibration_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            assert "node_path" in record, f"line {line_no}: missing node_path"
            assert "kernel" in record, f"line {line_no}: missing kernel"
            assert "analytical_cycles" in record, f"line {line_no}: missing analytical_cycles"
            assert "rust_cycles" in record, f"line {line_no}: missing rust_cycles"
            if record["node_path"] == node_path:
                if kernel is None:
                    kernel = str(record["kernel"])
                assert kernel == record["kernel"], (
                    f"line {line_no}: mixed kernels for node_path={node_path!r}: "
                    f"{kernel!r} and {record['kernel']!r}"
                )
                points.append((int(record["analytical_cycles"]), int(record["rust_cycles"])))
    assert points, f"no calibration records found for node_path={node_path!r}"
    assert kernel is not None
    return kernel, points


def load_memory_points(calibration_path: Path, node_path: str) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    with calibration_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            assert "node_path" in record, f"line {line_no}: missing node_path"
            assert "analytical_cycles" in record, f"line {line_no}: missing analytical_cycles"
            assert "analytical_on_chip" in record, f"line {line_no}: missing analytical_on_chip"
            assert "rust_cycles" in record, f"line {line_no}: missing rust_cycles"
            if record["node_path"] == node_path:
                points.append(
                    (
                        int(record["analytical_on_chip"]),
                        int(record["analytical_cycles"]),
                        int(record["rust_cycles"]),
                    )
                )
    assert points, f"no calibration records found for node_path={node_path!r}"
    return points


def default_output_path(calibration_path: Path, node_path: str) -> Path:
    node_slug = node_path.replace("/", "_")
    return calibration_path.parent / f"calibration_cycles_{node_slug}.png"


def default_memory_output_path(calibration_path: Path, node_path: str) -> Path:
    node_slug = node_path.replace("/", "_")
    return calibration_path.parent / f"calibration_cycles_vs_on_chip_{node_slug}.png"


def build_figure(points: list[tuple[int, int]], node_path: str, kernel: str):
    plt = _setup_matplotlib()

    analytical_cycles = [p[0] for p in points]
    rust_cycles = [p[1] for p in points]
    assert min([*analytical_cycles, *rust_cycles]) > 0, "log plot requires positive cycles"
    max_cycle = max([*analytical_cycles, *rust_cycles])
    min_cycle = min([*analytical_cycles, *rust_cycles])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(analytical_cycles, rust_cycles, alpha=0.75)
    ax.plot(
        [min_cycle, max_cycle],
        [min_cycle, max_cycle],
        linestyle="--",
        color="black",
        linewidth=1,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Calibration cycles: {kernel} / {node_path}", fontsize=18)
    ax.set_xlabel("Analytical cycles", fontsize=15)
    ax.set_ylabel("Rust cycles", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def build_memory_figure(points: list[tuple[int, int, int]], node_path: str, kernel: str):
    plt = _setup_matplotlib()

    sorted_points = sorted(points)
    on_chip = [p[0] for p in sorted_points]
    analytical_cycles = [p[1] for p in sorted_points]
    rust_cycles = [p[2] for p in sorted_points]
    assert min([*on_chip, *analytical_cycles, *rust_cycles]) > 0, (
        "log plot requires positive memory and cycles"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(on_chip, analytical_cycles, marker="o", label="Analytical cycles")
    ax.plot(on_chip, rust_cycles, marker="o", label="Rust cycles")
    for memory, analytical, rust in sorted_points:
        ax.plot([memory, memory], [analytical, rust], linestyle=":", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Cycles vs on-chip memory: {kernel} / {node_path}", fontsize=18)
    ax.set_xlabel("Analytical on-chip memory", fontsize=15)
    ax.set_ylabel("Cycles", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=13)
    fig.tight_layout()
    return fig


def plot_points(
    points: list[tuple[int, int]], node_path: str, kernel: str, output_path: Path
) -> None:
    fig = build_figure(points, node_path, kernel)
    fig.savefig(output_path, dpi=160)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_memory_points(
    points: list[tuple[int, int, int]], node_path: str, kernel: str, output_path: Path
) -> None:
    fig = build_memory_figure(points, node_path, kernel)
    fig.savefig(output_path, dpi=160)
    plt = _setup_matplotlib()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot analytical_cycles versus rust_cycles for one calibration node."
    )
    parser.add_argument("calibration_jsonl", type=Path)
    parser.add_argument("node_path")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path. Defaults next to calibration.jsonl using the node path in the filename.",
    )
    parser.add_argument(
        "--memory-output",
        type=Path,
        help="Cycles-vs-memory PNG path. Defaults next to calibration.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.calibration_jsonl, args.node_path)
    memory_output_path = args.memory_output or default_memory_output_path(
        args.calibration_jsonl, args.node_path
    )
    kernel, points = load_plot_data(args.calibration_jsonl, args.node_path)
    memory_points = load_memory_points(args.calibration_jsonl, args.node_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memory_output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_points(points, args.node_path, kernel, output_path)
    plot_memory_points(memory_points, args.node_path, kernel, memory_output_path)
    print(f"wrote {output_path}")
    print(f"wrote {memory_output_path}")


if __name__ == "__main__":
    main()
