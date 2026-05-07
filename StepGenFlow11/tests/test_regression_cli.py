# tests/test_regression_cli.py
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _make_fake_run_py(
    path: Path,
    exit_map: dict[tuple[str, str], int],
    per_outer_map: dict[tuple[str, str], list[bool]] | None = None,
) -> None:
    """Fake run.py that exits with a code from exit_map and writes a result.json
    (with `per_outer`) to the dir passed via --checkpoint-dir."""
    per_outer_map = per_outer_map or {}
    src = textwrap.dedent(f"""
        import argparse, json, sys
        from pathlib import Path

        exit_map = {exit_map!r}
        per_outer_map = {per_outer_map!r}

        p = argparse.ArgumentParser()
        p.add_argument("kernel")
        p.add_argument("preset")
        p.add_argument("--model", default=None)
        p.add_argument("--config", default=None)
        p.add_argument("--max-outer", type=int, default=None)
        p.add_argument("--max-turns", type=int, default=None)
        p.add_argument("--pipeline", default=None)
        p.add_argument("--translator", default=None)
        p.add_argument("--checkpoint-dir", default=None)
        p.add_argument("--autotune", action="store_true")
        p.add_argument("--autotune-config", default=None)
        p.add_argument("--autotune-max-turns", type=int, default=None)
        p.add_argument("--autotune-agent", default=None)
        args = p.parse_args()

        print(f"fake run for {{args.kernel}}/{{args.preset}}")
        if args.checkpoint_dir:
            ckpt = Path(args.checkpoint_dir)
            ckpt.mkdir(parents=True, exist_ok=True)
            outers = per_outer_map.get((args.kernel, args.preset))
            if outers is None:
                outers = [exit_map.get((args.kernel, args.preset), 0) == 0]
            (ckpt / "result.json").write_text(json.dumps({{
                "success": any(outers),
                "per_outer": [{{"outer": i, "success": s}} for i, s in enumerate(outers)],
            }}))
        sys.exit(exit_map.get((args.kernel, args.preset), 0))
    """)
    path.write_text(src)


def test_cli_all_presets_writes_summary(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    bench = work / "bench_config.yaml"
    bench.write_text(yaml.safe_dump({
        "gemm": {"presets": {"small": {}, "square": {}}},
        "silu": {"presets": {"small": {}}},
    }))
    fake_run_py = work / "run.py"
    _make_fake_run_py(
        fake_run_py,
        exit_map={("gemm", "square"): 1},
        per_outer_map={("gemm", "square"): [False, True, False]},
    )

    results_root = work / "regression_results"
    cmd = [
        sys.executable, str(REPO / "run_regression.py"),
        "--all-presets",
        "--max-parallel", "2",
        "--results-root", str(results_root),
        "--bench-config", str(bench),
        "--run-py", str(fake_run_py),
        "--model", "fake-model",
    ]
    completed = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    runs = list(results_root.iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["overall"] == {"passed": 2, "total": 3, "fraction": 2 / 3}
    assert summary["benchmarks"]["gemm"]["presets"]["square"]["status"] == "fail"
    assert summary["benchmarks"]["gemm"]["presets"]["square"]["outer_passed"] == 1
    assert summary["benchmarks"]["gemm"]["presets"]["square"]["outer_total"] == 3
    assert "outer_overall" in summary
    assert summary["outer_overall"]["total"] > 0
    assert (run_dir / "jobs" / "gemm__square.log").exists()
    assert "regression.log" in {p.name for p in run_dir.iterdir()}


def test_cli_subset_mode_writes_summary(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    bench = work / "bench_config.yaml"
    bench.write_text(yaml.safe_dump({
        "gemm":   {"presets": {"small": {}, "square": {}}},
        "silu":   {"presets": {"small": {}}},
        "rms":    {"presets": {"small": {}}},
    }))
    fake_run_py = work / "run.py"
    _make_fake_run_py(fake_run_py, exit_map={("gemm", "small"): 1}, per_outer_map={})

    subsets = work / "subsets.yaml"
    subsets.write_text(yaml.safe_dump({
        "simple_set": {"silu": "small", "rms": "small"},
        "gemm_set":   {"gemm": ["small", "square"]},
    }))

    results_root = work / "regression_results"
    cmd = [
        sys.executable, str(REPO / "run_regression.py"),
        "--subset", "gemm_set",
        "--subset-file", str(subsets),
        "--max-parallel", "2",
        "--results-root", str(results_root),
        "--bench-config", str(bench),
        "--run-py", str(fake_run_py),
        "--model", "fake-model",
    ]
    completed = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    runs = list(results_root.iterdir())
    assert len(runs) == 1
    summary = json.loads((runs[0] / "summary.json").read_text())
    # Subset only ran the two gemm presets
    assert set(summary["benchmarks"].keys()) == {"gemm"}
    assert summary["overall"] == {"passed": 1, "total": 2, "fraction": 0.5}
    assert summary["benchmarks"]["gemm"]["presets"]["small"]["status"] == "fail"
    assert summary["benchmarks"]["gemm"]["presets"]["square"]["status"] == "pass"
    assert summary["preset_mode"] == "subset:gemm_set"
    assert "outer_passed" in summary["benchmarks"]["gemm"]["presets"]["small"]
    assert "outer_total" in summary["benchmarks"]["gemm"]["presets"]["small"]
    assert "outer_overall" in summary
