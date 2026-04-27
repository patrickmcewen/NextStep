from pathlib import Path

import pytest
import yaml

from src.regression_planning import Job, load_bench_config


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def test_job_dataclass_is_hashable_and_ordered():
    a = Job(kernel="gemm", preset="small")
    b = Job(kernel="gemm", preset="small")
    c = Job(kernel="gemm", preset="square")
    assert a == b
    assert a < c  # ordering is (kernel, preset)


def test_load_bench_config_reads_yaml(tmp_path: Path):
    p = tmp_path / "bench_config.yaml"
    _write_yaml(p, {"gemm": {"presets": {"small": {"M": 1}}}})
    cfg = load_bench_config(p)
    assert cfg["gemm"]["presets"]["small"] == {"M": 1}


def test_load_bench_config_missing_path_asserts(tmp_path: Path):
    with pytest.raises(AssertionError):
        load_bench_config(tmp_path / "does_not_exist.yaml")
