from pathlib import Path

import pytest
import yaml

from src.regression_planning import Job, load_bench_config, plan_jobs


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


def test_load_bench_config_rejects_non_mapping(tmp_path: Path):
    p = tmp_path / "bench_config.yaml"
    p.write_text("- 1\n- 2\n")  # YAML list, not mapping
    with pytest.raises(AssertionError):
        load_bench_config(p)


def _bench_config_two_kernels() -> dict:
    return {
        "gemm": {"presets": {"square": {}, "small": {}}},
        "silu": {"presets": {"small": {}}},
    }


def test_plan_jobs_all_presets_emits_every_pair_sorted():
    jobs = plan_jobs(_bench_config_two_kernels(), preset_config=None, all_presets=True)
    assert jobs == [
        Job("gemm", "small"),
        Job("gemm", "square"),
        Job("silu", "small"),
    ]


def test_plan_jobs_requires_exactly_one_mode():
    cfg = _bench_config_two_kernels()
    with pytest.raises(AssertionError):
        plan_jobs(cfg, preset_config=None, all_presets=False)
    with pytest.raises(AssertionError):
        plan_jobs(cfg, preset_config={"gemm": "small"}, all_presets=True)


def test_plan_jobs_all_presets_zero_jobs_asserts():
    with pytest.raises(AssertionError):
        plan_jobs({}, preset_config=None, all_presets=True)
