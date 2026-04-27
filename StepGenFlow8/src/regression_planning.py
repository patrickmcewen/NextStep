"""Job planning for the regression runner.

Discovers kernels and presets from StepDB's `bench_config.yaml` and turns
them into a deterministic list of `(kernel, preset)` jobs to execute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

_log = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class Job:
    kernel: str
    preset: str


def load_bench_config(path: Path) -> dict:
    assert path.exists(), f"bench_config.yaml not found at {path}"
    with open(path) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"bench_config.yaml must be a YAML mapping: {path}"
    return data


def plan_jobs(
    bench_config: dict,
    preset_config: dict | None,
    all_presets: bool,
) -> list[Job]:
    """Return the sorted list of jobs to execute.

    Exactly one of `preset_config` (dict mapping kernel -> preset or list of
    presets) or `all_presets` (bool) must be supplied.
    """
    assert (preset_config is not None) ^ all_presets, \
        "exactly one of preset_config / all_presets is required"

    if all_presets:
        jobs = [
            Job(kernel, preset)
            for kernel, entry in bench_config.items()
            for preset in entry.get("presets", {})
        ]
    else:
        jobs = []  # filled in by Task 3

    jobs.sort()
    assert jobs, "no jobs to run — check bench_config / preset_config"
    return jobs
