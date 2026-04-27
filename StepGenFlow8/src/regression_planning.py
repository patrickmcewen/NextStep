"""Job planning for the regression runner.

Discovers kernels and presets from StepDB's `bench_config.yaml` and turns
them into a deterministic list of `(kernel, preset)` jobs to execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, order=True)
class Job:
    kernel: str
    preset: str


def load_bench_config(path: Path) -> dict:
    assert path.exists(), f"bench_config.yaml not found at {path}"
    with open(path) as f:
        return yaml.safe_load(f)
