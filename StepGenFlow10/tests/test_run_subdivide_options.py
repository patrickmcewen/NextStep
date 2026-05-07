"""Smoke test: --no-subdivide and the four numeric flags parse cleanly,
and bundle/direct combos with subdivide enabled assert."""
import asyncio
import subprocess
import sys

import pytest


def test_run_help_includes_subdivide_flags():
    out = subprocess.run(
        [sys.executable, "run.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    for f in ("--max-subdivide-turns", "--max-subdivides-per-outer",
              "--max-subdivide-depth", "--no-subdivide"):
        assert f in out, f"flag {f} missing from --help"


def test_regression_help_includes_subdivide_flags():
    out = subprocess.run(
        [sys.executable, "run_regression.py", "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    for f in ("--max-subdivide-turns", "--max-subdivides-per-outer",
              "--max-subdivide-depth", "--no-subdivide"):
        assert f in out, f"flag {f} missing from --help"


def test_run_kernel_bundle_with_subdivide_asserts(tmp_path):
    from src.orchestrator import run_kernel
    fake_bundle = tmp_path / "bundle"
    fake_bundle.mkdir()
    with pytest.raises(AssertionError, match="bundle"):
        asyncio.run(run_kernel(
            kernel_name="any", preset="any", llm_config={},
            bundle_dir=str(fake_bundle),
            subdivide_enabled=True,
        ))


def test_run_kernel_direct_with_subdivide_asserts():
    from src.orchestrator import run_kernel
    with pytest.raises(AssertionError, match="standard"):
        asyncio.run(run_kernel(
            kernel_name="any", preset="any", llm_config={},
            pipeline="direct",
            subdivide_enabled=True,
        ))
