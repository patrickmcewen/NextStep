"""Standalone verifier for the bufferize+streamify few-shot demo set.

Runs each demo's `tiled_reference` against its PyTorch reference and asserts
torch.allclose. No StepGenFlow11 pipeline, no LLM, no STeP graph compiler --
just the eager DSL functional model from step_dsl.py.

Usage (from repo root):
    python StepDB/seed_kernels/language_primitives/_verify_permute_demos.py
"""
import importlib.util
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_STEPGENFLOW_SRC = _THIS_DIR.parents[2] / "StepGenFlow11" / "src"
sys.path.insert(0, str(_STEPGENFLOW_SRC))

import step_dsl  # noqa: E402


_DSL_GLOBALS = {
    name: getattr(step_dsl, name)
    for name in step_dsl.DSL_FUNCTIONS
    if hasattr(step_dsl, name)
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_tiled_reference(dsl_path: Path):
    code = dsl_path.read_text()
    g = dict(_DSL_GLOBALS)
    g["torch"] = torch
    exec(compile(code, str(dsl_path), "exec"), g)
    assert "tiled_reference" in g, f"{dsl_path} did not define tiled_reference"
    return g["tiled_reference"]


def _run_demo(demo_dir: Path, dims: dict, input_keys_from_inputs):
    ref = _load_module(demo_dir / "reference.py", f"{demo_dir.name}_ref")
    tiled = _load_tiled_reference(demo_dir / "dsl_code.py")

    raw_inputs = ref.get_inputs(dims)
    tensors = input_keys_from_inputs(raw_inputs, dims)

    out = tiled(dims, tensors)
    gold = ref.compute_gold(dims)

    assert torch.is_tensor(out), f"{demo_dir.name}: tiled_reference returned {type(out)}"
    assert out.shape == gold.shape, (
        f"{demo_dir.name}: shape mismatch out={tuple(out.shape)} gold={tuple(gold.shape)}"
    )
    close = torch.allclose(out, gold, rtol=1e-5, atol=1e-5)
    max_abs = (out - gold).abs().max().item()
    return close, tuple(out.shape), tuple(gold.shape), max_abs


def main():
    demos = [
        {
            "dir": _THIS_DIR / "tile_grid_transpose_2d",
            "dims": {"M": 64, "K": 64, "tile_m": 16, "tile_k": 8},
            "to_tensors": lambda inputs, dims: {"input": inputs[0]},
        },
        {
            "dir": _THIS_DIR / "head_split_permute",
            "dims": {"S": 8, "H": 4, "D": 16},
            "to_tensors": lambda inputs, dims: {"input": inputs[0]},
        },
        {
            "dir": _THIS_DIR / "attn_layout_permute_4d",
            "dims": {"B": 2, "H": 4, "S": 8, "D": 16},
            "to_tensors": lambda inputs, dims: {"input": inputs[0]},
        },
    ]

    all_pass = True
    for demo in demos:
        ok, out_shape, gold_shape, max_abs = _run_demo(
            demo["dir"], demo["dims"], demo["to_tensors"]
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] {demo['dir'].name:30s}  "
            f"out={out_shape}  gold={gold_shape}  max|diff|={max_abs:.2e}"
        )
        if not ok:
            all_pass = False

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
