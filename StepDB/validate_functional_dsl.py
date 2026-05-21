"""Validate a step_dsl ``tiled_reference`` against the PyTorch reference and
its dsl_to_step translation.

Why a separate validator? ``validate_functional.py`` exec's a step_impl.py
that builds the STeP IR graph directly. The autotune pipeline instead
hands the LLM the higher-level step_dsl surface (``tiled_reference``),
then runs ``dsl_to_step.translate`` to produce IR. The DSL primitives are
more lenient than the IR ops they lower to — most prominently
``offchip_store`` strips a leading singleton in DSL form but the IR
``OffChipStore.tensor_shape_tiled = in_stream.shape[1:]`` treats the
leading dim as iteration count. That gap can pass DSL semantics yet
silently corrupt IR execution.

This script runs three computations on the same inputs and compares:

  1. PyTorch gold (kernel's reference.py / ``compute_gold``)
  2. DSL semantics   (exec ``tiled_reference`` with ``step_dsl`` imported)
  3. IR semantics    (``dsl_to_step.translate`` + functional simulator)

Usage:
    python validate_functional_dsl.py <dsl_file.py> <kernel> [preset]
    python validate_functional_dsl.py <dsl_file.py> <kernel> --dims M=64,K=64,N=64,tile_m=16,tile_k=16,tile_n=16

The dsl_file must define ``def tiled_reference(dims, tensors): ...`` and
return a torch.Tensor (typically via ``offchip_store``).
"""
import argparse
import inspect
import os
import sys
from pathlib import Path

import torch
import yaml

STEPDB_DIR = str(Path(__file__).resolve().parent)
NEXTSTEP_DIR = str(Path(__file__).resolve().parent.parent)
STEP_TL_SRC = str(Path(NEXTSTEP_DIR) / "step_tl" / "src")
STEP_TL_PROTO = str(Path(NEXTSTEP_DIR) / "step_tl" / "src" / "proto")
STEPGENFLOW = str(Path(NEXTSTEP_DIR) / "StepGenFlow12")

for p in (STEP_TL_SRC, STEP_TL_PROTO, STEPDB_DIR, STEPGENFLOW):
    if p not in sys.path:
        sys.path.insert(0, p)

from precompute import precompute_tensors
from validate_functional import IMPORT_SCAFFOLD  # IR scaffold mirrors validate_functional


# DSL scaffold matches StepGenFlow12 tools._DSL_IMPORTS: bring the step_dsl
# surface into the user namespace so the tiled_reference body can call DSL
# ops directly. We bind ``step_dsl`` to the StepGenFlow12 module so the
# class identity (StepTensor / StepRawTensor / Tile) matches what
# dsl_to_step's helpers will see when they recurse.
def _ensure_step_dsl_module():
    import sys as _sys
    if "step_dsl" not in _sys.modules:
        from src import step_dsl as _src_step_dsl
        _sys.modules["step_dsl"] = _src_step_dsl


DSL_SCAFFOLD = (
    "import torch\n"
    "import torch.nn.functional as F\n"
    "import math\n"
    "from step_dsl import *\n"
)


def load_config():
    config_path = os.path.join(STEPDB_DIR, "bench_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_dims(config, kernel_name, preset, cli_dims):
    if cli_dims is not None:
        return cli_dims
    assert kernel_name in config, f"Unknown kernel '{kernel_name}'"
    presets = config[kernel_name]["presets"]
    assert preset in presets, (
        f"Unknown preset '{preset}' for {kernel_name}. "
        f"Available: {sorted(presets.keys())}"
    )
    return dict(presets[preset])


def _parse_dims_arg(s):
    """Parse 'M=64,K=64,N=64' into a dict of int dims."""
    out = {}
    for part in s.split(","):
        k, v = part.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def _read_dsl_source(path):
    with open(path) as f:
        return f.read()


def run_dsl(dsl_source, dims, tensors):
    """Exec the DSL source and call tiled_reference(dims, wrapped_tensors).

    The wrap mirrors StepGenFlow12's _wrap_input_tensors: every torch.Tensor
    becomes a StepRawTensor so the DSL source ops accept them. We use the
    StepGenFlow12 StepRawTensor class so identity checks inside step_dsl match.
    """
    _ensure_step_dsl_module()
    from src.step_dsl import StepRawTensor

    wrapped = {k: (StepRawTensor(v) if isinstance(v, torch.Tensor) else v)
               for k, v in tensors.items()}

    namespace = {}
    exec(DSL_SCAFFOLD + "\n" + dsl_source, namespace)
    assert "tiled_reference" in namespace, (
        "DSL source must define `def tiled_reference(dims, tensors)`"
    )
    result = namespace["tiled_reference"](dims, wrapped)
    assert isinstance(result, torch.Tensor), (
        f"tiled_reference returned {type(result).__name__}; expected torch.Tensor "
        "(the final op should be offchip_store)"
    )
    return result


def run_ir_from_dsl(dsl_source, dims, tensors):
    """Translate DSL → IR via dsl_to_step, exec, then run the functional sim."""
    _ensure_step_dsl_module()
    from src.dsl_to_step import translate as dsl_to_step_translate
    from step_py.ops import StepOps
    from timing_and_emulator.functional import execute

    ir_source = dsl_to_step_translate(dsl_source)

    StepOps._counter = 0
    namespace = {}
    exec(IMPORT_SCAFFOLD + "\n" + ir_source, namespace)
    assert "build_graph" in namespace, (
        "DSL → IR translation must produce a `build_graph` function"
    )
    build_graph_fn = namespace["build_graph"]
    sig = inspect.signature(build_graph_fn)
    if "tensors" in sig.parameters:
        graph, output_op = build_graph_fn(dims, tensors)
    else:
        graph, output_op = build_graph_fn(dims)
    return execute(graph, output_op), ir_source


def run_reference(kernel_name, dims, tensors, config):
    ref_path = os.path.join(STEPDB_DIR, config[kernel_name]["problem"])
    with open(ref_path) as f:
        ref_code = f.read()
    namespace = {}
    exec(ref_code, namespace)
    assert "compute_gold" in namespace, f"compute_gold not in {ref_path}"
    compute_gold = namespace["compute_gold"]
    if "tensors" in inspect.signature(compute_gold).parameters:
        return compute_gold(dims, tensors)
    return compute_gold(dims)


def _compare(label, gold, candidate, tol=1e-5):
    assert gold.numel() == candidate.numel(), (
        f"{label}: numel mismatch — gold {tuple(gold.shape)} (n={gold.numel()}) "
        f"vs {label} {tuple(candidate.shape)} (n={candidate.numel()})"
    )
    g = gold.reshape(-1)
    c = candidate.reshape(-1)
    max_abs = (g - c).abs().max().item()
    scale = g.abs().max().item() + 1e-12
    rel = max_abs / scale
    status = "PASS" if rel < tol else "FAIL"
    print(f"  {status:4s}  {label:8s}  shape={tuple(candidate.shape)}  "
          f"max_abs={max_abs:.3e}  rel={rel:.3e}")
    return rel < tol


def validate(dsl_path, kernel_name, preset, cli_dims, dump_ir=None):
    config = load_config()
    dims = _resolve_dims(config, kernel_name, preset, cli_dims)

    print(f"DSL file: {dsl_path}")
    print(f"Kernel:   {kernel_name}")
    print(f"Dims:     {dims}")
    print()

    tensors = precompute_tensors(kernel_name, dims)
    dsl_source = _read_dsl_source(dsl_path)

    print("Running PyTorch reference ...")
    gold = run_reference(kernel_name, dims, tensors, config)

    print("Running DSL semantics ...")
    dsl_result = run_dsl(dsl_source, dims, tensors)

    print("Running DSL → IR (functional sim) ...")
    ir_result, ir_source = run_ir_from_dsl(dsl_source, dims, tensors)

    if dump_ir is not None:
        with open(dump_ir, "w") as f:
            f.write(ir_source)
        print(f"  (translated IR written to {dump_ir})")

    print()
    print("Comparison vs PyTorch gold:")
    dsl_ok = _compare("DSL", gold, dsl_result)
    ir_ok = _compare("IR", gold, ir_result)

    print()
    if dsl_ok and ir_ok:
        print("Both layers match the reference.")
    elif dsl_ok and not ir_ok:
        print("DSL passes but IR fails — the lowering exposes a mismatch the "
              "DSL semantics tolerate. Common culprit: leading-singleton "
              "stream dim consumed by `flatten`/`reshape_*` and not re-added "
              "before `offchip_store` (IR uses shape[1:] as tensor_shape_tiled).")
    elif not dsl_ok and ir_ok:
        print("IR passes but DSL fails — unusual; check that tiled_reference "
              "actually returns offchip_store's output (not an intermediate).")
    else:
        print("Both layers diverge from the reference — bug is in the DSL "
              "source itself, not the lowering.")
    return dsl_ok, ir_ok


def main():
    parser = argparse.ArgumentParser(
        description="Validate a step_dsl tiled_reference against PyTorch and "
                    "its STeP IR lowering."
    )
    parser.add_argument("dsl_path", help="Path to a .py file defining tiled_reference")
    parser.add_argument("kernel", help="Kernel name (used for compute_gold + precompute)")
    parser.add_argument("preset", nargs="?", help="Preset name from bench_config.yaml")
    parser.add_argument("--dims", help="Comma-separated dims, e.g. M=64,K=64,N=64,tile_m=16,...")
    parser.add_argument("--dump-ir", help="Write the translated IR to this path")
    args = parser.parse_args()

    cli_dims = _parse_dims_arg(args.dims) if args.dims else None
    assert args.preset or cli_dims, (
        "Provide either a preset name or --dims=M=...,K=..."
    )

    dsl_ok, ir_ok = validate(
        args.dsl_path, args.kernel, args.preset, cli_dims,
        dump_ir=args.dump_ir,
    )
    sys.exit(0 if (dsl_ok and ir_ok) else 1)


if __name__ == "__main__":
    main()
