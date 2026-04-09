"""Validate the functional simulation against PyTorch reference kernels.

Usage:
    python validate_functional.py                    # all seed kernels, first preset only
    python validate_functional.py gemm               # single kernel, all its presets
    python validate_functional.py gemm small          # specific kernel + preset
    python validate_functional.py --all               # all seed kernels, every preset
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

STEPDB_DIR = str(Path(__file__).resolve().parent)
STEP_TL_SRC = str(Path(__file__).resolve().parent.parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(__file__).resolve().parent.parent / "step_tl" / "src" / "proto")

sys.path.insert(0, STEP_TL_SRC)
sys.path.insert(0, STEP_TL_PROTO)

# Imports prepended to step_impl code (mirrors validate_timing.py)
IMPORT_SCAFFOLD = """\
import sys
import random
import math
from math import ceil
from pathlib import Path
import torch
import numpy as np

SEED = 42

from graph.graph import MultiDiGraph as Graph
from rewrite.broadcast import infer_broadcast
from step_py.datatype import (
    Float32, Float16, Uint32, Uint64, Bool,
    Tile, DynTile, Buffer, MultiHot, Index, Stream,
)
from step_py.dyndim import DynDim
from step_py.ops import (
    LinearOffChipLoad, LinearOffChipLoadRef, DynLinearOffChipLoad,
    RandomOffChipLoad, RandomOffChipStore,
    OffChipStore, DynOffChipStore,
    BinaryMap, UnaryMap, BinaryMapAccum, Accum,
    Promote, PromoteOuter, ExpandRef, RepeatRef, RepeatStatic,
    Flatten, Reshape, ReshapePadStream,
    Bufferize, Streamify, DynStreamify,
    Broadcast, Parallelize, StaticReassemble,
    FlatPartition, FlatReassemble, EagerMerge,
    RetileStreamify, FlatmapFilterRowStreamify, FlatmapCounter,
    MockStreamOp,
)
from step_py.utility_ops import (
    SelectGen, ExpertAddrGen, MetadataGen, CacheReadAddrGen,
    FilterLastTile, PrinterContext, ConsumerContext,
)
from step_py.functions import map_fn, map_accum_fn, accum_fn, init_fn
from step_py.functions.map_fn import (
    Matmul, DynMatmul, Mul, MulImmediate, IsEqual, Add, AddImmediate,
    SubImmediate, Div, Silu, RowWiseSum, Exp, Pow2, Rsqrt, Square, MaskRow,
    SetOffset, RowWiseAppend, CacheWriteAddrGen, SelectToScalar, ToConstInt,
)
from step_py.functions.map_accum_fn import (
    Matmul as MapAccumMatmul, DynMatmul as MapAccumDynMatmul,
)
from step_py.functions.accum_fn import (
    Mul as AccumMul, Add as AccumAdd,
    RetileRow, RetileCol, SignalReqAllRead,
)
from step_py.functions.init_fn import Zero, Empty, DynEmpty
from step_py.kernels.linear import Linear, LinearTileConfig
"""


def load_config():
    config_path = os.path.join(STEPDB_DIR, "bench_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# Prefixes of import lines the scaffold already provides — only these are stripped.
_SCAFFOLD_PREFIXES = (
    "import math", "from math ", "import torch", "import numpy",
    "import sys", "import random", "from pathlib ",
    "from graph.", "from rewrite.", "from step_py.", "from step_py ",
)


def _strip_imports(code):
    """Remove import lines already provided by IMPORT_SCAFFOLD; preserve all others."""
    lines = code.split("\n")
    result = []
    in_multiline = False
    skip_multiline = False
    for line in lines:
        s = line.strip()
        if in_multiline:
            if ")" in s:
                in_multiline = False
                if skip_multiline:
                    continue
            if skip_multiline:
                continue
            result.append(line)
            continue
        if s.startswith(("import ", "from ")) and s != "":
            is_scaffold = any(s.startswith(p) for p in _SCAFFOLD_PREFIXES)
            if "(" in s and ")" not in s:
                in_multiline = True
                skip_multiline = is_scaffold
                if is_scaffold:
                    continue
            elif is_scaffold:
                continue
        result.append(line)
    return "\n".join(result)


def build_graph_from_impl(kernel_name, dims, config):
    """Build the STeP graph by exec'ing the step_impl code."""
    impl_path = os.path.join(STEPDB_DIR, config[kernel_name]["step_impl"])
    with open(impl_path) as f:
        impl_code = f.read()

    full_code = IMPORT_SCAFFOLD + "\n" + _strip_imports(impl_code)
    namespace = {}
    exec(full_code, namespace)
    assert "build_graph" in namespace, f"build_graph not found in {impl_path}"
    graph, output_op = namespace["build_graph"](dims)
    return graph, output_op


def run_reference(kernel_name, dims, config):
    """Run the PyTorch reference and return the gold tensor."""
    ref_path = os.path.join(STEPDB_DIR, config[kernel_name]["problem"])
    with open(ref_path) as f:
        ref_code = f.read()

    namespace = {}
    exec(ref_code, namespace)
    assert "compute_gold" in namespace, f"compute_gold not found in {ref_path}"
    return namespace["compute_gold"](dims)


def run_functional_sim(graph, output_op):
    """Run the functional simulation and return the output tensor."""
    from step_py.functional import execute
    return execute(graph, output_op)


def validate_kernel(kernel_name, preset, config):
    """Validate one kernel+preset. Returns (kernel, preset, match, max_err, msg)."""
    dims = dict(config[kernel_name]["presets"][preset])

    # Reset the StepOps counter so instance IDs are deterministic
    from step_py.ops import StepOps
    StepOps._counter = 0
    graph, output_op = build_graph_from_impl(kernel_name, dims, config)

    StepOps._counter = 0
    graph2, output_op2 = build_graph_from_impl(kernel_name, dims, config)

    gold = run_reference(kernel_name, dims, config)
    sim = run_functional_sim(graph2, output_op2)

    # Compare
    assert gold.shape == sim.shape, (
        f"Shape mismatch: gold {gold.shape} vs sim {sim.shape}"
    )

    max_err = (gold - sim).abs().max().item()
    gold_scale = gold.abs().max().item() + 1e-12
    rel_err = max_err / gold_scale

    # Float32 matmul accumulation introduces errors proportional to dimension
    # size.  Use relative comparison: rel_err < 1e-5 covers typical float32
    # differences while still catching graph-level correctness bugs.
    match = rel_err < 1e-5
    msg = f"max_abs_err={max_err:.2e}  rel_err={rel_err:.2e}"
    return kernel_name, preset, match, max_err, msg


def _build_job_list(config, args):
    """Return list of (kernel, preset) pairs to validate."""
    seed_kernels = [k for k, v in config.items() if v.get("origin") == "seed"]
    if args.kernel and args.preset:
        return [(args.kernel, args.preset)]
    elif args.kernel:
        assert args.kernel in config, f"Unknown kernel: {args.kernel}"
        return [(args.kernel, p) for p in config[args.kernel]["presets"]]
    elif args.all:
        return [(k, p) for k in seed_kernels for p in config[k]["presets"]]
    else:
        return [(k, list(config[k]["presets"].keys())[0]) for k in seed_kernels]


def main():
    parser = argparse.ArgumentParser(description="Validate functional simulation")
    parser.add_argument("kernel", nargs="?", help="Kernel name")
    parser.add_argument("preset", nargs="?", help="Preset name")
    parser.add_argument("--all", action="store_true", help="All seed kernels, every preset")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config()
    jobs = _build_job_list(config, args)
    print(f"Running {len(jobs)} functional validation(s)\n")

    results = []
    for kernel, preset in jobs:
        status = ""
        try:
            kernel, preset, match, max_err, msg = validate_kernel(kernel, preset, config)
            status = "PASS" if match else "FAIL"
            print(f"  {status:4s}  {kernel:30s} {preset:15s}  {msg}")
            results.append((kernel, preset, match, max_err, msg))
        except Exception as e:
            status = "ERR"
            print(f"  {status:4s}  {kernel:30s} {preset:15s}  {e}")
            results.append((kernel, preset, False, float("inf"), str(e)))

    # Summary
    passed = sum(1 for _, _, m, _, _ in results if m)
    total = len(results)
    print(f"\n{'='*70}")
    print(f"  {passed}/{total} passed")
    if passed < total:
        print("  Failures:")
        for k, p, m, _, msg in results:
            if not m:
                print(f"    {k}/{p}: {msg}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
