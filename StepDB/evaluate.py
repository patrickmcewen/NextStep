"""Evaluation harness for StepDB kernel pairs.

Evaluates a STeP kernel implementation against its PyTorch reference.
Dimensions come from bench_config.yaml presets (single source of truth).

Usage:
    python evaluate.py gemm_tile_mk small         # kernel + preset
    python evaluate.py gemm_tile_mk --all-presets  # all presets for one kernel
    python evaluate.py --all                       # all kernels, all presets
    python evaluate.py --list                      # list available kernels + presets
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from loader import load_config, get_dims, list_kernels, list_presets, load_problem, load_step_impl


STEP_TL_SRC = str(Path(__file__).resolve().parent.parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(__file__).resolve().parent.parent / "step_tl" / "src" / "proto")

SIM_TIMEOUT_SECONDS = 100000
RTOL = 1e-3
ATOL = 1e-3


@dataclass
class EvalResult:
    kernel: str
    preset: str
    stage: str          # "exec" | "simulate" | "correctness" | "success"
    success: bool
    dims: dict | None = None
    error_message: str | None = None
    cycle_time: float | None = None
    max_diff: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# Standard imports prepended to step_impl code so build_graph can use STeP ops
# without explicit imports (mirrors the LLM-generated code pattern).
IMPORT_SCAFFOLD = """\
import math
from math import ceil
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


def _strip_imports(code: str) -> str:
    """Remove import lines from step_impl code since we prepend our own."""
    lines = code.split("\n")
    result = []
    in_multiline_import = False
    for line in lines:
        stripped = line.strip()
        if in_multiline_import:
            if ")" in stripped:
                in_multiline_import = False
            continue
        if stripped.startswith(("import ", "from ")) and "(" in stripped and ")" not in stripped:
            in_multiline_import = True
            continue
        if stripped.startswith(("import ", "from ")):
            continue
        result.append(line)
    return "\n".join(result)


def evaluate_kernel(kernel_name: str, preset: str, work_dir: str | None = None,
                    timing_only: bool = False) -> EvalResult:
    """Run the full evaluation pipeline for a single kernel pair + preset.

    Stages: exec -> simulate -> correctness -> success.
    When timing_only=True, skips correctness (stage 3) and disables functional sim.
    """
    dims = get_dims(kernel_name, preset)
    ref_mod = None if timing_only else load_problem(kernel_name)
    step_code = load_step_impl(kernel_name)

    if work_dir is None:
        work_dir = str(Path(__file__).resolve().parent / "kernels" / kernel_name / f"_work_{preset}")
    os.makedirs(work_dir, exist_ok=True)

    # Ensure step_tl/src and proto are on the path before exec so imports resolve.
    # proto/ must come AFTER src/ so that `from proto import X` works via package,
    # but bare `import ops_pb2` inside generated pb2 files also resolves.
    # Remove any old PytorchStepFlow paths to avoid proto module conflicts.
    sys.path = [p for p in sys.path if "PytorchStepFlow/" not in p or "PytorchStepFlowNew" in p]
    if STEP_TL_SRC not in sys.path:
        sys.path.insert(0, STEP_TL_SRC)
    if STEP_TL_PROTO not in sys.path:
        sys.path.append(STEP_TL_PROTO)

    # --- Stage 1: exec — load and execute the STeP impl ---
    full_code = IMPORT_SCAFFOLD + step_code#_strip_imports(step_code)
    (Path(work_dir) / "full_body.py").write_text(full_code)

    namespace = {}
    exec(full_code, namespace)

    build_graph = namespace.get("build_graph")
    assert build_graph is not None, "step_impl.py does not define build_graph"

    # --- Stage 2: simulate ---
    orig_dir = os.getcwd()

    from sim import serialize, SimConfig, HBMConfig
    from utils.gold_checking import reconstruct_numpy

    graph, output_op = build_graph(dims)

    os.chdir(work_dir)
    pb_path = os.path.join(os.getcwd(), "graph.pb")

    sim_config = SimConfig(channel_depth=2, functional_sim=not timing_only, mock_bf16=False)
    hbm_config = HBMConfig(
        addr_offset=64, channel_num=32,
        per_channel_latency=2, per_channel_init_interval=2,
        per_channel_outstanding=1, per_channel_start_up_time=14,
    )

    serialize(graph, pb_path, sim_config.functional_sim)

    sim_runner_script = (
        "import json, sys, os\n"
        "os.chdir(sys.argv[1])\n"
        "from sim import HBMConfig, SimConfig\n"
        "import step_perf\n"
        "pb_path = sys.argv[2]\n"
        "hbm_cfg = json.loads(sys.argv[3])\n"
        "sim_cfg = json.loads(sys.argv[4])\n"
        "hbm = HBMConfig(**hbm_cfg)\n"
        "sim = SimConfig(**sim_cfg)\n"
        "ret = step_perf.run_graph(pb_path, False, hbm, sim, None)\n"
        "if len(ret) == 4:\n"
        "    _, cycles, dur_ms, dur_s = ret\n"
        "elif len(ret) == 2:\n"
        "    _, cycles = ret\n"
        "    dur_ms, dur_s = 0.0, 0.0\n"
        "else:\n"
        "    raise RuntimeError(f'Unexpected return: {ret}')\n"
        "print(json.dumps({'cycles': cycles, 'dur_ms': dur_ms, 'dur_s': dur_s}))\n"
    )

    pythonpath = STEP_TL_SRC + ":" + STEP_TL_PROTO + ":" + os.environ.get("PYTHONPATH", "")
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath

    proc = subprocess.run(
        [sys.executable, "-c", sim_runner_script,
         os.getcwd(), pb_path,
         json.dumps(asdict(hbm_config)),
         json.dumps({"channel_depth": sim_config.channel_depth,
                      "functional_sim": sim_config.functional_sim,
                      "mock_bf16": sim_config.mock_bf16})],
        capture_output=True, text=True, timeout=SIM_TIMEOUT_SECONDS,
        env=env,
    )

    os.chdir(orig_dir)

    if proc.returncode != 0:
        return EvalResult(
            kernel=kernel_name, preset=preset, stage="simulate", success=False,
            dims=dims,
            error_message=f"Simulator failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}",
        )

    sim_result = json.loads(proc.stdout.strip().split("\n")[-1])
    cycles = sim_result["cycles"]

    # --- Stage 3: correctness ---
    if timing_only:
        result = EvalResult(
            kernel=kernel_name, preset=preset, stage="success", success=True,
            dims=dims, cycle_time=float(cycles),
        )
        (Path(work_dir) / "result.json").write_text(result.to_json())
        return result

    store_name = output_op.store_file_name
    store_path = os.path.join(work_dir, store_name)

    assert os.path.exists(f"{store_path}.npy"), (
        f"Simulation did not produce {store_name}.npy\nstderr: {proc.stderr[-2000:]}"
    )

    if os.path.exists(f"{store_path}.json"):
        sim_output = reconstruct_numpy(store_path, delete_npy=False)
    else:
        sim_output = np.load(f"{store_path}.npy")

    sim_tensor = torch.from_numpy(sim_output).float()
    gold = ref_mod.compute_gold(dims).float()

    assert sim_tensor.numel() == gold.numel(), (
        f"Element count mismatch: sim={sim_tensor.numel()} gold={gold.numel()}"
    )
    sim_tensor = sim_tensor.reshape(gold.shape)

    max_diff = (sim_tensor - gold).abs().max().item()
    passed = torch.allclose(sim_tensor, gold, rtol=RTOL, atol=ATOL)

    if not passed:
        return EvalResult(
            kernel=kernel_name, preset=preset, stage="correctness", success=False,
            dims=dims,
            error_message=f"Output incorrect: max_diff={max_diff}",
            cycle_time=float(cycles), max_diff=max_diff,
        )

    # --- Stage 4: success ---
    result = EvalResult(
        kernel=kernel_name, preset=preset, stage="success", success=True,
        dims=dims, cycle_time=float(cycles), max_diff=max_diff,
    )
    (Path(work_dir) / "result.json").write_text(result.to_json())
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate StepDB kernel pairs")
    parser.add_argument("kernel", nargs="?", help="Kernel name to evaluate")
    parser.add_argument("preset", nargs="?", help="Preset name from bench_config.yaml")
    parser.add_argument("--all", action="store_true", help="Evaluate all kernels, all presets")
    parser.add_argument("--all-presets", action="store_true", help="Evaluate all presets for one kernel")
    parser.add_argument("--list", action="store_true", help="List available kernels and presets")
    parser.add_argument("--timing-only", action="store_true",
                        help="Skip correctness check, run cycle-accurate timing only")
    args = parser.parse_args()

    if args.list:
        for name in list_kernels():
            presets = ", ".join(list_presets(name))
            print(f"  {name}: [{presets}]")
        return

    # Build list of (kernel, preset) pairs to evaluate
    pairs = []
    if args.all:
        for name in list_kernels():
            for preset in list_presets(name):
                pairs.append((name, preset))
    elif args.all_presets:
        assert args.kernel, "Specify a kernel name with --all-presets"
        for preset in list_presets(args.kernel):
            pairs.append((args.kernel, preset))
    else:
        assert args.kernel and args.preset, "Specify <kernel> <preset>, or use --all / --all-presets"
        pairs.append((args.kernel, args.preset))

    results = []
    for name, preset in pairs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} / {preset}")
        print(f"{'='*60}")
        result = evaluate_kernel(name, preset, timing_only=args.timing_only)
        results.append(result)
        status = "PASS" if result.success else f"FAIL @ {result.stage}"
        cycles_str = f" ({result.cycle_time} cycles)" if result.cycle_time else ""
        print(f"  -> {status}{cycles_str}")
        if result.error_message:
            print(f"  -> {result.error_message[:200]}")

    # Summary
    passed = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(results)} passed")
    for r in results:
        mark = "PASS" if r.success else "FAIL"
        print(f"  [{mark}] {r.kernel} / {r.preset}")


if __name__ == "__main__":
    main()
