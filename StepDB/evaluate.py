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
import inspect
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from loader import load_config, get_dims, list_kernels, list_presets, load_problem, load_step_impl
from precompute import precompute_tensors
from rust_sim_runner import RustSimDebugConfig, run_serialized_graph


STEP_TL_SRC = str(Path(__file__).resolve().parent.parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(__file__).resolve().parent.parent / "step_tl" / "src" / "proto")

SIM_TIMEOUT_SECONDS = 100000
# Tolerance for the sim-vs-gold correctness gate. Scales by max(|gold|) rather
# than per-element |gold[i]|, because the f32 accumulation noise on sim[i] is
# bounded by the *intermediates* getting summed into it (K-axis dot products
# of weights ~N(0,1) and activations) — not by gold[i]. When cancellation
# produces a small output from large intermediates (common in moe_routed,
# softmax tails), a per-element rtol*|gold[i]| budget collapses to ~0 and a
# bit-for-bit-noise-equivalent run gets flagged. Matches the metric in
# validate_functional.py; threshold is one decade looser because the rust
# path uses ndarray::dot whereas validate_functional.py runs torch.matmul
# (identical to the gold), so it sees slightly more accumulation drift.
REL_ERR_THRESHOLD = 1e-4


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
    rust_sim_classification: str | None = None
    rust_sim_log: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _write_result(work_dir: str, result: EvalResult) -> EvalResult:
    (Path(work_dir) / "result.json").write_text(result.to_json())
    return result


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
                    timing_only: bool = False,
                    step_impl_source: str | None = None,
                    max_total_compute_bw: int | None = None,
                    tensors_override: dict | None = None,
                    sim_timeout_seconds: float | None = None,
                    rust_sim_debug: bool = False,
                    rust_sim_log: str | None = None,
                    rust_stall_windows: int = 3) -> EvalResult:
    """Run the full evaluation pipeline for a single kernel pair + preset.

    Stages: exec -> simulate -> correctness -> success.
    When timing_only=True, skips correctness (stage 3) and disables functional sim.
    When step_impl_source is provided, it replaces StepDB's on-disk step_impl
    for this call (used by external autotuners that score generated kernels);
    dims, the reference module, and precompute still come from StepDB.
    When tensors_override is provided, build_graph receives it instead of
    StepDB's root-kernel precompute dict. Autotuners use this for isolated
    non-root node wrappers whose inputs are contract-local tensors.
    When max_total_compute_bw is set, every compute op's ``compute_bw`` is
    rescaled (in place) so the sum equals the budget before serialization —
    same routine as ``validate_timing.normalize_compute_bw``. Autotuners pass
    this so the rust sim runs against the same compute budget the analytical
    scorer used to rank candidates.
    ``sim_timeout_seconds`` overrides the default simulator subprocess timeout.
    Rust sim debug options stream/log instrumented DAM progress and attach a
    liveness classification to simulate-stage failures.
    """
    dims = get_dims(kernel_name, preset)
    ref_mod = None if timing_only else load_problem(kernel_name)
    step_code = step_impl_source if step_impl_source is not None else load_step_impl(kernel_name)

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
    # No os.chdir here: the parent process's cwd is shared global state, so
    # mutating it would race when multiple evaluate_kernel calls run
    # concurrently (e.g. autotune2's top-K rust promotion). The simulator
    # subprocess does its own chdir into work_dir; the parent stays put.

    from sim import serialize, SimConfig, HBMConfig
    from utils.gold_checking import reconstruct_numpy

    if "tensors" in inspect.signature(build_graph).parameters:
        tensors = (
            precompute_tensors(kernel_name, dims)
            if tensors_override is None else tensors_override
        )
        graph, output_op = build_graph(dims, tensors)
    else:
        graph, output_op = build_graph(dims)

    if max_total_compute_bw is not None:
        from validate_timing import normalize_compute_bw
        normalize_compute_bw(graph, max_total_compute_bw)

    pb_path = os.path.join(work_dir, "graph.pb")

    # channel_depth=1024 is large enough to absorb any in-flight token window
    # we've seen in LLM-generated kernels (Accum reductions up to a few hundred
    # tokens). The reducing-diamond deadlock from the original autotune2 hangs
    # only manifests when channel_depth < R; with depth >> R the broadcast
    # never stalls. Real hardware FIFOs are much shallower, so this is a
    # simulation-only relaxation that lets the autotuner score kernels by
    # cycles without spending hours hung on candidates that would deadlock at
    # depth=2 but run fine here. See validate_deadlock.py for the static check
    # we used to apply (kept as a diagnostic; no longer wired in).
    sim_config = SimConfig(channel_depth=10000000, functional_sim=not timing_only, mock_bf16=False)
    hbm_config = HBMConfig(
        addr_offset=64, channel_num=32,
        per_channel_latency=2, per_channel_init_interval=2,
        per_channel_outstanding=1, per_channel_start_up_time=14,
    )

    serialize(graph, pb_path, sim_config.functional_sim)

    timeout = (
        SIM_TIMEOUT_SECONDS
        if sim_timeout_seconds is None else float(sim_timeout_seconds)
    )
    sim_result = run_serialized_graph(
        work_dir=work_dir,
        graph_pb=pb_path,
        hbm_config=hbm_config,
        sim_config=sim_config,
        timeout_seconds=timeout,
        debug=RustSimDebugConfig(
            enabled=rust_sim_debug,
            log_path=rust_sim_log,
            stall_windows=rust_stall_windows,
        ),
    )

    if sim_result.timed_out:
        return _write_result(work_dir, EvalResult(
            kernel=kernel_name, preset=preset, stage="simulate",
            success=False, dims=dims,
            error_message=(
                f"Simulator timed out after {timeout:g} seconds. "
                f"{sim_result.summary()}\n"
                f"output tail:\n{sim_result.output[-2000:]}"
            ),
            rust_sim_classification=sim_result.classification,
            rust_sim_log=sim_result.log_path,
        ))

    if sim_result.returncode != 0:
        return _write_result(work_dir, EvalResult(
            kernel=kernel_name, preset=preset, stage="simulate", success=False,
            dims=dims,
            error_message=(
                f"Simulator failed (rc={sim_result.returncode}). "
                f"{sim_result.summary()}\n"
                f"output tail:\n{sim_result.output[-2000:]}"
            ),
            rust_sim_classification=sim_result.classification,
            rust_sim_log=sim_result.log_path,
        ))

    if sim_result.sim_json is None:
        return _write_result(work_dir, EvalResult(
            kernel=kernel_name, preset=preset, stage="simulate", success=False,
            dims=dims,
            error_message=(
                f"Simulator did not emit a JSON result. {sim_result.summary()}\n"
                f"output tail:\n{sim_result.output[-2000:]}"
            ),
            rust_sim_classification=sim_result.classification,
            rust_sim_log=sim_result.log_path,
        ))
    if not sim_result.sim_json["passed"]:
        return _write_result(work_dir, EvalResult(
            kernel=kernel_name, preset=preset, stage="simulate", success=False,
            dims=dims,
            error_message=(
                f"Simulator did not pass. {sim_result.summary()}\n"
                f"output tail:\n{sim_result.output[-2000:]}"
            ),
            rust_sim_classification=sim_result.classification,
            rust_sim_log=sim_result.log_path,
        ))
    cycles = sim_result.sim_json["cycles"]

    # --- Stage 3: correctness ---
    if timing_only:
        return _write_result(work_dir, EvalResult(
            kernel=kernel_name, preset=preset, stage="success", success=True,
            dims=dims, cycle_time=float(cycles),
            rust_sim_classification=sim_result.classification,
            rust_sim_log=sim_result.log_path,
        ))

    store_name = output_op.store_file_name
    store_path = os.path.join(work_dir, store_name)

    assert os.path.exists(f"{store_path}.npy"), (
        f"Simulation did not produce {store_name}.npy\noutput: {sim_result.output[-2000:]}"
    )

    if os.path.exists(f"{store_path}.json"):
        sim_output = reconstruct_numpy(store_path, delete_npy=False)
    else:
        sim_output = np.load(f"{store_path}.npy")

    sim_tensor = torch.from_numpy(sim_output).float()
    if "tensors" in inspect.signature(ref_mod.compute_gold).parameters:
        tensors = (
            precompute_tensors(kernel_name, dims)
            if tensors_override is None else tensors_override
        )
        gold = ref_mod.compute_gold(dims, tensors).float()
    else:
        gold = ref_mod.compute_gold(dims).float()

    assert sim_tensor.numel() == gold.numel(), (
        f"Element count mismatch: sim={sim_tensor.numel()} gold={gold.numel()}"
    )
    sim_tensor = sim_tensor.reshape(gold.shape)

    max_diff = (sim_tensor - gold).abs().max().item()
    gold_scale = gold.abs().max().item() + 1e-12
    rel_err = max_diff / gold_scale
    passed = rel_err < REL_ERR_THRESHOLD

    if not passed:
        return _write_result(work_dir, EvalResult(
            kernel=kernel_name, preset=preset, stage="correctness", success=False,
            dims=dims,
            error_message=f"Output incorrect: max_diff={max_diff}, rel_err={rel_err:.2e} (threshold {REL_ERR_THRESHOLD:.0e})",
            cycle_time=float(cycles), max_diff=max_diff,
            rust_sim_classification=sim_result.classification,
            rust_sim_log=sim_result.log_path,
        ))

    # --- Stage 4: success ---
    return _write_result(work_dir, EvalResult(
        kernel=kernel_name, preset=preset, stage="success", success=True,
        dims=dims, cycle_time=float(cycles), max_diff=max_diff,
        rust_sim_classification=sim_result.classification,
        rust_sim_log=sim_result.log_path,
    ))


def main():
    parser = argparse.ArgumentParser(description="Evaluate StepDB kernel pairs")
    parser.add_argument("kernel", nargs="?", help="Kernel name to evaluate")
    parser.add_argument("preset", nargs="?", help="Preset name from bench_config.yaml")
    parser.add_argument("--all", action="store_true", help="Evaluate all kernels, all presets")
    parser.add_argument("--all-presets", action="store_true", help="Evaluate all presets for one kernel")
    parser.add_argument("--list", action="store_true", help="List available kernels and presets")
    parser.add_argument("--timing-only", action="store_true",
                        help="Skip correctness check, run cycle-accurate timing only")
    parser.add_argument("--sim-timeout", type=float, default=None,
                        help="Override simulator subprocess timeout in seconds")
    parser.add_argument("--rust-sim-debug", action="store_true",
                        help="Stream Rust/DAM progress lines and print liveness context")
    parser.add_argument("--rust-sim-log", default=None,
                        help="Write combined Rust simulator stdout/stderr to this path")
    parser.add_argument("--rust-stall-windows", type=int, default=3,
                        help="Classify stall after this many zero-movement DAM progress windows")
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
        log_path = args.rust_sim_log
        if log_path and len(pairs) > 1:
            base = Path(log_path)
            log_path = str(base.with_name(f"{base.stem}_{name}_{preset}{base.suffix}"))
        result = evaluate_kernel(
            name,
            preset,
            timing_only=args.timing_only,
            sim_timeout_seconds=args.sim_timeout,
            rust_sim_debug=args.rust_sim_debug,
            rust_sim_log=log_path,
            rust_stall_windows=args.rust_stall_windows,
        )
        results.append(result)
        status = "PASS" if result.success else f"FAIL @ {result.stage}"
        cycles_str = f" ({result.cycle_time} cycles)" if result.cycle_time else ""
        print(f"  -> {status}{cycles_str}")
        if result.rust_sim_classification:
            print(f"  -> rust sim: {result.rust_sim_classification}")
        if result.rust_sim_log:
            print(f"  -> rust sim log: {result.rust_sim_log}")
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
