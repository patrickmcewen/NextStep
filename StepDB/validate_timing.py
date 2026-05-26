"""Validate the analytical timing model against the cycle-accurate simulator.

Usage:
    python validate_timing.py                    # all seed kernels, first preset only
    python validate_timing.py gemm               # single kernel, all its presets
    python validate_timing.py gemm small         # specific kernel + preset
    python validate_timing.py --all-small        # all seed kernels, small presets only
    python validate_timing.py --all              # all seed kernels, every preset
    python validate_timing.py --all -j 8         # all seed kernels, every preset, 8 workers
"""
import argparse
import inspect
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sympy
import yaml

STEPDB_DIR = str(Path(__file__).resolve().parent)
STEP_TL_SRC = str(Path(__file__).resolve().parent.parent / "step_tl" / "src")
STEP_TL_PROTO = str(Path(__file__).resolve().parent.parent / "step_tl" / "src" / "proto")
SIM_TIMEOUT_SECONDS = 100000
_serialize_lock = threading.Lock()

# Ensure imports work
sys.path.insert(0, STEP_TL_SRC)
sys.path.insert(0, STEP_TL_PROTO)
sys.path.insert(0, STEPDB_DIR)

from precompute import precompute_tensors
from rust_sim_runner import RustSimDebugConfig, run_serialized_graph
from timing_and_emulator.timing import DEFAULT_HW_CONFIG

# Standard imports prepended to step_impl code
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


def load_config():
    config_path = os.path.join(STEPDB_DIR, "bench_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _strip_imports(code):
    """Remove import lines from step_impl code since we prepend our own."""
    lines = code.split("\n")
    result = []
    in_multiline = False
    for line in lines:
        s = line.strip()
        if in_multiline:
            if ")" in s:
                in_multiline = False
            continue
        if s.startswith(("import ", "from ")) and s != "":
            if "(" in s and ")" not in s:
                in_multiline = True
            continue
        result.append(line)
    return "\n".join(result)


def _sym_to_int(expr):
    if hasattr(expr, "free_symbols") and expr.free_symbols:
        expr = expr.xreplace({s: 1 for s in expr.free_symbols})
    return int(sympy.N(expr))


def _fmt_bytes(n):
    if n < 1024:
        return f"{n}B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f}KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f}MB"
    return f"{n / 1024 ** 3:.1f}GB"


def compute_memory_totals(result, pmu_buffer_bytes):
    """Sum on-chip / off-chip memory across nodes.

    Mirrors `_compute_memory_totals` in StepGenFlow9/src/autotune.py so the
    numbers reported here match what the autotuner surfaces in its status,
    progress, and result JSONs.
    """
    info = result["per_node"]
    sym_subs = result.get("sym_subs", {}) or {}

    def _sub(expr):
        if sym_subs and hasattr(expr, "free_symbols") and expr.free_symbols:
            return expr.xreplace(sym_subs)
        return expr

    total_on_chip = 0
    total_off_chip = 0
    for _nid, i in info.items():
        n = i["node"]
        total_on_chip += _sym_to_int(_sub(n.on_chip_requirement(count_fifos=False)))
        total_off_chip += _sym_to_int(_sub(n.off_chip_traffic()))

    pmu_pct = (100.0 * total_on_chip / pmu_buffer_bytes) if pmu_buffer_bytes else None
    return {
        "on_chip_bytes": total_on_chip,
        "off_chip_bytes": total_off_chip,
        "pmu_buffer_bytes": pmu_buffer_bytes,
        "pmu_utilization_pct": pmu_pct,
    }


def normalize_compute_bw(graph, max_total_compute_bw):
    """Rescale every compute op's `compute_bw` so their sum equals the budget.

    Mirrors `_normalize_compute_bw` in StepGenFlow9/src/autotune.py so the
    analytical timing here matches what the autotuner reports. Mutates the
    graph in place; returns (sum_before, sum_after) for logging.
    """
    assert max_total_compute_bw >= 1, f"max_total_compute_bw must be >= 1, got {max_total_compute_bw}"
    compute_nodes = [n for n in graph.nodes if hasattr(n, "compute_bw")]
    if not compute_nodes:
        print("[normalize_compute_bw] note: graph has no compute ops exposing compute_bw; skipping rescale")
        return 0, 0
    old_sum = sum(n.compute_bw for n in compute_nodes)
    assert old_sum >= 1, "Sum of compute_bw across compute ops is zero — invalid graph"
    scale = max_total_compute_bw / old_sum
    new_sum = 0
    for n in compute_nodes:
        n.compute_bw = max(1, int(round(n.compute_bw * scale)))
        new_sum += n.compute_bw
    return old_sum, new_sum


def build_graph_from_impl(kernel_name, dims, config):
    """Build the STeP graph by exec'ing the step_impl code."""
    impl_path = os.path.join(STEPDB_DIR, config[kernel_name]["step_impl"])
    with open(impl_path) as f:
        impl_code = f.read()

    full_code = IMPORT_SCAFFOLD + "\n" + impl_code#_strip_imports(impl_code)
    namespace = {}
    exec(full_code, namespace)
    assert "build_graph" in namespace, f"build_graph not found in {impl_path}"
    build_graph_fn = namespace["build_graph"]
    if "tensors" in inspect.signature(build_graph_fn).parameters:
        tensors = precompute_tensors(kernel_name, dims)
        graph, output_op = build_graph_fn(dims, tensors)
    else:
        graph, output_op = build_graph_fn(dims)
    return graph, output_op


def run_analytical_model(graph, hw_config=None, sym_subs=None):
    """Run the analytical timing model and return predicted cycles.

    Args:
        sym_subs: dict mapping sympy symbol names to concrete values.
            If None and expression has free symbols, assumes uniform
            distribution (each symbolic dim gets value 1).
    """
    from timing_and_emulator.timing import analyze_timing
    result = analyze_timing(graph, hw_config)
    total = result["total_cycles"]

    # Substitute symbolic dims if needed
    if total.free_symbols:
        if sym_subs is None:
            # Use expected-value substitutions computed by analyze_timing
            # (derived from FlatPartition input_N_fire / num_consumers)
            sym_subs = result.get("sym_subs") or {s: 1 for s in total.free_symbols}
        total = total.subs(sym_subs)
        # Also substitute in per-node info for consistency
        for nid in result["per_node"]:
            for key in ("end", "fto", "st", "OCI", "OTI", "ICI", "ICD"):
                val = result["per_node"][nid].get(key)
                if val is not None and hasattr(val, 'free_symbols') and val.free_symbols:
                    result["per_node"][nid][key] = val.subs(sym_subs)

    total_val = int(sympy.N(total))
    return total_val, result


def run_simulator(
    graph,
    output_op,
    work_dir,
    *,
    sim_timeout_seconds=None,
    rust_sim_debug=False,
    rust_sim_log=None,
    rust_stall_windows=3,
):
    """Run the cycle-accurate simulator and return actual cycles."""
    from sim import serialize, SimConfig, HBMConfig

    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    pb_path = os.path.join(work_dir, "graph.pb")

    sim_config = SimConfig(channel_depth=10000000, functional_sim=False, mock_bf16=False)
    hbm_config = HBMConfig(
        addr_offset=64, channel_num=32,
        per_channel_latency=2, per_channel_init_interval=2,
        per_channel_outstanding=1, per_channel_start_up_time=0,
    )

    # serialize writes .npy files to cwd using relative paths, so we must
    # chdir into work_dir. Lock protects the cwd change across threads.
    with _serialize_lock:
        orig_dir = os.getcwd()
        os.chdir(work_dir)
        serialize(graph, pb_path, sim_config.functional_sim)
        os.chdir(orig_dir)

    timeout = SIM_TIMEOUT_SECONDS if sim_timeout_seconds is None else float(sim_timeout_seconds)
    result = run_serialized_graph(
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

    assert not result.timed_out, (
        f"Simulator timed out after {timeout:g} seconds.\n"
        f"{result.summary()}\noutput tail:\n{result.output[-2000:]}"
    )
    assert result.returncode == 0, (
        f"Simulator failed (rc={result.returncode}):\n"
        f"{result.summary()}\noutput tail:\n{result.output[-2000:]}"
    )
    assert result.sim_json is not None, (
        f"Simulator did not emit a JSON result.\n"
        f"{result.summary()}\noutput tail:\n{result.output[-2000:]}"
    )

    # The Rust simulator catches panics inside worker threads and still returns
    # (passed=False, cycles=<elapsed-at-time-of-panic>). If we ignore `passed`,
    # we end up comparing the analytical prediction against a crashed sim and
    # report a nonsense error. Surface the panic lines so the root cause is
    # visible instead of buried in megabytes of stderr.
    assert result.sim_json["passed"], (
        f"Simulator did not pass (cycles before crash: {result.sim_json['cycles']}).\n"
        f"{result.summary()}\n"
        f"Panic excerpts:\n"
        + "\n".join(
            l for l in result.output.split("\n")
            if "panicked" in l or "assertion" in l
        )[-2000:]
    )

    cycles = int(result.sim_json["cycles"])

    # Parse TRACE_EVENT lines from stderr if STEP_TRACE is set
    trace_events = []
    if os.environ.get("STEP_TRACE"):
        for line in result.output.split("\n"):
            if line.startswith("TRACE_EVENT|"):
                parts = line.split("|")
                assert len(parts) == 6, f"Bad trace line: {line}"
                trace_events.append({
                    "name": parts[1],
                    "id": int(parts[2]),
                    "start": int(parts[3]),
                    "end": int(parts[4]),
                    "is_stop": parts[5] == "true",
                })

    if trace_events:
        return cycles, trace_events
    if rust_sim_debug:
        print("\n=== rust-sim liveness summary ===")
        print(result.summary())
    return cycles


def validate_kernel(
    kernel_name,
    preset,
    config,
    verbose=False,
    max_compute_bw=None,
    show_memory=False,
    sim_timeout_seconds=None,
    rust_sim_debug=False,
    rust_sim_log=None,
    rust_stall_windows=3,
):
    """Validate one kernel+preset. Returns (kernel, preset, predicted, actual, error_pct, detail)."""
    dims = dict(config[kernel_name]["presets"][preset])
    graph, output_op = build_graph_from_impl(kernel_name, dims, config)

    if max_compute_bw is not None:
        old_sum, new_sum = normalize_compute_bw(graph, max_compute_bw)
        print(f"Rescaled compute_bw: sum {old_sum} -> {new_sum} (target={max_compute_bw})")

    predicted, detail = run_analytical_model(graph)

    print(f"Predicted: {predicted}")

    if show_memory:
        mem = compute_memory_totals(detail, DEFAULT_HW_CONFIG["pmu_buffer_bytes"])
        detail["memory"] = mem
        pmu_str = f" ({mem['pmu_utilization_pct']:.1f}% of PMU={mem['pmu_buffer_bytes']} B)" if mem["pmu_utilization_pct"] is not None else ""
        print(f"  on-chip:  {mem['on_chip_bytes']} B{pmu_str}")
        print(f"  off-chip: {mem['off_chip_bytes']} B")

    work_dir = os.path.join(STEPDB_DIR, "seed_kernels", kernel_name, f"_work_timing_{preset}")
    actual = run_simulator(
        graph,
        output_op,
        work_dir,
        sim_timeout_seconds=sim_timeout_seconds,
        rust_sim_debug=rust_sim_debug,
        rust_sim_log=rust_sim_log,
        rust_stall_windows=rust_stall_windows,
    )

    error_pct = (predicted - actual) / max(actual, 1) * 100
    return kernel_name, preset, predicted, actual, error_pct, detail


def print_kernel_result(kernel_name, preset, config, predicted, actual, error_pct, detail, verbose=False):
    """Print detailed result for a single kernel+preset."""
    dims = dict(config[kernel_name]["presets"][preset])
    print(f"\n{'='*60}")
    print(f"  {kernel_name} / {preset}  dims={dims}")
    print(f"{'='*60}")
    print(f"  Analytical model: {predicted} cycles")
    if verbose:
        for nid, ninfo in detail["per_node"].items():
            node = ninfo["node"]
            print(f"    {str(node):50s}  st={ninfo['st']}  end={ninfo['end']}  OCI={ninfo['OCI']}  OTI={ninfo['OTI']}")
    print(f"  Cycle-accurate sim: {actual} cycles")
    print(f"  Error: {error_pct:.1f}%")
    if "memory" in detail:
        mem = detail["memory"]
        pmu_str = f" ({mem['pmu_utilization_pct']:.1f}% of PMU={mem['pmu_buffer_bytes']} B)" if mem["pmu_utilization_pct"] is not None else ""
        print(f"  On-chip:  {mem['on_chip_bytes']} B{pmu_str}")
        print(f"  Off-chip: {mem['off_chip_bytes']} B")


def _build_job_list(config, args):
    """Return list of (kernel, preset) pairs to validate."""
    seed_kernels = [k for k, v in config.items() if v.get("origin") == "seed"]
    SMALL_PRESETS = {"small", "tiny", "square"}

    if args.kernel and args.preset:
        return [(args.kernel, args.preset)]
    elif args.kernel:
        assert args.kernel in config, f"Unknown kernel: {args.kernel}"
        return [(args.kernel, p) for p in config[args.kernel]["presets"]]
    elif args.all:
        return [(k, p) for k in seed_kernels for p in config[k]["presets"]]
    elif args.all_small:
        return [(k, p) for k in seed_kernels for p in config[k]["presets"] if p in SMALL_PRESETS]
    else:
        return [(k, list(config[k]["presets"].keys())[0]) for k in seed_kernels]


def _log_path(log_dir, kernel, preset):
    if log_dir is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    safe_kernel = kernel.replace("/", "__")
    safe_preset = preset.replace("/", "__")
    return os.path.join(log_dir, f"{safe_kernel}_{safe_preset}.log")


def _run_serial(
    jobs,
    config,
    verbose,
    max_compute_bw=None,
    show_memory=False,
    sim_timeout_seconds=None,
    rust_sim_debug=False,
    rust_sim_log_dir=None,
    rust_stall_windows=3,
):
    """Run jobs sequentially with full output."""
    results = []
    skipped = []
    for kernel, preset in jobs:
        #try:
        r = validate_kernel(
            kernel,
            preset,
            config,
            verbose,
            max_compute_bw=max_compute_bw,
            show_memory=show_memory,
            sim_timeout_seconds=sim_timeout_seconds,
            rust_sim_debug=rust_sim_debug,
            rust_sim_log=_log_path(rust_sim_log_dir, kernel, preset),
            rust_stall_windows=rust_stall_windows,
        )
        kernel, preset, predicted, actual, error_pct, detail = r
        print_kernel_result(kernel, preset, config, predicted, actual, error_pct, detail, verbose)
        results.append((kernel, preset, predicted, actual, error_pct))
        #except Exception as e:
        #    print(f"  SKIPPED {kernel}/{preset}: {e}")
        #    skipped.append((kernel, preset, str(e)))
    return results, skipped


def _run_parallel(
    jobs,
    config,
    verbose,
    max_workers,
    max_compute_bw=None,
    show_memory=False,
    sim_timeout_seconds=None,
    rust_sim_debug=False,
    rust_sim_log_dir=None,
    rust_stall_windows=3,
):
    """Run jobs in parallel, logging each as it completes."""
    results = []
    skipped = []
    print_lock = threading.Lock()
    total = len(jobs)
    done_count = [0]  # mutable counter for closure

    def _worker(kernel, preset):
        return validate_kernel(
            kernel,
            preset,
            config,
            verbose,
            max_compute_bw=max_compute_bw,
            show_memory=show_memory,
            sim_timeout_seconds=sim_timeout_seconds,
            rust_sim_debug=rust_sim_debug,
            rust_sim_log=_log_path(rust_sim_log_dir, kernel, preset),
            rust_stall_windows=rust_stall_windows,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {
            pool.submit(_worker, kernel, preset): (kernel, preset)
            for kernel, preset in jobs
        }
        for future in as_completed(future_to_job):
            kernel, preset = future_to_job[future]
            done_count[0] += 1
            idx = done_count[0]
            try:
                kernel, preset, predicted, actual, error_pct, detail = future.result()
                results.append((kernel, preset, predicted, actual, error_pct))
                mem_str = ""
                if "memory" in detail:
                    m = detail["memory"]
                    pct = f"({m['pmu_utilization_pct']:.1f}% PMU)" if m["pmu_utilization_pct"] is not None else ""
                    mem_str = f"  on_chip={_fmt_bytes(m['on_chip_bytes'])}{pct} off_chip={_fmt_bytes(m['off_chip_bytes'])}"
                with print_lock:
                    print(f"  [{idx}/{total}] {kernel:30s} {preset:15s}  pred={predicted:>8d}  actual={actual:>8d}  err={error_pct:.1f}%{mem_str}")
                    if verbose:
                        for nid, ninfo in detail["per_node"].items():
                            node = ninfo["node"]
                            print(f"           {str(node):50s}  st={ninfo['st']}  end={ninfo['end']}  OCI={ninfo['OCI']}  OTI={ninfo['OTI']}")
            except Exception as e:
                skipped.append((kernel, preset, str(e)))
                with print_lock:
                    print(f"  [{idx}/{total}] {kernel:30s} {preset:15s}  SKIPPED: {e}")

    return results, skipped


def main():
    parser = argparse.ArgumentParser(description="Validate analytical timing model")
    parser.add_argument("kernel", nargs="?", help="Kernel name")
    parser.add_argument("preset", nargs="?", help="Preset name")
    parser.add_argument("--all-small", action="store_true", help="All seed kernels, small presets only")
    parser.add_argument("--all", action="store_true", help="All seed kernels, every preset")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Parallel workers (default: 1 = serial)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--max-compute-bw", type=int, default=None,
        help="Rescale every compute op's compute_bw so the sum equals this budget "
             "(matches the autotuner's _normalize_compute_bw). Default: no rescaling.",
    )
    parser.add_argument(
        "--show-memory", action="store_true",
        help="Compute and report on-chip / off-chip memory totals (matches the "
             "autotuner's _compute_memory_totals).",
    )
    parser.add_argument(
        "--sim-timeout", type=float, default=None,
        help="Override simulator subprocess timeout in seconds.",
    )
    parser.add_argument(
        "--rust-sim-debug", action="store_true",
        help="Stream Rust/DAM progress lines and include liveness classification on failures.",
    )
    parser.add_argument(
        "--rust-sim-log-dir", default=None,
        help="Write combined Rust simulator stdout/stderr logs under this directory.",
    )
    parser.add_argument(
        "--rust-stall-windows", type=int, default=3,
        help="Classify stall after this many zero-movement DAM progress windows.",
    )
    args = parser.parse_args()

    config = load_config()
    jobs = _build_job_list(config, args)
    print(f"Running {len(jobs)} benchmark(s) with {args.jobs} worker(s)\n")

    if args.jobs > 1:
        results, skipped = _run_parallel(
            jobs, config, args.verbose, args.jobs,
            max_compute_bw=args.max_compute_bw, show_memory=args.show_memory,
            sim_timeout_seconds=args.sim_timeout,
            rust_sim_debug=args.rust_sim_debug,
            rust_sim_log_dir=args.rust_sim_log_dir,
            rust_stall_windows=args.rust_stall_windows,
        )
    else:
        results, skipped = _run_serial(
            jobs, config, args.verbose,
            max_compute_bw=args.max_compute_bw, show_memory=args.show_memory,
            sim_timeout_seconds=args.sim_timeout,
            rust_sim_debug=args.rust_sim_debug,
            rust_sim_log_dir=args.rust_sim_log_dir,
            rust_stall_windows=args.rust_stall_windows,
        )

    # Summary sorted by kernel name then preset
    results.sort(key=lambda r: (r[0], r[1]))

    print(f"\n{'='*80}")
    print(f"  {'Kernel':30s} {'Preset':15s} {'Predicted':>10s} {'Actual':>10s} {'Error%':>8s}")
    print(f"{'='*80}")
    for kernel, preset, predicted, actual, err in results:
        print(f"  {kernel:30s} {preset:15s} {predicted:>10d} {actual:>10d} {err:>7.1f}%")
    if skipped:
        print(f"  --- Skipped {len(skipped)} kernel(s) due to errors ---")
        for kernel, preset, reason in skipped:
            print(f"    {kernel}/{preset}: {reason}")
    print(f"{'='*80}")

    avg_err = sum(abs(e) for _, _, _, _, e in results) / len(results) if results else 0
    print(f"  Average error: {avg_err:.1f}% ({len(results)} kernels, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
