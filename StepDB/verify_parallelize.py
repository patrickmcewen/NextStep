"""Verify Python emulator vs Rust simulator semantics for Parallelize.

For the parallelize_observe benchmark (which adds a per-consumer marker
between Parallelize and StaticReassemble), the output reveals which
consumer saw each input row. We compare:
  (1) the PyTorch reference (assumes round-robin per rank-1 unit)
  (2) the Python functional emulator (DEIOpt: round-robin slicing)
  (3) the Rust step-perf simulator output (.npy)
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO_ROOT = "/workspace/DEIOpt"
STEPDB_DIR = os.path.join(REPO_ROOT, "StepDB")
STEP_TL_SRC = os.path.join(REPO_ROOT, "step_tl/src")
STEP_TL_PROTO = os.path.join(REPO_ROOT, "step_tl/src/proto")

sys.path.insert(0, STEPDB_DIR)
sys.path.insert(0, STEP_TL_SRC)
sys.path.insert(0, STEP_TL_PROTO)


def load_reference(kernel_name):
    path = os.path.join(
        STEPDB_DIR, "seed_kernels/language_primitives", kernel_name, "reference.py"
    )
    spec = importlib.util.spec_from_file_location("ref", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    return ref


def run_rust_functional(graph, output_op, work_dir):
    from sim import SimConfig, HBMConfig, serialize

    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    pb_path = os.path.join(work_dir, "graph.pb")

    sim_config = SimConfig(channel_depth=2, functional_sim=True, mock_bf16=False)
    hbm_config = HBMConfig(
        addr_offset=64, channel_num=32,
        per_channel_latency=2, per_channel_init_interval=2,
        per_channel_outstanding=1, per_channel_start_up_time=0,
    )

    orig_dir = os.getcwd()
    os.chdir(work_dir)
    try:
        serialize(graph, pb_path, sim_config.functional_sim)
    finally:
        os.chdir(orig_dir)

    runner = (
        "import json, sys, os\n"
        "os.chdir(sys.argv[1])\n"
        "from sim import HBMConfig, SimConfig\n"
        "import step_perf\n"
        "hbm = HBMConfig(**json.loads(sys.argv[3]))\n"
        "sim = SimConfig(**json.loads(sys.argv[4]))\n"
        "ret = step_perf.run_graph(sys.argv[2], False, hbm, sim, None)\n"
        "if len(ret) == 4:\n"
        "    _, cycles, *_ = ret\n"
        "elif len(ret) == 2:\n"
        "    _, cycles = ret\n"
        "else:\n"
        "    raise RuntimeError(f'unexpected return: {ret}')\n"
        "print(json.dumps({'cycles': cycles}))\n"
    )

    from dataclasses import asdict
    env = os.environ.copy()
    env["PYTHONPATH"] = STEP_TL_SRC + ":" + STEP_TL_PROTO + ":" + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", runner, work_dir, pb_path,
         json.dumps(asdict(hbm_config)),
         json.dumps({"channel_depth": sim_config.channel_depth,
                     "functional_sim": sim_config.functional_sim,
                     "mock_bf16": sim_config.mock_bf16})],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert proc.returncode == 0, f"sim failed:\n{proc.stderr}"

    candidates = [
        output_op.store_file_name + ".npy",
        output_op.store_file_name,
        f"OffChipStore_{output_op.instance_id}.npy",
        f"{output_op.store_file_name}_0.npy",
    ]
    npy_path = None
    for c in candidates:
        p = os.path.join(work_dir, c)
        if os.path.exists(p):
            npy_path = p
            break
    if npy_path is None:
        files = sorted(os.listdir(work_dir))
        raise FileNotFoundError(
            f"No output .npy in {work_dir}. Files present: {files}\n"
            f"sim stdout: {proc.stdout}\nsim stderr (last 1000 chars): {proc.stderr[-1000:]}"
        )
    return np.load(npy_path), proc.stdout, proc.stderr


def main():
    from validate_timing import build_graph_from_impl, load_config
    from timing_and_emulator.functional import execute
    from step_py.ops import StepOps

    cfg = load_config()
    kernel = "parallelize_observe"
    ref = load_reference(kernel)

    print(f"Verifying {kernel}\n")
    for preset, dims in cfg[kernel]["presets"].items():
        print(f"=== preset={preset}  dims={dims} ===")

        gold = ref.compute_gold(dims)

        StepOps._counter = 0
        graph_emu, output_emu = build_graph_from_impl(kernel, dims, cfg)
        emu_out = execute(graph_emu, output_emu)

        StepOps._counter = 0
        graph_rust, output_rust = build_graph_from_impl(kernel, dims, cfg)
        work_dir = tempfile.mkdtemp(prefix=f"par_obs_{preset}_")
        rust_arr, sim_stdout, sim_stderr = run_rust_functional(
            graph_rust, output_rust, work_dir
        )
        print(f"  rust work_dir: {work_dir}")
        print(f"  rust files: {sorted(os.listdir(work_dir))}")

        rust_t = torch.from_numpy(rust_arr).float()
        # Rust output may have leading singleton or extra axes — squeeze to 2-D.
        while rust_t.ndim > gold.ndim:
            rust_t = rust_t.squeeze(0)
        if rust_t.shape != gold.shape:
            rust_t = rust_t.reshape(gold.shape)

        emu_match = torch.allclose(emu_out.float(), gold.float(), atol=1e-4)
        rust_match = torch.allclose(rust_t, gold.float(), atol=1e-4)
        emu_rust_match = torch.allclose(emu_out.float(), rust_t, atol=1e-4)

        print(f"  reference (round-robin assumption) shape: {tuple(gold.shape)}")
        print(f"  Python emulator vs reference: {emu_match}")
        print(f"  Rust simulator   vs reference: {rust_match}")
        print(f"  Python emulator vs Rust:       {emu_rust_match}")

        if not (emu_match and rust_match):
            print("  --- first column comparison ---")
            print(f"  ref:  {gold[:, 0].tolist()[:16]}")
            print(f"  emu:  {emu_out[:, 0].tolist()[:16]}")
            print(f"  rust: {rust_t[:, 0].tolist()[:16]}")
        print()


if __name__ == "__main__":
    main()
