"""Build the failing prefill_transformer graph and tap an intermediate node
(by name) into an OffChipStore. Compare functional vs Rust sim at that tap.
"""
import os, sys, subprocess, json, inspect
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "/workspace/NextStep/step_tl/src")
sys.path.append("/workspace/NextStep/step_tl/src/proto")

sys.path.insert(0, "/workspace/NextStep/StepDB")

import numpy as np
import torch

from validate_functional import IMPORT_SCAFFOLD
from loader import get_dims
from precompute import precompute_tensors

KERNEL = "generated_prefill_transformer"
PRESET = "small"
TAP_MODE = sys.argv[1] if len(sys.argv) > 1 else "class"
# Modes:
#   class <ClassName> <nth>     tap by class name + index
#   id    <instance_id>         tap node with that instance_id
#   list                         just list all nodes
TAP_CLASS = sys.argv[2] if len(sys.argv) > 2 else "Streamify"
TAP_NTH = int(sys.argv[3]) if len(sys.argv) > 3 else 0

dims = get_dims(KERNEL, PRESET)
tensors = precompute_tensors(KERNEL, dims)

impl_path = f"/workspace/NextStep/StepDB/seed_kernels/transformer_layer/{KERNEL}/step_impl.py"
with open(impl_path) as f:
    impl_code = f.read()

full_code = IMPORT_SCAFFOLD + "\n" + impl_code

ns = {}
exec(full_code, ns)

from step_py.ops import StepOps, OffChipStore, PromoteOuter, Flatten, Bufferize, Streamify, Promote
from step_py import ops as ops_mod

StepOps._counter = 0
graph, _orig_out = ns["build_graph"](dims, tensors)

if TAP_MODE == "list":
    for node in graph.nodes:
        try:
            sh = (tuple(node.stream.shape), tuple(node.stream.stream_dtype.shape))
        except Exception:
            sh = None
        fn = getattr(node, 'fn', None)
        print(f"  {node}  cls={type(node).__name__}  shape={sh}  fn={type(fn).__name__ if fn else None}")
    sys.exit(0)

if TAP_MODE == "class":
    candidates = [n for n in graph.nodes if type(n).__name__ == TAP_CLASS]
    print(f"Found {len(candidates)} {TAP_CLASS} nodes:")
    for i, c in enumerate(candidates):
        sh = None
        try:
            sh = (tuple(c.stream.shape), tuple(c.stream.stream_dtype.shape))
        except Exception:
            pass
        print(f"  [{i}] {c}: shape={sh}")
    assert TAP_NTH < len(candidates), f"TAP_NTH {TAP_NTH} out of range"
    tap = candidates[TAP_NTH]
elif TAP_MODE == "id":
    target_id = int(TAP_CLASS)
    tap = None
    for n in graph.nodes:
        if getattr(n, 'instance_id', None) == target_id:
            tap = n; break
    assert tap is not None, f"No node with instance_id={target_id}"
    print(f"Found node {tap} ({type(tap).__name__})")
else:
    sys.exit(f"Unknown TAP_MODE: {TAP_MODE}")

# Build a fresh graph containing only the path up to tap. Easiest: just
# add an OffChipStore on the tap directly in the existing graph; we'll
# control what gets serialized by changing the output_op argument to the
# OffChipStore we just created. We need an OffChipStore-compatible stream
# (stream rank >= 1), so Flatten if needed.
in_shape = tap.stream.shape
in_tile = tap.stream.stream_dtype.shape
print(f"\nTap stream shape: {tuple(in_shape)}  tile: {tuple(in_tile)}")

# OffChipStore drops the outermost stream dim from tensor_shape_tiled
# (ops.py:1899 — `tensor_shape_tiled = in_stream.shape[1:]`). The kernel's own
# final store always does `out = PromoteOuter(out); OffChipStore(out, ...)`.
# We have to mirror that or the Rust accumulator asserts at finalize.
tap_for_store = PromoteOuter(graph, tap)
while len(tap_for_store.stream.shape) < 2:
    tap_for_store = PromoteOuter(graph, tap_for_store)
# Flatten any trailing >1d stream dims to 1d for OffChipStore convenience.
# Actually OffChipStore just needs the input to have at least 1 stream dim.
# Don't reshape: we want raw stream output.

new_store = OffChipStore(graph, tap_for_store, par_dispatch=4096, store_file_name="tap_output")
print(f"New tap store: {new_store} -> {new_store.store_file_name}.npy; total nodes (no trim): {len(graph.nodes)}")

from rewrite.broadcast import infer_broadcast
# Diagnostic: show tap node neighbors BEFORE re-broadcasting.
print(f"\nBefore infer_broadcast: tap node {tap} successors:")
for nbr in graph.successors(tap):
    print(f"  -> {nbr}")
graph = infer_broadcast(graph)
print(f"After infer_broadcast: tap node {tap} successors:")
for nbr in graph.successors(tap):
    print(f"  -> {nbr}")
print(f"After infer_broadcast: tap_for_store {tap_for_store} successors:")
for nbr in graph.successors(tap_for_store):
    print(f"  -> {nbr}")

# === Functional sim ===
from timing_and_emulator.functional import execute
print("\n=== functional sim ===")
func_out = execute(graph, new_store)
print(f"Functional sim out shape: {func_out.shape}")
func_arr = func_out.numpy().reshape(-1)
print(f"Functional sim first 8: {func_arr[:8]}")
print(f"Functional sim |max|: {np.abs(func_arr).max()}")

# === Rust sim ===
WORK = Path(f"/tmp/tap_work_{TAP_CLASS}_{TAP_NTH}")
WORK.mkdir(exist_ok=True)
from sim import serialize, SimConfig, HBMConfig
CHANNEL_DEPTH = int(os.environ.get("CHANNEL_DEPTH", "2"))
sim_config = SimConfig(channel_depth=CHANNEL_DEPTH, functional_sim=True, mock_bf16=False)
hbm_config = HBMConfig(
    addr_offset=64, channel_num=32, per_channel_latency=2, per_channel_init_interval=2,
    per_channel_outstanding=1, per_channel_start_up_time=14,
)
os.chdir(WORK)
pb_path = str(WORK / "graph.pb")
serialize(graph, pb_path, sim_config.functional_sim)

runner = (
    "import json, sys, os\n"
    "os.chdir(sys.argv[1])\n"
    "from sim import HBMConfig, SimConfig\n"
    "import step_perf\n"
    "ret = step_perf.run_graph(sys.argv[2], False, HBMConfig(**json.loads(sys.argv[3])), SimConfig(**json.loads(sys.argv[4])), None)\n"
    "if len(ret) == 4: _, c, _, _ = ret\n"
    "else: _, c = ret\n"
    "print(json.dumps({'cycles': c}))\n"
)
env = os.environ.copy()
env["PYTHONPATH"] = "/workspace/NextStep/step_tl/src:/workspace/NextStep/step_tl/src/proto:" + env.get("PYTHONPATH", "")
proc = subprocess.run(
    [sys.executable, "-c", runner, str(WORK), pb_path,
     json.dumps(asdict(hbm_config)),
     json.dumps({"channel_depth": CHANNEL_DEPTH, "functional_sim": True, "mock_bf16": False})],
    capture_output=True, text=True, env=env, timeout=300,
)
print(f"\n=== Rust sim ===")
print(f"rc={proc.returncode}, stdout last={proc.stdout.strip().split(chr(10))[-1]}")
if proc.returncode != 0:
    print("STDERR:", proc.stderr[-3000:])
    sys.exit(2)
# Always print last bit of stderr so we can see Rust panic messages even when rc==0.
if proc.stderr.strip():
    print("STDERR tail:", proc.stderr.strip()[-2000:])
rust_arr = np.load(WORK / f"{new_store.store_file_name}.npy").reshape(-1)
print(f"Rust sim out shape: {rust_arr.shape}")
print(f"Rust sim first 8: {rust_arr[:8]}")
print(f"Rust sim |max|: {np.abs(rust_arr).max()}")

# Reconcile shapes
print()
print("=" * 70)
if func_arr.size != rust_arr.size:
    print(f"NUMEL MISMATCH: func={func_arr.size} rust={rust_arr.size}")
    print("This is itself a Rust/functional divergence (e.g. extra/missing emissions).")
else:
    diff = np.abs(func_arr - rust_arr)
    print(f"numel ok; max_diff={diff.max():.6e} mean_diff={diff.mean():.6e}")
    print(f"as ratio of |func|max: {diff.max() / (np.abs(func_arr).max() + 1e-12):.6e}")
    bad = np.where(diff > 1e-3)[0]
    if len(bad):
        for b in bad[:6]:
            print(f"  idx {b}: func={func_arr[b]:.4f} rust={rust_arr[b]:.4f}")
