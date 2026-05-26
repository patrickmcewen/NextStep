"""Minimal repro for the Rust Flatten/Accum/RepeatRef stop-token bug.

The Python functional emulator treats this graph as dense tensor ops and
produces a valid result. The Rust simulator currently drops the stop token in
Flatten(0, 1), so Accum(rank=1) emits no scalar and the downstream BinaryMap
panics with "One stream closed earlier".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

STEPDB_DIR = Path(__file__).resolve().parent
STEP_TL_SRC = STEPDB_DIR.parent / "step_tl" / "src"
STEP_TL_PROTO = STEP_TL_SRC / "proto"
sys.path.insert(0, str(STEP_TL_SRC))
sys.path.insert(0, str(STEP_TL_PROTO))
sys.path.insert(0, str(STEPDB_DIR))

from graph.graph import MultiDiGraph as Graph
from rewrite.broadcast import infer_broadcast
from rust_sim_runner import RustSimDebugConfig, run_serialized_graph
from step_py.datatype import Float32, Tile
from step_py.functions.accum_fn import Max
from step_py.functions.init_fn import Zero
from step_py.functions.map_fn import Add, MulImmediate
from step_py.ops import (
    Accum,
    BinaryMap,
    Broadcast,
    Flatten,
    LinearOffChipLoad,
    OffChipStore,
    Parallelize,
    RepeatRef,
    ReshapePadStream,
    StepOps,
    UnaryMap,
)
from step_py.utility_ops import ConsumerContext
from timing_and_emulator.functional import execute


def build_graph(*, fixed: bool):
    StepOps._counter = 0
    graph = Graph()
    data = torch.tensor(
        [
            [0.0, 4.0],
            [10.0, 11.0],
            [4.0, 20.0],
            [30.0, 31.0],
        ]
    )

    load = LinearOffChipLoad(
        underlying=data,
        stride=(2, 1),
        out_shape_tiled=(4, 2),
        tile_row=1,
        tile_col=1,
        par_dispatch=4,
    )

    # Normal load shape is (1, 4, 2). The broken path further collapses the
    # stream to rank 0, which removes the stop-token boundary Accum(rank=1)
    # needs. The fixed path keeps the logical reduce dimension as stream rank 1.
    load_flat = Flatten(graph, load, min_rank=1, max_rank=2)
    flattened = load_flat if fixed else Flatten(graph, load_flat, min_rank=0, max_rank=1)
    par = Parallelize(
        graph,
        flattened,
        parallelize_rank=flattened.stream.rank,
        num_consumers=4,
        switch_cycles=[1, 1, 1, 1],
        write_back_mu=False,
    )

    # Keep the unused Parallelize outputs connected. Without these consumers,
    # DAM fails graph initialization with disconnected channels before reaching
    # the bug under test.
    for idx in range(1, 4):
        ConsumerContext(graph, (par, idx))

    scores = (par, 0)
    scores_bc = Broadcast(graph, scores, 3)
    max_score = Accum(
        graph,
        (scores_bc, 0),
        Tile(Float32(), (1, 1)),
        Max(),
        Zero((1, 1), Float32()),
        accum_rank=1,
        write_back_mu=False,
        compute_bw=1024,
    )
    max_repeated = RepeatRef(graph, max_score, (scores_bc, 1))
    neg_max = UnaryMap(
        graph,
        max_repeated,
        MulImmediate(-1.0),
        write_back_mu=False,
        compute_bw=1024,
    )
    shifted = BinaryMap(
        graph,
        (scores_bc, 2),
        neg_max,
        Add(),
        write_back_mu=False,
        compute_bw=1024,
    )
    if fixed:
        output_stream = shifted
    else:
        output_stream = ReshapePadStream(
            graph,
            shifted,
            chunk_size=1,
            reshape_rank=0,
            write_back_mu=False,
            have_pad_stream=False,
        )
    output = OffChipStore(graph, output_stream, par_dispatch=4, store_file_name="output")
    return infer_broadcast(graph), output


def run_functional(graph, output):
    result = execute(graph, output)
    expected = torch.tensor([-4.0, 0.0])
    assert torch.equal(result.reshape(-1), expected), (
        f"functional result {result} != flattened {expected}"
    )
    return result


def serialize_graph(graph, work_dir: Path) -> Path:
    from sim import serialize

    work_dir.mkdir(parents=True, exist_ok=True)
    graph_pb = work_dir / "graph.pb"
    orig_dir = os.getcwd()
    os.chdir(work_dir)
    serialize(graph, str(graph_pb), False)
    os.chdir(orig_dir)
    return graph_pb


def run_rust(graph_pb: Path, work_dir: Path, timeout: float):
    from sim import HBMConfig, SimConfig

    return run_serialized_graph(
        work_dir=work_dir,
        graph_pb=graph_pb,
        hbm_config=HBMConfig(
            addr_offset=64,
            channel_num=32,
            per_channel_latency=2,
            per_channel_init_interval=2,
            per_channel_outstanding=1,
            per_channel_start_up_time=0,
        ),
        sim_config=SimConfig(channel_depth=1024, functional_sim=False, mock_bf16=False),
        timeout_seconds=timeout,
        debug=RustSimDebugConfig(
            enabled=True,
            log_path=str(work_dir / "rust.log"),
            stall_windows=2,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        default="/workspace/rust_debug/repro_flatten_repeatref_stop_bug",
    )
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Preserve the rank-1 stop-token boundary before Accum.",
    )
    args = parser.parse_args()

    graph, output = build_graph(fixed=args.fixed)
    functional = run_functional(graph, output)
    graph_pb = serialize_graph(graph, Path(args.work_dir))
    rust = run_rust(graph_pb, Path(args.work_dir), args.timeout)

    assert not rust.timed_out, rust.summary()
    assert rust.sim_json is not None, rust.output[-2000:]
    if args.fixed:
        assert rust.sim_json["passed"] is True, rust.output[-2000:]
    else:
        assert rust.sim_json["passed"] is False, rust.output[-2000:]
        assert "src/operator/map.rs:115" in rust.output, rust.output[-2000:]
        assert "One stream closed earlier" in rust.output, rust.output[-2000:]

    print(f"Functional output: {functional.tolist()}")
    print("Rust fixed graph passed:" if args.fixed else "Rust repro matched expected panic:")
    print(rust.summary())


if __name__ == "__main__":
    main()
