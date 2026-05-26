#!/usr/bin/env python3
"""CLI wrapper around the shared Rust/DAM simulator liveness workflow."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rust_sim_runner import (
    RustSimDebugConfig,
    check_pid_activity,
    inspect_graph_pb,
    run_serialized_graph,
)


DEFAULT_HBM_CONFIG = {
    "addr_offset": 64,
    "channel_num": 32,
    "per_channel_latency": 2,
    "per_channel_init_interval": 2,
    "per_channel_outstanding": 1,
    "per_channel_start_up_time": 0,
}
DEFAULT_SIM_CONFIG = {
    "channel_depth": 1024,
    "functional_sim": False,
    "mock_bf16": False,
}


def check_pid(args: argparse.Namespace) -> None:
    active, cpu_seconds = check_pid_activity(int(args.pid), args.seconds)
    print(
        f"pid={args.pid} sample_seconds={args.seconds:.1f} "
        f"cpu_seconds_delta={cpu_seconds:.2f}"
    )
    print("classification=CPU_ACTIVE" if active else "classification=NOT_CPU_ACTIVE")
    subprocess.run(
        ["ps", "-p", str(args.pid), "-o", "pid,ppid,etime,pcpu,pmem,stat,args"],
        check=False,
    )


def run_pb(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    graph_pb = Path(args.graph_pb).resolve() if args.graph_pb else work_dir / "graph.pb"
    result = run_serialized_graph(
        work_dir=work_dir,
        graph_pb=graph_pb,
        hbm_config=DEFAULT_HBM_CONFIG,
        sim_config=DEFAULT_SIM_CONFIG,
        timeout_seconds=args.timeout,
        debug=RustSimDebugConfig(
            enabled=True,
            log_path=args.log,
            stall_windows=args.stall_windows,
        ),
    )
    print("\n=== rust-sim liveness summary ===")
    print(result.summary())
    deltas = [p.movement_delta for p in result.progress if p.movement_delta is not None]
    if deltas:
        print(f"movement_deltas_tail={deltas[-min(len(deltas), 8):]}")


def inspect_pb(args: argparse.Namespace) -> None:
    info = inspect_graph_pb(args.graph_pb)
    print(f"graph={info['graph']}")
    print(f"operators={info['operators']} max_op_id={info['max_op_id']}")
    counts = info["op_counts"]
    print("top_op_counts=" + ", ".join(f"{name}:{count}" for name, count in counts.most_common(12)))

    duplicates = info["duplicate_branches"]
    print(f"duplicate_consumed_multi_output_branches={len(duplicates)}")
    ops_by_id = info["ops_by_id"]
    for (source_id, stream_idx), users in sorted(duplicates.items())[: args.max_duplicates]:
        source = ops_by_id[source_id]
        print(f"  source={source.name}#{source_id} stream_idx={stream_idx} consumers={users}")

    if args.around_id is not None:
        low = args.around_id - args.radius
        high = args.around_id + args.radius
        op_types = info["op_types"]
        print(f"ops_around_id=[{low}, {high}]")
        for op_id in range(low, high + 1):
            if op_id in ops_by_id:
                op = ops_by_id[op_id]
                print(f"  {op.id}: {op.name} ({op_types[op.id]})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pid = sub.add_parser("check-pid", help="sample /proc CPU ticks for an existing simulator PID")
    pid.add_argument("pid")
    pid.add_argument("--seconds", type=float, default=5.0)
    pid.set_defaults(func=check_pid)

    run = sub.add_parser("run-pb", help="run an existing graph.pb and classify DAM channel progress")
    run.add_argument("work_dir")
    run.add_argument("--graph-pb")
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--stall-windows", type=int, default=3)
    run.add_argument("--log")
    run.set_defaults(func=run_pb)

    inspect = sub.add_parser("inspect-pb", help="inspect graph.pb topology and operator IDs")
    inspect.add_argument("graph_pb")
    inspect.add_argument("--around-id", type=int)
    inspect.add_argument("--radius", type=int, default=12)
    inspect.add_argument("--max-duplicates", type=int, default=20)
    inspect.set_defaults(func=inspect_pb)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
