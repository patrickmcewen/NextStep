"""Shared Rust/DAM simulator runner and liveness diagnostics."""

from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


STEPDB_DIR = Path(__file__).resolve().parent
NEXTSTEP_DIR = STEPDB_DIR.parent
STEP_TL_SRC = NEXTSTEP_DIR / "step_tl" / "src"
STEP_TL_PROTO = STEP_TL_SRC / "proto"

DAM_PROGRESS_RE = re.compile(
    r"\[dam-progress\] total=(?P<total>\d+) spawned=(?P<spawned>\d+) "
    r"started=(?P<started>\d+) completed=(?P<completed>\d+) failed=(?P<failed>\d+)"
    r"(?: sends=(?P<sends>\d+) \(\+(?P<send_delta>\d+)\) "
    r"recvs=(?P<recvs>\d+) \(\+(?P<recv_delta>\d+)\) "
    r"try_recv_hits=(?P<try_recv_hits>\d+) \(\+(?P<try_recv_delta>\d+)\))? "
    r"pending_sample=\[(?P<pending>.*)\]"
)
BUILD_RE = re.compile(r"\[step-perf-progress\] (?P<message>.*)")


def _config_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if is_dataclass(config):
        return asdict(config)
    assert hasattr(config, "__dict__"), f"unsupported config type: {type(config).__name__}"
    return dict(config.__dict__)


@dataclass(frozen=True)
class RustSimDebugConfig:
    enabled: bool = False
    log_path: str | None = None
    stall_windows: int = 3


@dataclass(frozen=True)
class DamProgress:
    total: int
    spawned: int
    started: int
    completed: int
    failed: int
    sends: int | None
    send_delta: int | None
    recvs: int | None
    recv_delta: int | None
    try_recv_hits: int | None
    try_recv_delta: int | None
    pending: str

    @property
    def movement_delta(self) -> int | None:
        if self.send_delta is None:
            return None
        assert self.recv_delta is not None
        assert self.try_recv_delta is not None
        return self.send_delta + self.recv_delta + self.try_recv_delta


@dataclass(frozen=True)
class RustSimResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    sim_json: dict[str, Any] | None
    progress: tuple[DamProgress, ...]
    build_messages: tuple[str, ...]
    classification: str
    log_path: str | None

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    @property
    def latest_progress(self) -> DamProgress | None:
        return self.progress[-1] if self.progress else None

    def summary(self) -> str:
        parts = [
            f"timed_out={self.timed_out}",
            f"returncode={self.returncode}",
            f"classification={self.classification}",
        ]
        latest = self.latest_progress
        if latest is not None:
            parts.append(
                "latest="
                f"total={latest.total} spawned={latest.spawned} started={latest.started} "
                f"completed={latest.completed} failed={latest.failed} "
                f"movement_delta={latest.movement_delta} "
                f"pending_sample=[{latest.pending}]"
            )
        if self.log_path:
            parts.append(f"log={self.log_path}")
        return "\n".join(parts)


def pythonpath() -> str:
    entries = [str(STEPDB_DIR), str(STEP_TL_SRC), str(STEP_TL_PROTO)]
    current = os.environ.get("PYTHONPATH")
    if current:
        entries.append(current)
    return ":".join(entries)


def parse_progress(line: str) -> DamProgress | None:
    match = DAM_PROGRESS_RE.search(line)
    if match is None:
        return None

    def optional_int(name: str) -> int | None:
        value = match.group(name)
        return None if value is None else int(value)

    return DamProgress(
        total=int(match.group("total")),
        spawned=int(match.group("spawned")),
        started=int(match.group("started")),
        completed=int(match.group("completed")),
        failed=int(match.group("failed")),
        sends=optional_int("sends"),
        send_delta=optional_int("send_delta"),
        recvs=optional_int("recvs"),
        recv_delta=optional_int("recv_delta"),
        try_recv_hits=optional_int("try_recv_hits"),
        try_recv_delta=optional_int("try_recv_delta"),
        pending=match.group("pending"),
    )


def classify_progress(
    progress: list[DamProgress] | tuple[DamProgress, ...],
    timed_out: bool,
    stall_windows: int,
) -> str:
    if not progress:
        return (
            "UNKNOWN: no [dam-progress] lines were observed. Rebuild/install an "
            "instrumented step_perf before classifying DAM liveness."
        )
    latest = progress[-1]
    if latest.failed:
        return f"FAILED: DAM reported {latest.failed} failed context(s)."
    if not timed_out:
        return "FINISHED: simulator process exited before the timeout."
    if latest.movement_delta is None:
        return (
            "UNKNOWN: DAM context progress was observed, but this build does not "
            "print channel traffic counters."
        )
    window = progress[-stall_windows:]
    deltas = [p.movement_delta for p in window]
    assert all(d is not None for d in deltas)
    if len(deltas) >= stall_windows and all(d == 0 for d in deltas):
        return (
            f"LIKELY STALLED: no successful send/receive movement in the last "
            f"{stall_windows} DAM progress windows."
        )
    return (
        "PROGRESSING: context completions may be flat, but DAM channel traffic "
        "continued during the timeout window."
    )


def run_serialized_graph(
    work_dir: str | Path,
    graph_pb: str | Path,
    hbm_config: Any,
    sim_config: Any,
    timeout_seconds: float,
    debug: RustSimDebugConfig | None = None,
) -> RustSimResult:
    work_dir = Path(work_dir).resolve()
    graph_pb = Path(graph_pb).resolve()
    assert work_dir.is_dir(), f"work_dir does not exist: {work_dir}"
    assert graph_pb.is_file(), f"graph.pb does not exist: {graph_pb}"
    debug = debug or RustSimDebugConfig()
    assert debug.stall_windows >= 1, f"stall_windows must be >= 1, got {debug.stall_windows}"

    runner = (
        "import json, os, sys\n"
        "os.chdir(sys.argv[1])\n"
        "from sim import HBMConfig, SimConfig\n"
        "import step_perf\n"
        "hbm = HBMConfig(**json.loads(sys.argv[3]))\n"
        "sim = SimConfig(**json.loads(sys.argv[4]))\n"
        "ret = step_perf.run_graph(sys.argv[2], False, hbm, sim, None)\n"
        "if len(ret) == 4:\n"
        "    passed, cycles, dur_ms, dur_s = ret\n"
        "elif len(ret) == 2:\n"
        "    passed, cycles = ret\n"
        "    dur_ms, dur_s = 0.0, 0.0\n"
        "else:\n"
        "    raise RuntimeError(f'Unexpected return: {ret}')\n"
        "print(json.dumps({'passed': bool(passed), 'cycles': cycles, 'dur_ms': dur_ms, 'dur_s': dur_s}))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath()
    if debug.enabled:
        env["STEP_PERF_PROGRESS"] = "1"
        env["DAM_PROGRESS"] = "1"
    else:
        env.setdefault("STEP_PERF_PROGRESS", "0")
        env.setdefault("DAM_PROGRESS", "0")

    cmd = [
        sys.executable,
        "-c",
        runner,
        str(work_dir),
        str(graph_pb),
        json.dumps(_config_dict(hbm_config)),
        json.dumps(_config_dict(sim_config)),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(STEPDB_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    log = Path(debug.log_path).resolve().open("w") if debug.log_path else None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    lines: list[str] = []
    progress: list[DamProgress] = []
    build_messages: list[str] = []
    timed_out = False

    while proc.poll() is None and time.monotonic() < deadline:
        for key, _mask in selector.select(timeout=0.5):
            line = key.fileobj.readline()
            if not line:
                continue
            lines.append(line)
            if debug.enabled:
                print(line, end="")
            if log is not None:
                log.write(line)
                log.flush()
            parsed = parse_progress(line)
            if parsed is not None:
                progress.append(parsed)
            build_match = BUILD_RE.search(line)
            if build_match is not None:
                build_messages.append(build_match.group("message"))

    if proc.poll() is None:
        timed_out = True
        proc.kill()
        proc.wait()

    remainder = proc.stdout.read()
    if remainder:
        lines.append(remainder)
        if debug.enabled:
            print(remainder, end="")
        if log is not None:
            log.write(remainder)
    if log is not None:
        log.close()

    output = "".join(lines)
    sim_json = None
    for line in reversed(output.splitlines()):
        if line.startswith("{") and line.endswith("}"):
            sim_json = json.loads(line)
            break

    classification = classify_progress(progress, timed_out, debug.stall_windows)
    return RustSimResult(
        returncode=proc.returncode,
        stdout=output,
        stderr="",
        timed_out=timed_out,
        sim_json=sim_json,
        progress=tuple(progress),
        build_messages=tuple(build_messages),
        classification=classification,
        log_path=str(Path(debug.log_path).resolve()) if debug.log_path else None,
    )


def check_pid_activity(pid: int, seconds: float = 5.0) -> tuple[bool, float]:
    stat_path = Path(f"/proc/{pid}/stat")
    assert stat_path.exists(), f"PID {pid} does not exist"

    def cpu_ticks() -> int:
        after_comm = stat_path.read_text().rsplit(") ", 1)[1].split()
        return int(after_comm[11]) + int(after_comm[12])

    start = cpu_ticks()
    time.sleep(seconds)
    end = cpu_ticks()
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    cpu_seconds = (end - start) / hz
    return end > start, cpu_seconds


def _load_graph_pb(path: Path):
    sys.path.insert(0, str(STEP_TL_SRC))
    sys.path.insert(0, str(STEP_TL_PROTO))
    from proto import graph_pb2  # pylint: disable=import-outside-toplevel

    graph = graph_pb2.ProgramGraph()  # pylint: disable=no-member
    graph.ParseFromString(path.read_bytes())
    return graph


def _input_refs(op) -> list[tuple[int, int]]:
    op_type = op.WhichOneof("op_type")
    msg = getattr(op, op_type)
    present = {field.name for field, _value in msg.ListFields()}
    refs: list[tuple[int, int]] = []

    def add_pair(id_name: str, idx_name: str) -> None:
        fields = msg.DESCRIPTOR.fields_by_name
        if id_name not in fields:
            return
        if id_name not in present and getattr(msg, id_name) == 0:
            return
        stream_idx = getattr(msg, idx_name) if idx_name in fields else 0
        refs.append((int(getattr(msg, id_name)), int(stream_idx)))

    if "stream_idx" in msg.DESCRIPTOR.fields_by_name:
        add_pair("input_id", "stream_idx")
    else:
        add_pair("input_id", "input_stream_idx")
    add_pair("input_id1", "stream_idx1")
    add_pair("input_id2", "stream_idx2")
    add_pair("ref_id", "ref_stream_idx")

    fields = msg.DESCRIPTOR.fields_by_name
    if "input_id_list" in fields:
        ids = list(getattr(msg, "input_id_list"))
        idxs = list(getattr(msg, "input_stream_idx_list")) if "input_stream_idx_list" in fields else []
        if idxs:
            assert len(ids) == len(idxs), f"bad input list lengths in op {op.id}"
        refs.extend((int(source), int(idxs[i]) if idxs else 0) for i, source in enumerate(ids))

    return refs


def inspect_graph_pb(graph_pb: str | Path) -> dict[str, Any]:
    graph_pb = Path(graph_pb).resolve()
    assert graph_pb.is_file(), f"graph.pb does not exist: {graph_pb}"
    graph = _load_graph_pb(graph_pb)
    ops_by_id = {op.id: op for op in graph.operators}
    op_types = {op.id: op.WhichOneof("op_type") for op in graph.operators}
    counts = Counter(op_types.values())

    multi_output_types = {"broadcast", "parallelize", "flat_partition", "eager_merge"}
    consumers: dict[tuple[int, int], list[str]] = defaultdict(list)
    for op in graph.operators:
        for source_id, stream_idx in _input_refs(op):
            if op_types.get(source_id) in multi_output_types:
                consumers[(source_id, stream_idx)].append(f"{op.name}#{op.id}")

    duplicates = {
        branch: users
        for branch, users in consumers.items()
        if len(users) > 1
    }
    return {
        "graph": str(graph_pb),
        "operators": len(graph.operators),
        "max_op_id": max(ops_by_id),
        "op_counts": counts,
        "duplicate_branches": duplicates,
        "ops_by_id": ops_by_id,
        "op_types": op_types,
    }
