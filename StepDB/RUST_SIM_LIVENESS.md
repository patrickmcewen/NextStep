# Rust Simulator Liveness Debugging

Use this workflow when `validate_timing.py` or another StepDB timing run looks
hung, livelocked, or deadlocked in the Rust/DAM simulator.

## 1. Check The Existing Process

If a simulator subprocess is already running, first check whether it is still
CPU-active:

```bash
cd /workspace/NextStep/StepDB
conda activate testenv
python debug_rust_sim_liveness.py check-pid <pid> --seconds 5
```

Interpretation:

- `CPU_ACTIVE`: not a classical sleeping/blocking deadlock.
- `NOT_CPU_ACTIVE`: likely blocked, idle, or waiting externally; inspect stderr
  and parent process state.

`py-spy --pid` may fail in containers without `CAP_SYS_PTRACE`. In that case,
prefer launching a fresh repro under `py-spy record -- ...` or use the DAM
progress workflow below.

## 2. Inspect The Serialized Graph

After `validate_timing.py` writes its work directory, inspect the protobuf:

```bash
python debug_rust_sim_liveness.py inspect-pb \
  seed_kernels/<suite>/<kernel_work_dir>/graph.pb \
  --around-id <interesting_id>
```

This reports operator counts and duplicate consumption of point-to-point
multi-output branches. Any `duplicate_consumed_multi_output_branches > 0` is a
Rust topology bug even if `validate_functional.py` passes.

## 3. Run A Timed Liveness Repro

Run the existing `graph.pb` with the standard StepDB timing configs:

```bash
python debug_rust_sim_liveness.py run-pb \
  seed_kernels/<suite>/<kernel_work_dir> \
  --timeout 120 \
  --stall-windows 3 \
  --log /tmp/rust_sim_liveness.log
```

The strongest signal is DAM channel movement, not context completion count.
Large streaming graphs can keep most contexts open for a long time, so
`completed` may plateau while data still flows.

Interpretation:

- `PROGRESSING`: context completions may be flat, but successful channel traffic
  continued during the timeout.
- `LIKELY STALLED`: no successful send/receive movement occurred across the last
  `--stall-windows` DAM progress windows.
- `UNKNOWN`: the installed `step_perf` does not emit the required progress lines
  or lacks channel counters.
- `FAILED`: DAM reported at least one failed context.

## Instrumented step_perf Build

`run-pb` can classify liveness only when the installed Rust simulator emits
`[step-perf-progress]` and `[dam-progress]` lines with channel counters. After
editing either `step-perf` or the checked-out `dam` dependency, rebuild from the
`step-perf` crate:

```bash
cd /workspace/NextStep/step_tl/step-perf
conda activate testenv
cargo clean -p dam --release
PROTOC=/root/miniconda3/envs/testenv/lib/python3.12/site-packages/torch/bin/protoc \
  maturin develop --release
```

If the summary says `UNKNOWN: no [dam-progress] lines were observed`, the active
`step_perf` install is stale or not instrumented.

## Integrated Evaluate / Timing Flags

The same runner is available from the normal StepDB entry points.

For `validate_timing.py`:

```bash
python validate_timing.py generated_end_to_end mixtral_small_b64 \
  --rust-sim-debug \
  --sim-timeout 120 \
  --rust-sim-log-dir /tmp/rust_sim_logs
```

For `evaluate.py`:

```bash
python evaluate.py generated_end_to_end mixtral_small_b64 \
  --timing-only \
  --rust-sim-debug \
  --sim-timeout 120 \
  --rust-sim-log /tmp/rust_sim_eval.log
```

`--rust-sim-debug` streams Rust/DAM progress lines as the simulator runs and
prints a liveness summary. `--rust-sim-log-dir` creates one log per
`validate_timing.py` kernel/preset pair, which is safer for `-j` parallel runs.
`evaluate.py` writes the liveness classification and log path into
`result.json`.
