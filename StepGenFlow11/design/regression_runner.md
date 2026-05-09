# Regression runner

The regression runner is the multi-kernel batch driver. It enumerates a
set of `(kernel, preset)` jobs from configuration, fans them out as
subprocess invocations of the per-kernel CLI, caps parallelism, and
aggregates per-job results into a single JSON summary. It is the layer
the AbstractionOpt outer flow invokes to score a bundle across a
regression suite.

## Job planning

The set of jobs comes from one of three mutually-exclusive sources:

| selector | source |
|---|---|
| `--all-presets` | every preset of every kernel in `StepDB/bench_config.yaml` |
| `--preset-config <yaml>` | a YAML mapping `kernel: preset` (or `kernel: [presets]`) |
| `--subset <name>` | a named group from a multi-group YAML (default `regression_subsets.yaml`) |

The named-subset file is structured as `{group_name: {kernel: preset(s)}}`
and lets a single file carry several curated suites (smoke, simple,
transformer subparts, etc.). A subset's group must exist and be
non-empty; missing kernels or unknown presets are warned-and-skipped
rather than fatal, on the principle that suites evolve faster than the
bench config.

The planner emits a sorted list of `Job(kernel, preset)` records;
sorting makes job order deterministic across runs and tests.

## Subprocess model

Each job runs the per-kernel CLI as an independent subprocess, with its
own checkpoint subdirectory and its own log file. Why subprocesses
rather than in-process tasks?

- **Crash isolation.** A segfault, OOM kill, or unhandled exception in
  one kernel's pipeline cannot take down the rest of the suite.
- **CPU and memory accounting.** The host can limit per-job resources
  via the kernel; in-process tasks would all share the parent's limits.
- **Independence from the implementer's process state.** Per-kernel
  imports (the bundle's `step_dsl`, the bundle's `transpiler`, gold
  caches) cannot interfere across jobs, so module-eviction discipline
  in bundle mode is unnecessary at the suite level.

Pass-through arguments to each subprocess: the LLM profile (model /
config), the per-job loop knobs (max-outer, max-turns, pipeline,
translator, bundle-dir), the gate-ordering and decomposition-planner
knobs (`--check-order`, `--no-plan`, `--max-replans`, `--node-attempts`,
`--max-plan-depth`, `--non-root-sequential` / `--no-non-root-sequential`),
and the autotune knobs (`--autotune`, `--autotune-config`,
`--autotune-max-turns`, `--autotune-agent`). The runner does not
mediate these; it just forwards them.

Per-outer autotune integration and Phase-0 decomposition are therefore
suite-level toggles: enabling `--autotune` (or disabling the planner)
on `run_regression.py` flips it for every job's pipeline, with no
regression-specific autotune or planner knobs of the runner's own.

The runner does **not** raise on a non-zero exit. A failing kernel is a
real outcome the caller (the outer flow's scoring step, or a human)
must observe. Failures are recorded with their exit code and duration.

## Parallelism

A single asyncio semaphore caps the number of in-flight subprocesses at
`--max-parallel`. There is no other queueing — jobs are submitted in
sorted order and run as soon as a slot opens. The runner logs `START` /
`PASS` / `FAIL` lines as jobs progress so the live tail of
`regression.log` shows progress.

Parallelism is at the suite level. Within a single job, the per-kernel
flow already runs `max_outer` independent attempts in parallel via
`asyncio.gather`; those attempts share a process and a precomputed-gold
cache.

## Run directory

Every regression invocation owns a timestamped output directory under
`--results-root` (default `/workspace/regression_results`):

```
<results-root>/<YYYYmmdd-HHMMSS>/
├── config.json          # the full invocation snapshot — see below
├── regression.log       # lifecycle log — START / PASS / FAIL per job + summary
├── jobs/
│   └── <kernel>__<preset>.log   # per-job stdout+stderr captured from the subprocess
├── checkpoints/
│   └── <kernel>__<preset>/      # per-job checkpoint root, passed to the subprocess via --checkpoint-dir
└── summary.json         # aggregated outcome — see below
```

The per-job checkpoint directories under `checkpoints/` are the same
trees the per-kernel CLI would write on its own; the regression runner
just chooses their location and gives each job a stable, predictable
path.

`config.json` records: `argv`, `preset_mode`, `max_parallel`, `model` /
`config`, every pass-through knob (`max_outer`, `max_turns`, `pipeline`,
`translator`, `check_order`, the autotune flags, etc.), the resolved
`bench_config` path, the `run_py` path, the `subset_file` (when
relevant), and the full enumerated `jobs` list. The intent is that
`config.json` alone is enough to reproduce the run.

## Summary structure

`summary.json` aggregates per-kernel and overall outcomes:

```
SuiteSummary:
  started_at:    iso8601
  finished_at:   iso8601
  wall_seconds:  float
  max_parallel:  int
  model:         string
  preset_mode:   "all_presets" | "preset_config" | "subset:<name>"
  overall:
    passed:    int
    total:     int
    fraction:  float
  outer_overall:
    passed:    int          # sum over jobs of per-outer passed counts
    total:     int          # sum over jobs of per-outer totals
  autotune_overall:                # always present; fields null when no data
    jobs_with_data:   int          # jobs whose headline autotune surfaced a speedup
    min_speedup:      float | null
    max_speedup:      float | null
    geomean_speedup:  float | null # geomean across those jobs
  benchmarks:
    <kernel>:
      passed:    int
      total:     int
      fraction:  float
      presets:
        <preset>:
          status:        "pass" | "fail"
          duration_s:    float
          exit_code:     int
          outer_passed:  int
          outer_total:   int
          autotune:                # null when no outer's autotune chain produced a usable best
            best_outer:       int       # outer with the lowest overall.best_cycles
            baseline_cycles:  int
            best_cycles:      int
            speedup:          float
            per_outer:
              - outer:            int
                status:           "ok" | "halted" | "error" | "missing"
                baseline_cycles:  int | null
                best_cycles:      int | null
                speedup:          float | null
  total_tokens:  int        # sum over jobs of LLM token usage
```

The outer flow scores a suite by reading `pass_rate = overall.fraction`
and `total_tokens = total_tokens` from this file. Per-kernel and
per-preset breakdowns are diagnostic surface for humans.

Per-job `outer_passed` / `outer_total` are recovered by reading each
job's `result.json` (which records the pass/fail of each parallel outer
attempt). When a job's result file is missing — typical on a hard crash
— the runner falls back to `(0, 0)` rather than crashing the summary.

Per-job `autotune` is the chain-aware headline: the runner reads each
outer's `autotune.overall.best_cycles` from `result.json` (when the
chain ran cleanly or was halted on infeasibility), picks the lowest,
and surfaces that outer as `best_outer` along with its overall
baseline / best / speedup. The per-outer breakdown distinguishes four
states:

- `ok` — the outer's autotune chain ran cleanly to completion;
- `halted` — the chain stopped on a pass that ended `feasible=False`;
- `error` — a pass crashed; partial best is recovered from the pass's
  `progress.json` so the outer still contributes a `best_cycles` if
  any work landed before the crash;
- `missing` — the outer's functional pipeline never reached the
  autotune hook (typically because the outer failed in Phase 1 or 2).

When no outer produced a usable best, the headline `autotune` block is
null and the job contributes nothing to `autotune_overall`. The "best
across outers" framing is intentional: the chosen outer for the
*functional* result is selected by correctness (first success), and
the performance number you want is the best the run achieved, not
necessarily the chosen one's.

`autotune_overall` is always present in the summary, with all-null
fields when no jobs produced any speedup data — this keeps consumers'
schema parsing uniform across `--autotune` and non-`--autotune` runs.

## Logging

The runner sets up a dedicated logger that tees to stdout and to
`regression.log`. The two source modules (`regression_planning`,
`regression_runner`) re-route into the same file so all suite-level
output is in one place. Per-job stdout/stderr stay in
`jobs/<kernel>__<preset>.log` rather than mixing into the suite log.

## Inputs the runner depends on

- `StepDB/bench_config.yaml` — kernel inventory and preset definitions.
- `regression_subsets.yaml` (or any file passed via `--subset-file`) —
  named curated suites.
- A model profile under `configs/<name>.json`, or an explicit
  `--config` JSON path.
- Optionally, a bundle directory; this is forwarded to every per-job
  subprocess unchanged.
