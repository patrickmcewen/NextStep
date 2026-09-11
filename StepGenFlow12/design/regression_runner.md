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
translator, bundle-dir), and the gate-ordering and decomposition-planner
knobs (`--check-order`, `--no-plan`, `--max-replans`, `--node-attempts`,
`--max-plan-depth`, `--non-root-sequential` / `--no-non-root-sequential`).
The runner does not mediate these; it just forwards them.

Phase-0 decomposition is therefore a suite-level toggle: disabling the
planner on `run_regression.py` flips it for every job's pipeline, with
no regression-specific planner knobs of the runner's own. Performance
autotuning is **not invoked** by the regression runner — autotune2
runs as a separate post-pipeline pass via `run_autotune2.py` against
a chosen outer; see [autotuner.md](autotuner.md).

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
`translator`, `check_order`, etc.), the resolved `bench_config` path,
the `run_py` path, the `subset_file` (when relevant), and the full
enumerated `jobs` list. The intent is that `config.json` alone is
enough to reproduce the run.

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
  total_tokens:  int        # sum over jobs of LLM token usage
```

The outer flow scores a suite by reading `pass_rate = overall.fraction`
and `total_tokens = total_tokens` from this file. Per-kernel and
per-preset breakdowns are diagnostic surface for humans.

Per-job `outer_passed` / `outer_total` are recovered by reading each
job's `result.json` (which records the pass/fail of each parallel outer
attempt). When a job's result file is missing — typical on a hard crash
— the runner falls back to `(0, 0)` rather than crashing the summary.

Performance metrics are intentionally absent from the suite summary:
autotuning is a separate post-pipeline pass (`run_autotune2.py`) that
operates on a chosen outer's snapshot and writes its own
`autotune2_summary.json`. The regression runner only scores
*functional* outcomes.

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
