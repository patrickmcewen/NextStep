# Logging

A StepGenFlow run produces a layered, on-disk record sufficient to
reconstruct what every LLM saw and what every gate decided. There are
three logical scopes: the per-kernel run (one invocation of the
implementer pipeline), the regression suite (one invocation of the
batch driver wrapping many per-kernel runs), and the autotune run.
Each owns its own tree.

## Per-kernel checkpoint tree

A per-kernel invocation owns a directory whose default name is a UTC
timestamp and which can be overridden via `--checkpoint-dir`. The
directory is the durable record of the run.

```
<checkpoint-dir>/
├── config.json                     # frozen invocation arguments
└── <kernel>/                       # one subtree per kernel; in single-kernel runs there's exactly one
    ├── result.json                 # outcome of the run; the file the regression runner parses
    └── outer_<i>/                  # one per parallel outer attempt
        ├── log.txt                 # this attempt's lifecycle log (per-pass progress, gate verdicts)
        ├── dsl_code.py             # phase-1 verified DSL output (when refactor_final succeeded)
        ├── autotune/               # only present when --autotune was set and this outer succeeded
        │   └── <kernel>/           # the autotune subsystem's own checkpoint tree
        │       ├── config.json
        │       ├── baseline.py / baseline_timing.txt
        │       ├── progress.json   # crash-safe best-so-far (see autotuner.md)
        │       ├── turn_<n>/...
        │       ├── best.py / best_timing.txt
        │       └── result.json     # only present on a clean autotune completion
        └── <pass>/                 # one per LLM pass that ran
            ├── system_prompt.txt   # the rendered system prompt this pass actually used
            └── turn_<m>/
                ├── user_prompt.txt        # the next-turn input the model saw
                ├── response.txt           # assistant text content
                ├── reasoning.txt          # provider-supplied reasoning (when present)
                ├── extracted_code.py      # the parsed code block
                ├── correctness_result.txt # gate 1 output
                ├── shape_trace.txt        # captured DSL op trace (when correctness raised)
                ├── status.txt             # one-line summary of how this turn ended
                ├── judge_response.txt     # gate 3 output (when judge ran)
                ├── judge_reasoning.txt    # judge reasoning (when present)
                └── translate_check/       # post-validator artifacts (refactor_final + auto)
                    ├── step_extracted_code.py     # what the deterministic translator emitted
                    ├── error.txt                  # translator exception, when one was raised
                    ├── graph_error.txt            # simulator exception, when graph failed to execute
                    └── graph_correctness.txt      # gold comparison of the lowered graph
```

The `autotune/<kernel>/` nesting under each outer is the per-outer
autotune subsystem's own checkpoint tree (see
[autotuner.md](autotuner.md)). It only appears when `--autotune` was
set on the invocation *and* that particular outer's functional
pipeline reached the verified-graph step.

`status.txt` is the most useful single file for triage — its values
are a fixed vocabulary:

| value | meaning |
|---|---|
| `PASS` | every gate passed; pass loop exited successfully |
| `NO_CODE_EXTRACTED` | response had no parseable code block |
| `FAIL: <error head>` | correctness check rejected the candidate |
| `CORRECT_BUT_NONCOMPLIANT` | correctness PASS, regex compliance found violations |
| `CORRECT_BUT_JUDGE_REJECTED` | correctness + regex PASS, judge said REJECT |
| `CORRECT_BUT_POST_VALIDATOR_REJECTED` | correctness + regex + judge PASS, post-validator rejected |

The shape trace, judge artifacts, and `translate_check/` subtree only
appear on turns where they ran — their absence is informative.

## `result.json`

The per-kernel result file is the canonical record of how the run ended.
Its schema (relevant fields):

```
PerKernelResult:
  success:           bool                  # did any outer attempt succeed
  outer_iteration:   int                   # which attempt is described in the rest (the chosen one)
  outer_iterations:  int                   # how many attempts ran
  total_tool_calls:  int
  cycle_count:       int | null
  final_diagnosis:   string | null         # populated only on failure (or crash)
  tiled_code:        string                # the verified DSL form (phase-1 output)
  per_outer:
    - outer:    int
      success:  bool
      autotune: dict | null                # per-outer autotune block (see autotuner.md);
                                           # absent when --autotune was off or the outer
                                           # never reached the verified-graph step
  autotune:          dict | null           # the chosen outer's autotune block (when present)
  total_tokens:      int                   # summed across every outer attempt's LLM calls
  traces:            list                  # writer-style code/tool-output captures (legacy)
```

`per_outer` is the field the regression runner reads to derive
`outer_passed` / `outer_total` and per-job autotune for its summary.
`total_tokens` is what the outer flow reads to score the bundle.

## Run config

`config.json` at the checkpoint root captures:

- the LLM config dict (with the API key redacted),
- the kernel and preset,
- the resolved dims,
- `max_outer`, `max_turns`,
- `resume_from`, `translator`, `few_shot_paths`.

This is the file you read to know how the run was *invoked*; for the
arguments the actual outer attempts saw, the per-pass `system_prompt.txt`
and `user_prompt.txt` artifacts are the ground truth.

## Per-attempt log

Each `outer_<i>/log.txt` is the human-readable narration of that
attempt's lifecycle: which passes ran, which gate decided what, why a
pass was skipped (already compliant), where the pipeline halted on
failure, and which judge outputs were folded into feedback. The
terminal printout is deliberately minimal (one line per pass entry/exit)
so a multi-attempt run's stdout stays readable; the per-attempt log is
where the detail goes.

## Regression run directory

A regression-runner invocation owns its own timestamped directory (see
[regression_runner.md](regression_runner.md) for the full layout):

```
regression_results/<YYYYmmdd-HHMMSS>/
├── config.json
├── regression.log              # suite lifecycle log (START / PASS / FAIL per job + summary)
├── jobs/<kernel>__<preset>.log # per-job stdout+stderr captured from the subprocess
├── checkpoints/<kernel>__<preset>/   # the per-kernel checkpoint tree above
└── summary.json                # aggregate outcomes; the file the outer flow scores against
```

`summary.json`'s structure is documented in
[regression_runner.md](regression_runner.md).

## Autotune checkpoint tree

Standalone autotune (`run_autotune.py`) owns its own top-level tree.
Per-outer autotune (`run.py --autotune`) writes the same shape under
each outer's `autotune/<kernel>/` subdirectory; the only structural
difference is location.

```
checkpoints_autotune/<YYYY-MM-DD-HHMMSS>/<kernel>/
├── config.json                # kernel, preset, dims, autotune config, resume path
├── baseline.py                # the verified build_graph the run started from
├── baseline_timing.txt        # the baseline timing report
├── progress.json              # crash-safe best-so-far snapshot (see autotuner.md)
├── best.py                    # the lowest-cycles verified build_graph found
├── best_timing.txt            # the best's timing report
├── result.json                # only present on a clean completion of the loop
└── turn_<n>/
    ├── user_prompt.txt
    ├── response.txt
    ├── reasoning.txt
    ├── extracted_code.py
    ├── correctness_result.txt
    ├── timing.txt             # only present when correctness passed
    └── status.txt             # see autotuner.md for the status vocabulary
```

The autotuner's checkpoint tree is structurally simpler than the
implementer's because there is only one pass running and it has only
two real gates (correctness + timing model).

## Where to look when something breaks

| symptom | first file |
|---|---|
| Run died with no obvious output | `<checkpoint-dir>/<kernel>/outer_*/log.txt` (last lines) |
| All outer attempts failed | `<checkpoint-dir>/<kernel>/result.json` `per_outer` |
| A pass burned all its turns | `<...>/outer_<i>/<pass>/turn_*/status.txt` (look for the recurring failure mode) |
| Curious what the model actually saw | `<...>/<pass>/system_prompt.txt` + `turn_<m>/user_prompt.txt` |
| Compliance kept rejecting | `turn_<m>/correctness_result.txt` (PASS) + `status.txt` (`CORRECT_BUT_NONCOMPLIANT`) + the user_prompt of `turn_<m+1>` (carries the violation list) |
| Judge kept rejecting | `turn_<m>/judge_response.txt` |
| Deterministic translator kept failing | `turn_<m>/translate_check/error.txt` or `graph_error.txt` |
| Regression-suite kernel never started | `regression_results/<stamp>/jobs/<kernel>__<preset>.log` |
| Autotuner regressed correctness | `<...>/turn_<n>/correctness_result.txt` (the loop ignores it; baseline is preserved as `baseline.py`) |
| Per-outer autotune crashed | `<checkpoint-dir>/<kernel>/outer_<i>/autotune/<kernel>/progress.json` for last-known best; outer's `result.json` `autotune.status` will be `error` with the exception message |
