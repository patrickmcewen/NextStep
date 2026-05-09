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
        ├── plan/                   # only when Phase 0 (decomposition planner) ran; see planner.md
        │   └── iteration_<k>/
        │       ├── tree.json
        │       ├── <node_path>/{reference.py, refactored.py}
        │       └── turns/<node_path>/turn_<N>/{system_prompt.txt, user_prompt.txt, response.txt, reasoning.txt, status.txt}
        ├── pass1/                  # only when Phase 0 ran — per-node Pass-1 artifacts
        │   └── iteration_<k>/      # one per replan iteration (matches plan/iteration_<k>/)
        │       └── <node_path>/[attempt_<i>/]refactor_final/turn_<N>/...
        │                           # attempt_<i>/ layer only present when --node-attempts > 1
        ├── pass2_composed.py       # Pass-2 output: the composed root DSL after name rebinding
        │                           # (only present when Pass 2 ran and succeeded)
        ├── dsl_code.py             # phase-1 verified DSL output (when refactor_final succeeded)
        ├── autotune/               # only present when --autotune was set and this outer succeeded
        │   └── pass_<idx>_<agent>/
        │       └── <kernel>/       # the autotune subsystem's own checkpoint tree
        │           ├── config.json
        │           ├── baseline_dsl.py / baseline_translated.py
        │           ├── baseline_timing.txt / baseline_verbose_timing.txt
        │           ├── progress.json   # crash-safe best-so-far (see autotuner.md)
        │           ├── turn_<n>/...
        │           ├── best.py / best_translated.py
        │           ├── best_timing.txt / best_verbose_timing.txt
        │           └── result.json     # only present on a clean autotune completion
        └── <pass>/                 # one per LLM pass that ran (no-plan path; under Phase 0, see pass1/iteration_<k>/<node_path>/)
            ├── system_prompt.txt   # the rendered system prompt this pass actually used
            └── turn_<m>/
                ├── user_prompt.txt        # the next-turn input the model saw
                ├── response.txt           # assistant text content
                ├── reasoning.txt          # provider-supplied reasoning (when present)
                ├── extracted_code.py      # the parsed code block
                ├── correctness_result.txt # gate output (refactor passes; dsl executor)
                ├── shape_trace.txt        # captured DSL op trace (when correctness raised)
                ├── status.txt             # one-line summary of how this turn ended
                ├── judge_response.txt     # judge gate output (when judge ran)
                ├── judge_reasoning.txt    # judge reasoning (when present)
                └── translate_check/       # post-validator artifacts (refactor_final + auto)
                    ├── step_extracted_code.py     # what the deterministic translator emitted
                    ├── error.txt                  # translator exception, when one was raised
                    ├── graph_error.txt            # simulator exception, when graph failed to execute
                    └── graph_correctness.txt      # gold comparison of the lowered graph
```

The `autotune/pass_<idx>_<agent>/<kernel>/` nesting under each outer
mirrors the autotune subsystem's chain-of-passes execution: each
configured pass writes its own subdirectory, with the next pass
resuming from the previous pass's `best.py`. See
[autotuner.md](autotuner.md) for the chain schema. The whole
`autotune/` block only appears when `--autotune` was set on the
invocation *and* that particular outer's functional pipeline reached
the verified-DSL step.

The `plan/` and `pass1/` subtrees only appear when Phase 0 (the
decomposition planner) ran for that outer. In the legacy single-shot
path (`--no-plan`), the refactor pass writes directly to a top-level
`<pass>/turn_<m>/...` subdirectory under `outer_<i>/` instead.

Under the planner path, Pass-1 artifacts are laid out with the iteration
level above the node:
`pass1/iteration_<k>/<node_path>/[attempt_<i>/]refactor_final/turn_<N>/...`.
The `iteration_<k>/` layer is what keeps each replan iteration's
per-node refactor work isolated on disk; without it, replanning would
overwrite earlier iterations' attempts at the same node path. The
`attempt_<i>/` layer is inserted only when `--node-attempts > 1`. There
is no per-node `pass2/` directory — Pass 2 runs once at the root and
writes a single `pass2_composed.py` artifact directly under `outer_<i>/`.

`status.txt` is the most useful single file for triage — its values
are a fixed vocabulary:

| value | meaning |
|---|---|
| `PASS` | every gate passed; pass loop exited successfully |
| `NO_CODE_EXTRACTED` | response had no parseable code block |
| `LLM_BAD_REQUEST: <exc>` | provider returned a structured error; turn burned, loop continues |
| `FAIL: <error head>` | correctness check rejected the candidate |
| `CORRECT_BUT_NONCOMPLIANT` | correctness PASS, regex compliance found violations |
| `NONCOMPLIANT` | regex compliance found violations under `compliance-first` (no correctness yet) |
| `CORRECT_BUT_JUDGE_REJECTED` | correctness + regex PASS, judge said REJECT |
| `JUDGE_REJECTED` | judge rejected under `compliance-first` |
| `CORRECT_BUT_POST_VALIDATOR_REJECTED` | correctness + regex + judge PASS, post-validator rejected |

The shape trace, judge artifacts, and `translate_check/` subtree only
appear on turns where they ran — their absence is informative. The
deterministic-translate post-validator additionally writes its own
status (`PASS` or `MISMATCH`) into `translate_check/`.

The planner phase has its own status vocabulary on its turns'
`status.txt` files (`LEAF`, `SPLIT_OK: …`, `MALFORMED_SPLIT`, etc.) —
see [planner.md](planner.md).

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
      autotune: dict | null                # per-outer autotune chain (see autotuner.md);
                                           # absent when --autotune was off or the outer
                                           # never reached the verified-DSL step
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
- `resume_from` / `resume_planner`, `translator`, `few_shot_paths`,
- the gate-ordering flag (`check_order`),
- the planner flags (`plan_enabled`, `max_replans`, `node_attempts`,
  `max_plan_depth`, `non_root_sequential`),
- `stateless_refactor`,
- `bundle_dir` (when set).

This is the file you read to know how the run was *invoked*; for the
arguments the actual outer attempts saw, the per-pass `system_prompt.txt`
and `user_prompt.txt` artifacts are the ground truth.

## Per-attempt log

Each `outer_<i>/log.txt` is the human-readable narration of that
attempt's lifecycle: which planner iterations ran, which nodes the
refactor pass touched, which gate decided what, why a pass was skipped
(already compliant), where the pipeline halted on failure, and which
judge outputs were folded into feedback. The terminal printout is
deliberately minimal (one line per pass entry/exit) so a multi-attempt
run's stdout stays readable; the per-attempt log is where the detail
goes.

## Regression run directory

A regression-runner invocation owns its own timestamped directory (see
[regression_runner.md](regression_runner.md) for the full layout):

```
<results-root>/<YYYYmmdd-HHMMSS>/
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
Per-outer autotune (`run.py --autotune`) writes one subdirectory per
chained pass under each outer's `autotune/pass_<idx>_<agent>/<kernel>/`;
the only structural difference is location and the chain wrapping.

```
checkpoints_autotune/<YYYY-MM-DD-HHMMSS>/<kernel>/
├── config.json                  # kernel, preset, dims, autotune config, resume path, agent
├── baseline_dsl.py              # the verified DSL form the run started from
├── baseline_translated.py       # the deterministic translator's output for the baseline
├── baseline_timing.txt          # baseline timing report (compact)
├── baseline_verbose_timing.txt  # baseline timing report (per-node + critical path)
├── progress.json                # crash-safe best-so-far snapshot (see autotuner.md)
├── best.py                      # the lowest-cycles verified DSL form found
├── best_translated.py           # its translated build_graph
├── best_timing.txt              # best's compact timing report
├── best_verbose_timing.txt      # best's verbose timing report
├── result.json                  # only present on a clean completion of the loop
└── turn_<n>/
    ├── user_prompt.txt
    ├── response.txt
    ├── reasoning.txt
    ├── extracted_code.py                # the proposed DSL
    ├── dsl_correctness_result.txt       # gate 1 output
    ├── shape_trace.txt                  # captured DSL trace (when gate 1 raised)
    ├── translated_code.py               # gate 2 output (when gate 2 succeeded)
    ├── translate_error.txt              # gate 2 raised
    ├── graph_correctness_result.txt     # gate 3 output (when gate 3 ran)
    ├── timing.txt                       # only present when timing model succeeded
    ├── timing_error.txt                 # timing model raised
    ├── judge_response.txt               # judge ran
    └── status.txt                       # see autotuner.md for the status vocabulary
```

The autotuner's checkpoint tree carries more artifacts than the
implementer's because the per-turn loop runs four sequential gates
(DSL exec, translate, IR sim, timing) instead of two — each gets its
own per-turn output file so the failure mode is recoverable from
disk.

## Where to look when something breaks

| symptom | first file |
|---|---|
| Run died with no obvious output | `<checkpoint-dir>/<kernel>/outer_*/log.txt` (last lines) |
| All outer attempts failed | `<checkpoint-dir>/<kernel>/result.json` `per_outer` |
| A pass burned all its turns | `<...>/outer_<i>/.../turn_*/status.txt` (look for the recurring failure mode) |
| Curious what the model actually saw | `<...>/<pass>/system_prompt.txt` + `turn_<m>/user_prompt.txt` |
| Compliance kept rejecting | `turn_<m>/correctness_result.txt` + `status.txt` (`*NONCOMPLIANT`) + the user_prompt of `turn_<m+1>` (carries the violation list) |
| Judge kept rejecting | `turn_<m>/judge_response.txt` |
| Deterministic translator kept failing | `turn_<m>/translate_check/error.txt` or `graph_error.txt` |
| Planner refused to split a node | `outer_<i>/plan/iteration_*/turns/<node_path>/turn_*/status.txt` (look for `MALFORMED_SPLIT` / `GUARD_FAILED`) |
| Planner gave up on a node | look for `EXHAUSTED_FALLBACK_TO_LEAF.txt` or `MAX_DEPTH_FORCED_LEAF.txt` under `plan/iteration_*/<node_path>/` |
| Per-node refactor failure under planner | `outer_<i>/pass1/iteration_<k>/<node_path>/.../turn_*/status.txt` (the highest-numbered iteration is the one that ran last) |
| Pass 2 composition failed | `outer_<i>/log.txt` (last lines) — Pass 2 writes no per-turn artifacts; failure escalates to replan |
| Regression-suite kernel never started | `<results-root>/<stamp>/jobs/<kernel>__<preset>.log` |
| Autotuner regressed correctness | `<...>/turn_<n>/dsl_correctness_result.txt` or `graph_correctness_result.txt` (the loop ignores it; baseline is preserved as `baseline_dsl.py`) |
| Per-outer autotune crashed | `<checkpoint-dir>/<kernel>/outer_<i>/autotune/pass_<idx>_<agent>/<kernel>/progress.json` for last-known best; outer's `result.json` `autotune.passes[idx].status` will be `error` with the exception message |
| Autotune chain halted early | `outer_<i>/.../autotune/...` — outer's `result.json` `autotune.halt_reason` and `halted_pass_index` identify which pass; `feasible=False` indicates infeasibility |
