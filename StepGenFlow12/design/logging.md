# Logging

A StepGenFlow run produces a layered, on-disk record sufficient to
reconstruct what every LLM saw and what every gate decided. There are
three logical scopes: the per-kernel run (one invocation of the
implementer pipeline), the regression suite (one invocation of the
batch driver wrapping many per-kernel runs), and the autotune run
(`run_autotune2.py`, a separate post-pipeline pass). Each owns its
own tree.

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

Autotune2 runs as a separate, standalone post-pipeline pass against a
finished `outer_<i>/` checkpoint (see [autotuner.md](autotuner.md));
it does not write inside the implementer's checkpoint tree above. The
autotune2 tree is described in its own section further down.

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
  total_tokens:      int                   # summed across every outer attempt's LLM calls
  traces:            list                  # writer-style code/tool-output captures (legacy)
```

`per_outer` is the field the regression runner reads to derive
`outer_passed` / `outer_total` for its summary. `total_tokens` is
what the outer flow reads to score the bundle. Autotune outcomes
are not embedded in this file — autotune2 runs as a separate
post-pipeline pass against a chosen outer and writes its own
`autotune2_summary.json` (see below).

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

## Autotune2 checkpoint tree

`run_autotune2.py` snapshots the source checkpoint into a fresh
timestamped directory under `--checkpoint-dir` (default: parent of
the source `<ts>/` directory). Sibling `outer_*` directories of the
chosen outer are excluded from the copy so the snapshot stays
small. All autotune2 artifacts live under the snapshot copy; the
original outer is never modified.

```
<checkpoint-dir>/<YYYY-MM-DD-HHMMSS>/                # snapshot root
├── config.json                              # copied from the source <ts>/config.json
└── <kernel>/
    └── outer_<N>/                           # the chosen source outer, copied
        ├── plan/  pass1/  pass2_composed.py  dsl_code.py   # carried forward from the source
        ├── autotune2_summary.json           # final summary — root_pareto, rust_winners, best_rust_entry
        └── autotune2/                       # everything autotune2 writes lives under here
            ├── _rust_work/                  # composed sources passed to StepDB/evaluate.py
            │   └── step_impl.py
            └── <node_path>/                 # one subtree per plan-tree node
                ├── system_prompt.txt        # the rendered autotune2 system prompt for this node
                ├── pass1_baseline_score.json   # baseline (cycles, on_chip, provenance)
                ├── variants.py              # declarative variant registry — overwritten on each admission
                └── attempt_<i>/turn_<j>/
                    ├── user_prompt.txt
                    ├── response.txt
                    ├── reasoning.txt        # when the agent returned reasoning summaries
                    ├── tokens.json          # when the agent returned a usage object
                    ├── extracted_code.py    # the parsed DSL block (when YAML/python parsed)
                    ├── composed_source.py   # wrapper + leaf + descendants (when the verifier was reached)
                    ├── verify_result.txt    # "PASS" or the full gate-feedback string
                    ├── score.json           # admitted entries' (cycles, on_chip, provenance)
                    └── status.txt           # see autotuner.md for the per-turn status vocabulary
```

The per-node `<node_path>` mirrors the planner-tree path (e.g.
`attention_o_proj/attention/attention_compute`). Each node runs
its own per-turn loop independently; sibling subtrees fan out in
parallel under `asyncio.gather` with the only ordering constraint
being "children before parent" (parents render each child's library
as a variant table in the user prompt). For parent nodes the
`composed_source.py` artifact is the *first* Cartesian combination's
composed source — the parent DSL is identical across combinations;
descendants differ.

`autotune2_summary.json` is the canonical handoff artifact:

```
{
  "root_path":          "<root node path>",
  "library_sizes":      {"<node_path>": <num cells>, ...},
  "root_pareto":        [{"cycles": ..., "on_chip": ..., "provenance": ...}, ...],
  "rust_winners":       [
    {"analytical_cycles": ..., "analytical_on_chip": ...,
     "rust_cycles": ..., "rust_dur_ms": ..., "provenance": ...},
    ...
  ],
  "best_rust_entry":    {<same shape>} | null,
  "best_composed_source": "..."             // only when --include-sources was passed
}
```

The summary is sorted by `rust_cycles` ascending — `rust_winners[0]`
== `best_rust_entry` when any promotion ran. A rust evaluator
failure on a promoted entry is a hard assertion (not silently
skipped); the file is only written on clean completion of the run.

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
| Autotune2 turn failed verification | `<snapshot>/<kernel>/outer_<N>/autotune2/<node_path>/attempt_<i>/turn_<j>/verify_result.txt` — carries the gate-feedback string the LLM saw next turn |
| Autotune2 LLM kept emitting bad contracts | look for `WRAPPER_BUILD_FAIL` / `BAD_CHILD_PICK` in `<...>/turn_<j>/status.txt` |
| Autotune2 timing model crashed | `<...>/turn_<j>/status.txt == SCORE_FAIL`; the traceback is embedded in `verify_result.txt` and was already routed back as next-turn feedback by `_safe_score` |
| Where is the best variant? | `autotune2_summary.json` at the snapshot outer dir — `best_rust_entry` gives the rust-validated winner; pass `--include-sources` to embed its composed source |
