# Autotuner

The autotuner is a separate subsystem that takes a correctness-verified
`build_graph(dims, tensors)` and runs an LLM loop that proposes
performance-oriented rewrites — different tile sizes, different
parallelism choices, different buffering. Each proposal is gated on
correctness against gold and then scored by the analytical timing model
on STeP IR. The autotuner never produces incorrect graphs by
construction.

There are two ways to run the autotuner:

- **Standalone** (`run_autotune.py`): run after the fact against any
  finished implementer checkpoint. This is the original entry point and
  is what gets used when iterating on the autotuner itself.
- **Per-outer integration** (`run.py --autotune ...`): the implementer
  pipeline calls the autotuner inline at the end of each outer attempt
  that produced a verified `build_graph`. This is the path the
  AbstractionOpt outer flow uses to score a bundle on both correctness
  *and* performance in a single pass.

Either way, the autotuner is not part of the per-kernel pipeline that
produces a graph in the first place — it is a strictly downstream
performance pass with its own checkpoint tree.

## Inputs

The autotuner is invoked with a kernel/preset pair, a model
configuration, an autotune config (hardware constraints + max turns),
and a `--resume` path that points at a *successful* implementer
checkpoint. Resume resolution accepts:

- a path to an `extracted_code.py` directly,
- a path to a turn directory whose `status.txt` says PASS,
- any directory above that — the autotuner walks the subtree to find
  the most recent passing turn under the kernel name.

Picking the latest-modified passing checkpoint is the desired behavior:
the implementer pipeline may produce several passing turns within a
single run, and the deepest/latest one is closest to the form the
implementer was finished with.

The hardware config and constraints (HBM channels, channel latency,
PMU buffer sizes, max compute bandwidth, max parallel dispatch, etc.)
are loaded from a JSON config file (default `autotune_config.json`).
They are used both to drive the timing model and to render the
constraints into the agent's system prompt so the LLM's proposals stay
within the target hardware envelope.

When invoked per-outer from `run.py --autotune`, the resume path is
implicit (the just-finished outer's directory) and the LLM config is
reused from the implementer pipeline; the only autotune-specific
arguments accepted at the `run.py` boundary are `--autotune-config`,
`--autotune-max-turns`, and `--autotune-agent`. The functional pipeline
and the autotuner share an LLM by design — running them under the same
profile is what the integration is for.

## Baseline measurement

Before the loop starts, the autotuner:

1. Resolves the resume path to the baseline `build_graph` source.
2. Runs the orchestrator's correctness checker on the baseline. This
   is a hard invariant: if the baseline doesn't match gold, the run
   aborts loud rather than silently tuning incorrect code.
3. Measures baseline `total_cycles` via the analytical timing model.
4. Builds a verbose timing report (graph structure, per-node timing
   breakdown, critical path) — the same report shape the LLM will see
   on every subsequent turn.

The verified baseline source and its timing report are persisted as
`baseline.py` / `baseline_timing.txt` under the autotune checkpoint
directory.

## Loop

Each turn:

1. Build the user prompt from the current code, the current timing
   report, the baseline cycles, and the best-so-far cycles.
2. Send to the LLM. Extract a Python code block. If extraction fails,
   burn one turn with a "no code block" reminder.
3. Run correctness against gold. If correctness fails, the turn is
   rejected and the next prompt asks the model to re-base on the *last
   correct* code (kept as `current_code`) — *not* the failed proposal.
   This is the key invariant: regressions don't cascade.
4. If correct, run the timing model. A timing-model exception (knob out
   of valid range) is reported back distinctly from a correctness
   failure.
5. Compare new cycles against `best_cycles`. Tag the turn `NEW_BEST`,
   `SAME`, or `REGRESSION` and persist a `status.txt` line that
   includes the cycle count and delta.
6. Update `current_code` / `current_report` to the proposal (regardless
   of whether it improved the best — moving forward through "same" or
   "regression" turns is allowed because a future proposal may build on
   them).
7. If the proposal improved the best, persist `best.py` /
   `best_timing.txt` and update `best_code` / `best_cycles`.

The loop runs for a fixed turn budget — there is no early-stop on
no-improvement. Empirically the LLM often finds wins late in the budget
after dead-end exploration.

A passing baseline plus an autotuner that always tracks the best gives
a monotone-non-degrading guarantee: `best_cycles ≤ baseline_cycles`
always, regardless of what the LLM does.

## Two agent variants

The autotuner ships two system prompts behind a `--agent` switch:

- **`general`** — covers tile-size knobs, compute bandwidth, and
  larger structural rewrites.
- **`parallel`** — narrowly focused on inserting and retuning
  `Parallelize` / `StaticReassemble` nodes.

The two share the same prompt-fill machinery (timing-model source,
ops surface, hardware constraints). The split exists because the
parallelism space is qualitatively different from the knob-tuning
space and benefits from a more focused prompt; the two are not
strictly disjoint and can be run in sequence on the same baseline.

## Timing report

The verbose per-node timing report is what the LLM sees each turn.
It has three sections:

1. **Graph structure** — predecessors and successors per node, listed
   in topological order.
2. **Per-node timing** — for each node: input rates (T_fire, OTPC,
   N_fire), output times (st, ICD, fto, end), throughput (OCI, OTI,
   ICI), the per-predecessor `NIT` map, and an OCI max-breakdown that
   labels each candidate term by the predecessor it came from with a
   `WINS` marker on the binding term. The breakdown encodes the timing
   model's case analysis (off-chip op vs on-chip op, one-shot
   predecessor vs steady-state predecessor) so the model can read off
   *why* a given node is the bottleneck.
3. **Critical path** — the last-finishing leaf, the latency chain
   (walks `st = max-pred-fto` back to a source), and the throughput
   origin (walks OCI argmax back to a self-gated node), labeled with
   whether the regime is throughput-bound or latency-bound based on
   what fraction of total cycles is steady-state.

The report is regenerated on every successful turn so the model sees
the *new* bottleneck after each proposed rewrite.

## Per-turn artifacts

Every autotune turn writes:

- `turn_<N>/user_prompt.txt`
- `turn_<N>/response.txt`
- `turn_<N>/reasoning.txt` (when the model returned reasoning)
- `turn_<N>/extracted_code.py`
- `turn_<N>/correctness_result.txt`
- `turn_<N>/timing.txt` (when correctness passed)
- `turn_<N>/status.txt` — one of `NO_CODE`, `CORRECTNESS_FAIL`,
  `TIMING_ERROR`, or `PASS <tag> cycles=<N> delta=<±N>`

Plus the run-level files:

- `config.json` — kernel, preset, dims, autotune config, resume path
- `baseline.py` / `baseline_timing.txt`
- `best.py` / `best_timing.txt`
- `progress.json` — running snapshot of `baseline_cycles`,
  `best_cycles`, last-completed `turn`, and `last_status`. Written
  immediately after baseline measurement and re-written at the end of
  every turn iteration. This is the file external code reads to
  recover best-so-far when the autotune loop did not get a chance to
  emit `result.json` (process killed, OOM, time-cap, etc.).
- `result.json` — `baseline_cycles`, `best_cycles`, `speedup`, turns,
  resume path. Only written on a clean completion of the loop.

`progress.json` and `result.json` are deliberately separate: the
former is the crash-safe checkpoint, the latter is the
clean-completion record. Code that summarizes a run should prefer
`result.json` when present and fall back to `progress.json`
otherwise.

## Per-outer integration

Under `run.py --autotune`, autotune runs inside the implementer's
outer-iteration coroutine immediately after a verified `build_graph`
is produced for that outer. Each outer's autotune therefore runs
concurrently with whatever the other outers are still doing, since
the outer iterations themselves run under `asyncio.gather`.

The autotuner's checkpoint root for that outer is
`outer_<i>/autotune/`. Aside from the directory placement and the
implicit resume from the just-finished outer, the loop is identical
to the standalone path — same baseline measurement, same turn
structure, same artifacts.

**Failure trap.** If the autotune loop raises (model timeout, bad
config, etc.) the exception is caught at the
implementer↔autotuner boundary. The outer's functional success is
preserved, and the outer's result dict gains an `autotune` block
with `status="error"`, the exception message, and whatever
baseline / best cycle counts were recoverable from `progress.json`.
This is the *only* try/except in the integration path: the principle
is that an autotune failure is a perf-side outcome, not a
correctness-side outcome, and should not invalidate work the
functional pipeline already finished.

**Outer-result schema.** When `--autotune` is enabled, each outer's
result dict (and the surviving entry in `per_outer`) gains an
`autotune` field:

```json
{
  "status":           "ok" | "error",
  "baseline_cycles":  int | null,
  "best_cycles":      int | null,
  "speedup":          float | null,
  "checkpoint_dir":   str,
  "error":            str   // present only when status == "error"
}
```

When `--autotune` is disabled the field is absent. When an outer's
functional pipeline failed, the field is absent for that outer
(autotune never ran). The regression runner's per-job summary picks
the lowest-`best_cycles` outer with `status=="ok"` and surfaces it as
the job's headline autotune number — see
[regression_runner.md](regression_runner.md).

## Why correctness is re-checked every turn

The system prompt forbids algorithmic changes (the autotuner is told to
only change knobs and parallelism), but the LLM sometimes drifts. Re-
checking correctness against gold every turn turns the prompt-level
constraint into an enforced one: a proposal that subtly breaks the
algorithm fails the gate, gets re-pointed at the last correct code, and
does not corrupt the run. The cost is one extra simulator pass per
turn, which is small compared to the LLM call.
