# Autotuner

The autotuner is a separate subsystem that takes a correctness-verified
DSL `tiled_reference(dims, tensors)` and runs an LLM loop that proposes
performance-oriented rewrites — different tile sizes, different
parallelism choices, different buffering. Each proposal is gated on
correctness against gold *and* on a deterministic translate to STeP IR
*and* on simulator output before the analytical timing model scores it.
The autotuner never produces incorrect graphs by construction, and never
silently advances on a proposal whose translator-side semantics drift
from the DSL semantics.

There are two ways to run the autotuner:

- **Standalone** (`run_autotune.py`): run after the fact against any
  finished implementer checkpoint. This is the original entry point and
  is what gets used when iterating on the autotuner itself.
- **Per-outer integration** (`run.py --autotune ...`): the implementer
  pipeline calls the autotuner inline at the end of each outer attempt
  that produced a verified `dsl_code.py`. This is the path the
  AbstractionOpt outer flow uses to score a bundle on both correctness
  *and* performance in a single pass.

Either way, the autotuner is not part of the per-kernel pipeline that
produces the verified DSL in the first place — it is a strictly
downstream performance pass with its own checkpoint tree.

## Why DSL-form

The autotuner edits the same surface the implementer's refactor pass
already emits, so correctness errors come back in DSL terms (the same
vocabulary the LLM is writing in), not in lowered STeP-IR terms. The
deterministic translator is folded into the per-turn loop, which means
algorithmic-drift feedback is actionable in the surface the LLM
controls. The performance side is unchanged — the timing model still
runs on the translated `build_graph` and references STeP node
attributes; perf knobs (`compute_bw`, `par_dispatch`) are first-class
DSL kwargs and flow through the deterministic translator into STeP node
attributes.

## Inputs

The autotuner is invoked with a kernel/preset pair, a model
configuration, an autotune config (hardware, constraints, optionally a
chain of passes — see below), and a `--resume` path that points at a
*successful* implementer checkpoint. Resume resolution accepts:

- a path to a `dsl_code.py` file directly,
- a path to an `outer_<N>/` directory containing one,
- a checkpoint root, in which case the runner globs
  `<root>/<kernel_name>/outer_*/dsl_code.py` and uses the first match.

There is no `build_graph`/`extracted_code.py` fallback. Pointing the
runner at a legacy build_graph-only checkpoint fails loud at the
resolver.

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

## Pass chain

The autotune config can declare a sequence of passes:

```json
"passes": [
  {"agent": "memory",  "max_turns": 12, "feasibility": {"on_chip_bytes": 393216}},
  {"agent": "general", "max_turns": 12, "feasibility": {"on_chip_bytes": 393216}}
]
```

Each entry runs an independent autotuner loop. Pass `i+1`'s baseline is
pass `i`'s `best.py`, threaded through the same resume resolver as a
fresh run — so the next pass starts from the previous pass's accepted
output. Each pass writes its own checkpoint subtree; the chain's
overall baseline / best / speedup is `pass[0].baseline_cycles` →
`pass[-1].best_cycles`.

The chain halts on either of:

- **Infeasibility.** A pass that ends with `feasible=False` (any of its
  declared `feasibility` upper bounds violated by the best it found)
  stops the chain and the per-outer block records `status="halted"`.
- **Pass exception.** A crashed pass records `status="error"` with the
  exception message, and the partial best is recovered from the
  pass's `progress.json` so the harness still surfaces work-in-progress.
  This is the only try/except in the integration path.

When `passes` is absent from the config, `run.py` synthesizes a
single-element chain from `--autotune-agent` and `--autotune-max-turns`
so legacy callers keep working unchanged. `run_autotune.py` (the
standalone CLI) always runs a single pass.

## Agent variants

The autotuner ships three system prompts behind a `agent` selector:

- **`general`** — the workhorse: tile sizes, `compute_bw`,
  `par_dispatch`, `write_back_mu`, broadcast / retile / buffering
  choices, larger structural rewrites.
- **`parallel`** — narrowly focused on inserting and retuning
  `parallelize` / `static_reassemble` / `flat_partition` DSL ops.
- **`memory`** — narrowly focused on driving on-chip / off-chip memory
  totals down (re-tiling, re-buffering, removing unnecessary on-chip
  materialization). The memory agent's prompt embeds a memory-traffic
  shim source (`step_dsl_memory.py`) so the LLM can read off how each
  DSL op contributes to the on/off-chip totals; the agent is told to
  optimize memory rather than cycles.

The three share the same prompt-fill machinery (timing-model source,
DSL surface, hardware constraints). The split exists because the
parallelism and memory spaces are qualitatively different from the
knob-tuning space and benefit from focused prompts; the three are not
disjoint and can be chained on the same baseline.

## Perf knobs as DSL kwargs

Two perf knobs are first-class keyword-only arguments on the DSL
surface:

- **`compute_bw=1`** on every DSL function whose lowered STeP node
  carries `compute_bw` (binary maps, unary maps, accums, fused
  binary-map-accum). Default 1; an assertion guards against zero or
  negative values so the LLM gets DSL-level feedback rather than
  silent flooring at translate time.
- **`par_dispatch=1`** on every DSL function whose lowered STeP node
  is an off-chip memory op (`offchip_load*`, `random_offchip_*`,
  `dyn_offchip_load`, `offchip_store`).

Tile-shape kwargs (`tile_row`, `tile_col`) already flow through DSL→STeP
and need no autotune-specific plumbing.

The two knobs are mutually exclusive — no DSL function carries both —
because their lowered STeP nodes don't overlap. Multi-output ops
(`broadcast`, `parallelize`, `flat_partition`, `eager_merge`) carry
neither.

The deterministic translator forwards each kwarg into the corresponding
STeP node ctor. Default-of-1 preserves byte-for-byte translator output
for any DSL source that does not pass the new kwargs, which is what
makes "edit only the kwargs the autotuner is meant to edit" a clean
contract.

`compute_bw` totals are post-rescaled to the hardware's
`max_total_compute_bw` budget by an unchanged normalizer that operates
on the translated graph; the LLM's mental model — "write ratios in
DSL, see post-rescaled values back in the timing report" — is the same
across all three agent variants.

## Baseline measurement

Before each pass's loop starts, the autotuner:

1. Resolves the resume path to the baseline DSL source.
2. Runs the orchestrator's DSL-level correctness checker on the
   baseline (gate 1).
3. Runs the deterministic translator on the baseline (gate 2). A
   translator failure on the baseline is a hard abort — a baseline
   whose translator is broken can't be tuned.
4. Runs the IR-level correctness checker on the translated graph
   (gate 3). Hard-fails on mismatch.
5. Measures baseline `total_cycles` and computes baseline memory
   totals via the analytical timing model.
6. Builds a verbose timing + memory report — the same report shape the
   LLM will see on every subsequent turn.

Persisted under the pass's checkpoint root: `baseline_dsl.py` (the DSL
source), `baseline_translated.py` (the translated build_graph for
diagnostics), `baseline_timing.txt` (compact report), and
`baseline_verbose_timing.txt` (the full per-node + critical-path report).

## Per-turn loop

Each turn:

1. Build the user prompt from the current DSL, the current timing
   report, the baseline / best cycles, and the active feasibility
   block.
2. Send to the LLM. Extract a Python code block. If extraction fails,
   burn one turn with a "no code block" reminder.
3. **Gate 1 — DSL correctness.** Exec the proposal as
   `tiled_reference(dims, tensors)` against the DSL surface and
   compare its output to gold. Failure routes the LLM back to the last
   accepted DSL.
4. **Compliance.** Run the `refactor_final` regex compliance table
   over the proposal body. Failure routes the LLM back; if a judge is
   configured, its line-specific feedback is appended.
5. **Judge.** Run the `refactor_final` LLM judge against the proposal
   for canonical-form review. The judge sees an explicit "this code's
   output already matches the reference" notice so it evaluates
   structure only.
6. **Gate 2 — translate.** Run the deterministic translator on the
   proposal. A translator exception is reported back distinctly from a
   correctness failure.
7. **Gate 3 — IR correctness.** Exec the translated `build_graph` on
   the simulator and compare its output to gold. A divergence means
   the DSL passed eager exec but the lowered graph doesn't agree —
   typically a translator bug, occasionally a DSL pattern the
   translator doesn't fully model.
8. **Timing model.** Run the analytical timing model. A timing-model
   exception (knob out of valid range) is reported back distinctly from
   a correctness failure.
9. **Promote.** Compare new `total_cycles` and memory totals against
   the running best. Tag the turn `NEW_BEST`, `SAME`, or `REGRESSION`,
   write the status line, and persist `best.py` /
   `best_translated.py` / `best_timing.txt` /
   `best_verbose_timing.txt` if it improved the best.
10. Update `current_dsl` / `current_report` to the proposal — even on
    `SAME` and `REGRESSION` — because a future proposal may build on a
    same-or-worse intermediate. Don't update on a failed gate.

The compliance and judge gates between DSL eager-exec and translate are
inherited from the implementer's `refactor_final` loop. They exist
because a numerically correct DSL can still use disallowed ops or drift
from canonical form, and rejecting that early avoids paying the
translation/IR/timing cost on a doomed proposal.

The loop runs for a fixed turn budget — there is no early-stop on
no-improvement. Empirically the LLM often finds wins late in the budget
after dead-end exploration.

A passing baseline plus an autotuner that always tracks the best gives
a monotone-non-degrading guarantee on the chosen metric: when feasibility
is active, best is selected by `(overshoot, cycles)` lexicographically,
so a feasible turn always beats an infeasible one and an infeasible turn
that moves *closer* to the bounds can still become the new best.

### Status vocabulary

| status | meaning |
|---|---|
| `NO_CODE` | response had no parseable ```python``` block |
| `DSL_FAIL` | gate 1 rejected — DSL exec error or mismatch with gold |
| `NONCOMPLIANT` | regex compliance found banned ops |
| `JUDGE_REJECTED` | compliance passed but the judge rejected canonical form |
| `TRANSLATE_ERROR` | gate 2 raised — DSL passed eager exec but the translator could not lower it |
| `IR_FAIL` | gate 3 rejected — DSL exec passed and translate succeeded, but the lowered graph's simulator output disagreed with gold |
| `TIMING_ERROR` | timing model raised — knob almost certainly out of range |
| `PASS NEW_BEST cycles=N delta=±N on_chip=… off_chip=…` | every gate passed, lower than `best_cycles` (or feasibility-better) |
| `PASS SAME cycles=N delta=0 on_chip=… off_chip=…` | every gate passed, equal to `best_cycles` |
| `PASS REGRESSION cycles=N delta=+N on_chip=… off_chip=…` | every gate passed, higher than `best_cycles` |

`SAME` and `REGRESSION` advance `current_dsl` (a future proposal may
build on them) but do not update `best_cycles`.

### Per-turn artifacts

Under `turn_<N>/`:

| artifact | written when |
|---|---|
| `user_prompt.txt`, `response.txt` | always |
| `reasoning.txt` | model returned reasoning |
| `extracted_code.py` | a code block was extracted (this is the **DSL source**) |
| `dsl_correctness_result.txt` | gate 1 ran |
| `shape_trace.txt` | gate 1 raised an exception (captured DSL op trace) |
| `judge_response.txt`, `judge_reasoning.txt` | judge ran |
| `translated_code.py` | gate 2 succeeded (snapshot of translator output for diagnostics) |
| `translate_error.txt` | gate 2 raised |
| `graph_correctness_result.txt` | gate 3 ran |
| `timing.txt` | timing model succeeded |
| `timing_error.txt` | timing model raised |
| `status.txt` | always — fixed vocabulary above |

## Run-level artifacts

Under each pass's checkpoint root (`<...>/<kernel>/`):

- `config.json` — kernel, preset, dims, autotune config, resume path,
  resolved baseline source, max_turns, agent variant.
- `baseline_dsl.py` / `baseline_translated.py` /
  `baseline_timing.txt` / `baseline_verbose_timing.txt`.
- `best.py` (the lowest-cycles **DSL source**) /
  `best_translated.py` / `best_timing.txt` /
  `best_verbose_timing.txt`. `best.py` being the DSL form is what lets
  the next pass in the chain consume it directly as a starting point.
- `progress.json` — running snapshot of `baseline_cycles`,
  `best_cycles`, last-completed `turn`, `last_status`, plus per-side
  on-chip / off-chip byte totals (baseline / best / last). Written
  immediately after baseline measurement and re-written at the end of
  every turn iteration. This is the file the per-outer integration
  reads to recover work-in-progress when the autotune loop did not get
  a chance to emit `result.json` (process killed, OOM, time-cap, etc.).
- `result.json` — `baseline_cycles`, `best_cycles`, `speedup`,
  `feasible`, memory totals, turn count, resume path. Only written on
  a clean completion of the loop.

`progress.json` and `result.json` are deliberately separate: the
former is the crash-safe checkpoint, the latter is the
clean-completion record. Code that summarizes a run should prefer
`result.json` when present and fall back to `progress.json`
otherwise.

## Per-outer integration

Under `run.py --autotune`, autotune runs inside the implementer's
outer-iteration coroutine immediately after a verified `dsl_code.py`
is produced for that outer. Each outer's autotune therefore runs
concurrently with whatever the other outers are still doing, since
the outer iterations themselves run under `asyncio.gather`.

The autotuner's checkpoint root for that outer is
`outer_<i>/autotune/`; each pass in the chain gets a subdirectory
`pass_<idx>_<agent>/`. Aside from the directory placement, the implicit
resume from the just-finished outer, and the chain-pass orchestration,
the loop is identical to the standalone path — same baseline
measurement, same turn structure, same artifacts.

**Failure trap.** If a pass raises (model timeout, bad config, etc.)
the exception is caught at the implementer↔autotuner boundary. The
outer's functional success is preserved, the chain halts at that pass,
and the outer's result dict gains an `autotune` block whose halted
pass entry has `status="error"`, the exception message, and whatever
baseline / best cycle counts and memory totals were recoverable from
that pass's `progress.json`.

**Outer-result schema.** When `--autotune` is enabled, each outer's
result dict (and the surviving entry in `per_outer`) gains an
`autotune` field:

```
{
  "status":             "ok" | "halted" | "error",   // chain-level
  "halt_reason":        "infeasible" | "error" | null,
  "halted_pass_index":  int | null,
  "error":              str | null,
  "checkpoint_dir":     str,
  "overall": {
    "baseline_cycles":  int | null,    // first pass's baseline
    "best_cycles":      int | null,    // last pass's best
    "speedup":          float | null,
    "feasible":         bool
  },
  "passes": [
    {
      "index":            int,
      "agent":            "general" | "parallel" | "memory",
      "status":           "ok" | "error",
      "baseline_cycles":  int | null,
      "best_cycles":      int | null,
      "speedup":          float | null,
      "feasible":         bool,
      "baseline_on_chip_bytes":  int | null,
      "best_on_chip_bytes":      int | null,
      "baseline_off_chip_bytes": int | null,
      "best_off_chip_bytes":     int | null,
      "turns":            int | null,
      "checkpoint_dir":   str,
      "error":            str   // present only when status == "error"
    },
    ...
  ]
}
```

When `--autotune` is disabled the field is absent. When an outer's
functional pipeline failed, the field is absent for that outer
(autotune never ran). The regression runner's per-job summary picks
the lowest-`overall.best_cycles` outer with `status` in `{"ok",
"halted"}` and surfaces it as the job's headline autotune number — see
[regression_runner.md](regression_runner.md).

## Why correctness is re-checked every turn

The system prompt forbids algorithmic changes (the autotuner is told to
only change knobs, retiling, parallelism, and buffering choices), but
the LLM sometimes drifts. Re-checking correctness against gold every
turn — through the full DSL → translate → IR chain — turns the
prompt-level constraint into an enforced one: a proposal that subtly
breaks the algorithm fails the gate, gets re-pointed at the last
accepted DSL, and does not corrupt the run. The cost is one extra
simulator pass plus the translator pass per turn, which is small
compared to the LLM call.
