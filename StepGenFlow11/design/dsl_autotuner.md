# DSL-form autotuner

A change to the autotuner subsystem (see [autotuner.md](autotuner.md) for the
unchanged surface) so that the autotuning LLM rewrites the DSL-form
`tiled_reference(dims, tensors)` instead of the post-translate
`build_graph(dims, tensors)`. The translator is folded into the per-turn loop;
new perf knobs (`compute_bw`, `par_dispatch`) become first-class kwargs on the
DSL surface and flow through the deterministic translator into STeP nodes.

## Why

The autotuner edits the same surface the implementer's refactor pass already
emits — meaning correctness errors come back in DSL terms (the same vocabulary
the LLM is writing in), not in lowered STeP-IR terms. Knob feedback stays
identical to today (the timing model still runs on the translated build_graph
and references STeP node attributes), but algorithmic-drift feedback becomes
actionable in the surface the LLM controls.

## What changes vs the existing autotuner

| concern | today | after |
|---|---|---|
| What the LLM rewrites | post-translate `build_graph` | post-refactor `tiled_reference` (DSL) |
| Per-turn correctness gate | IR-level only (`_run_graph_correctness`) | DSL-level → translate → IR-level (triple-gate) |
| Resume-from path | walks subtree for a passing `extracted_code.py` (build_graph) | resolves to `dsl_code.py` (DSL) — no build_graph fallback |
| `compute_bw` / `par_dispatch` knobs | only on STeP node ctors (post-translate, set by the LLM directly) | exposed as kwargs on the relevant DSL functions; forwarded by the deterministic translator |
| Autotune system prompt | embeds `ops.py + utility_ops.py + functional.py + timing.py` | embeds `step_dsl.py + timing.py` |
| `_normalize_compute_bw` | unchanged — operates on the translated graph |
| Per-outer integration | unchanged — `outer_<i>/dsl_code.py` already exists at the integration boundary |
| `progress.json` schema | unchanged |
| Failure-trap discipline | unchanged |
| Best-tracking | `best.py` is the lowest-cycles **DSL source** (was build_graph); `best_translated.py` written alongside |
| `general` / `parallel` agent variants | both retained; both retargeted to DSL form |

## Knob plumbing

Two perf knobs become first-class kwargs across three layers.

### DSL surface (`step_dsl.py`)

Each DSL function whose lowered STeP node carries the corresponding knob gains
a single keyword-only kwarg with default 1. The body is unchanged — the kwarg
is inert at eager-exec time. One assertion per kwarg guards against junk
values so the LLM gets clear feedback at the DSL gate rather than silent
flooring at translate time.

**`compute_bw=1`** on 27 DSL functions (all map to STeP nodes that carry
`compute_bw`):

| family | DSL functions |
|---|---|
| binary (→ BinaryMap) | `binary_matmul`, `binary_mul`, `binary_add`, `binary_div`, `binary_is_equal`, `binary_row_wise_append`, `binary_set_offset`, `binary_cache_write_addr_gen` |
| fused (→ BinaryMapAccum) | `binary_map_accum` |
| unary (→ UnaryMap) | `unary_silu`, `unary_square`, `unary_exp`, `unary_rsqrt`, `unary_pow2`, `unary_mul_imm`, `unary_add_imm`, `unary_sub_imm`, `unary_rowwise_sum`, `unary_select_to_scalar`, `unary_to_const_int`, `unary_mask_row` |
| accum (→ Accum) | `accum_add`, `accum_mul`, `accum_max`, `accum_retile_row`, `accum_retile_col`, `accum_signal_req_all_read` |

**`par_dispatch=1`** on 6 DSL functions (all map to STeP off-chip memory ops):

`offchip_load`, `offchip_load_ref`, `dyn_offchip_load`, `random_offchip_load`,
`offchip_store`, `random_offchip_store`.

The two knobs are mutually exclusive — no DSL function gets both. Multi-output
ops (`broadcast`, `parallelize`, `flat_partition`, `eager_merge`) get neither;
their lowered STeP nodes don't carry these knobs. Tile-shape parameters
(`tile_row`, `tile_col`) already flow through DSL→STeP and need no new
infrastructure.

### Translator (`dsl_to_step.py`)

Each handler whose STeP ctor carries the knob gains one extraction line
(`_arg_or_default(call, ..., "compute_bw", "1")` or `"par_dispatch"`) and
forwards the value into the ctor string. Three factory functions
(`_make_binary_map`, `_make_unary_map`, `_make_accum`) and four special-case
handlers cover compute knobs; six memory handlers cover dispatch knobs. The
existing handlers already use this pattern for similar optional kwargs.

**Hardcoded `par_dispatch=1` literals are replaced** in every memory handler
including the offchip_store branch inside `_rewrite_return`. Default-of-1
preserves byte-for-byte translator output for any DSL source that does not
pass the new kwargs.

### Post-translate rescaling

`_normalize_compute_bw` is unchanged. The LLM's DSL `compute_bw=...` values
flow through `translate()`, land on STeP node attributes, and get rescaled
to the budget. The rescaling table prepended to the timing report is also
unchanged. The LLM's mental model — "write ratios in DSL, see post-rescaled
values back" — reads identically to today.

## Per-turn loop

Conversation state carries `current_dsl_code` (the last DSL form that passed
all three correctness gates). On any failure the LLM is re-pointed at this
form, preserving today's regression-doesn't-cascade discipline.

```
LLM proposes new tiled_reference
        │
        ▼
[gate 1] _exec_dsl_ref(dsl_code, dims, tensors)  vs gold
        │ DSL_FAIL → feedback = DSL exec error / mismatch + current_dsl_code
        │ PASS ↓
[gate 2] translate(dsl_code) → build_graph_code
        │ TRANSLATE_ERROR → feedback = translator exception + current_dsl_code
        │ PASS ↓
[gate 3] _run_graph_correctness(build_graph_code, …)  vs gold
        │ IR_FAIL → feedback = "DSL passed, IR diverged" + sim error + current_dsl_code
        │ PASS ↓
[step 4] _measure(build_graph_code, …) → cycles + report
        │ TIMING_ERROR → feedback = timing-model exception + "knob out of range likely" hint
        │ PASS ↓
update current_dsl_code, regenerate report, advance
```

### Status vocabulary

| status | meaning |
|---|---|
| `NO_CODE` | response had no parseable ```python block |
| `DSL_FAIL` | gate 1 rejected — DSL exec error or mismatch with gold |
| `TRANSLATE_ERROR` | gate 2 raised — DSL passed eager exec but the translator could not lower it |
| `IR_FAIL` | gate 3 rejected — DSL exec passed and translate succeeded, but the lowered graph's simulator output disagreed with gold |
| `TIMING_ERROR` | step 4 raised — knob almost certainly out of range |
| `PASS NEW_BEST cycles=N delta=±N` | every gate passed, lower than `best_cycles` |
| `PASS SAME cycles=N delta=0` | every gate passed, equal to `best_cycles` |
| `PASS REGRESSION cycles=N delta=+N` | every gate passed, higher than `best_cycles` |

`SAME` and `REGRESSION` advance `current_dsl_code` (a future proposal may
build on them) but do not update `best_code` / `best_cycles`.

### Per-turn artifacts

Under `turn_<N>/`:

| artifact | written when |
|---|---|
| `user_prompt.txt`, `response.txt` | always |
| `reasoning.txt` | model returned reasoning |
| `extracted_code.py` | a code block was extracted (this is the **DSL source**) |
| `dsl_correctness_result.txt` | gate 1 ran |
| `translated_code.py` | gate 2 succeeded (snapshot of translator output for diagnostics) |
| `translate_error.txt` | gate 2 raised |
| `graph_correctness_result.txt` | gate 3 ran |
| `timing.txt` | step 4 succeeded |
| `timing_error.txt` | step 4 raised |
| `status.txt` | always — fixed vocabulary above |

The shape mirrors the per-pass turn artifacts in
[logging.md](logging.md), with two new artifacts (`translated_code.py`,
`translate_error.txt`) reflecting the new translate stage and one renamed
(`correctness_result.txt` → split into `dsl_correctness_result.txt` and
`graph_correctness_result.txt` so the LLM-facing feedback can name the
gate that failed).

## Baseline measurement

Before the loop starts, the autotuner:

1. Resolves the resume path to `baseline_dsl_code` via `_resolve_resume_dsl`.
2. Runs gate 1 (`_exec_dsl_ref` vs gold). Hard-fails if mismatch — silent
   tuning of incorrect code is unacceptable. Same invariant as today.
3. Runs `translate(baseline_dsl_code)` → `baseline_build_graph`. Hard-fails
   if it raises (a baseline whose translator is broken can't be tuned).
4. Runs gate 3 on the translated graph. Hard-fails on mismatch.
5. Runs `_measure(baseline_build_graph, …)` → `baseline_cycles`.

Persisted as `baseline_dsl.py` (the DSL source — replaces today's `baseline.py`),
`baseline_translated.py` (the translated build_graph for diagnostics), and
`baseline_timing.txt`.

## Resume path

`_resolve_resume_build_graph` is removed. The autotuner uses
`orchestrator._resolve_resume_dsl` (already public-ish, unchanged) which
accepts a `dsl_code.py` file, an `outer_<N>/` directory containing one, or
a checkpoint root searched as `<root>/<kernel_name>/outer_*/dsl_code.py`.
There is no build_graph fallback — pointing the standalone runner at an
old build_graph-only checkpoint fails with `_resolve_resume_dsl`'s
assertion (`No dsl_code.py found under …`).

The per-outer integration in `orchestrator._run_outer_autotune` is
unchanged. It already passes `resume_from=str(outer_dir)`, and the new
resolver finds `outer_dir/dsl_code.py` (which the orchestrator persists at
the end of phase 1). The wiring is byte-for-byte identical to today.

## Prompt

`prompts/autotune_system.txt` and `prompts/autotune_parallel_system.txt`
swap their template placeholders:

- Drop: `{ops_code}`, `{utility_ops_code}`, `{functional_code}`.
- Keep: `{timing_code}`, `{hw_constraints}`.
- Add: `{step_dsl_code}` — verbatim contents of `src/step_dsl.py`.

`build_autotune_system_prompt` (`prompts.py`) is updated to drop the three
removed loaders and add a single new one for `_DSL_PY` (already resolvable
from `tools._DSL_PY`).

The narrative shifts:

- "rewrite this `build_graph(dims, tensors)`" → "rewrite this
  `tiled_reference(dims, tensors)`".
- "STeP API reference (ops.py)" → "DSL surface (step_dsl.py)".
- "every compute op's `compute_bw`" → "every compute DSL call's
  `compute_bw` kwarg (post-translate)".
- "off-chip op may have `par_dispatch > max_par_dispatch`" → "off-chip DSL
  call may have `par_dispatch > max_par_dispatch`".
- Output format closes with "single ```python block containing the full
  updated `tiled_reference`".

A new short paragraph before the budget rules block explains the perf-knob
convention so the LLM does not have to derive it from `step_dsl.py` alone:

> Each DSL function whose lowered STeP node carries a perf knob accepts
> that knob as a keyword-only argument with default 1. Specifically:
> compute DSL calls (`binary_*`, `unary_*`, `accum_*`, `binary_map_accum`)
> accept `compute_bw=N`, and off-chip DSL calls (`offchip_load*`,
> `dyn_offchip_load`, `random_offchip_*`, `offchip_store`) accept
> `par_dispatch=N`. The kwarg is ignored at DSL eval time and consumed by
> the deterministic translator that produces `build_graph` for the timing
> model. Use these to express the relative compute share / dispatch
> parallelism you want each call to receive.

`build_autotune_user_prompt` (`prompts.py`) updates two strings: the
"Current build_graph (correctness verified)" header and the closing
"Output the full updated `build_graph(dims, tensors)`" line, both swapped
to `tiled_reference`.

The `parallel` variant's "do not change any other knob (tiling,
`par_dispatch`, `compute_bw`, `write_back_mu`, etc.)" line stays valid —
the DSL-form analog (don't add or change kwargs on existing calls) is
already covered by that wording.

## Per-outer integration

Unchanged. The existing `_run_outer_autotune` in `orchestrator.py` passes
`resume_from=str(outer_dir)`, the new resolver finds `dsl_code.py` under
it, and the failure-trap branch (`progress.json` recovery) keeps working
without edits. The outer-result `autotune` block schema is unchanged —
see [autotuner.md](autotuner.md) for that contract.

## Best-tracking & progress recovery

- `best.py` becomes the lowest-cycles **DSL source** (was the build_graph).
  This lets a re-run consume `best.py` directly as a starting point.
  `best_translated.py` is written alongside for inspection.
- `best_timing.txt` is the timing report for the translated form (no
  semantic change).
- `progress.json` schema unchanged: `{baseline_cycles, best_cycles, turn,
  last_status}`. The per-outer integration's failure-trap recovery
  consumes it identically.
- `result.json` schema unchanged.

## Testing

| target | test |
|---|---|
| `step_dsl.py` knob kwargs | one representative function per family (`binary_matmul`, `unary_silu`, `accum_add`, `binary_map_accum`, `offchip_load`): the kwarg is accepted, the output matches the no-kwarg call exactly, and `compute_bw=0` / `par_dispatch=0` raise AssertionError |
| `dsl_to_step.translate` knob forwarding | for one DSL function per family with a non-default knob (`binary_matmul(..., compute_bw=4)`, `offchip_load(..., par_dispatch=2)`, etc.): the translated build_graph's AST contains the expected ctor kwarg; AST-walked, no exec |
| `dsl_to_step.translate` BC | a DSL source with no new kwargs translates byte-for-byte identically to a checked-in golden translation for one fixture |
| `_resolve_resume_dsl` from autotune | feeding an `outer_<N>/` dir with a `dsl_code.py` returns its contents (one test from the autotune entry point so a future reorg cannot quietly break the contract) |
| `run_autotune` per-turn loop | mock the LLM to walk the gate-failure ladder (`NO_CODE` → `DSL_FAIL` → `TRANSLATE_ERROR` → `IR_FAIL` → `TIMING_ERROR` → clean `PASS`); assert `status.txt` vocabulary and that `current_dsl_code` was preserved across each failure |
| `run_autotune` baseline | a baseline whose DSL exec mismatches gold raises loud before the loop starts |

Existing tests stay valid without edits: `test_orchestrator_autotune.py`
(integration boundary unchanged), `test_autotune_progress.py`
(`progress.json` schema unchanged), `test_regression_runner.py` and
`test_regression_cli.py` (autotune-overall block unchanged).

A real one-turn smoke test on a small kernel — run as a manual final
check during implementation, not in the test suite — confirms the
prompt + LLM + translate pipeline talks to itself end-to-end.

## Out of scope

- **`write_back_mu`.** Today's translator hardcodes `write_back_mu=False`
  on every relevant op. Exposing it as a third DSL kwarg is a natural
  follow-up but is not part of this change. The `parallel` variant prompt
  line listing knobs to leave alone still mentions it, accurately
  reflecting that it is an unchanged knob.
- **Standalone autotune of legacy build_graph checkpoints.** The hard
  resume-path switch removes this. If needed, a small one-off tool that
  reverse-renders a build_graph back to a DSL form would be a separate
  follow-up; the current change makes no provision for it.
- **Folding this spec into `design/autotuner.md`.** Once shipped, the
  contents of this document supersede the relevant sections of
  `autotuner.md` and should be merged into that file as a single
  reference; this spec exists as a forward-looking plan and can be
  retired then.
