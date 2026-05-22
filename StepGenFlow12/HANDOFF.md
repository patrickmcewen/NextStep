# autotune2 SimulationManager — Handoff

## Goal

Add a `SimulationManager` seam to the autotune2 pass so the choice of cycle
estimator per verified variant becomes pluggable. Four configurations:

1. `AnalyticalOnly` — every variant scored by the STeP analytical timing model
   (today's behavior).
2. `RustAll` — every variant cycle-scored by the rust simulator; analytical
   still runs for `on_chip` bytes.
3. `DeterministicSplit(rule)` — rule-driven mix.
4. `AgentManager` — LLM decides per variant whether to spend rust time on
   that variant, given a per-pass TIME budget (rust calls take seconds to
   hours) and a cross-kernel calibration library of past
   `(analytical, rust)` measurement pairs.

At end of run, regardless of mode, **one** variant is picked from the root
Pareto and rust-evaluated for ground truth. That single number is the
experiment's comparable metric across all four configs. This replaces
today's `promote_top_k(k=3)`.

## Current progress

**PR1 is landed and verified.** It introduces the seam with zero behavior
change. The autotuner runs identically; the only difference is the
abstraction layer between the search loop and the analytical scorer.

PR1 new modules:
- `src/autotune2/sim_manager.py` — `SimulationManager` Protocol,
  `SimContext`, `SimulationResult`, `AnalyticalOnly` impl.
- `src/autotune2/calibration.py` — `CalibrationRecord` + `CalibrationStore`
  (append-only JSONL, atomic single-write via <4096-byte-line assertion,
  composed source stored by path reference not inline).

PR1 changed modules:
- `src/autotune2/contracts.py` — `DesignEntry.cycle_source: str =
  "analytical"`.
- `src/autotune2/persistence.py` — round-trips `cycle_source`; old snapshots
  load with default `"analytical"`. No format_version bump.
- `src/autotune2/search.py` — `score_fn: ScoreFn` removed everywhere,
  replaced with `sim_manager: SimulationManager`. `_seed_baseline` and
  `_seed_root_baseline` are now `async def` (they `await sim_manager.score`).
  `_safe_score` and `_maybe_breakdown` helpers deleted — their behavior is
  inside `AnalyticalOnly` now. The `autotune(...)` public signature swaps
  `make_score_fn` → `make_sim_manager`.
- `run_autotune2.py` — builds `AnalyticalOnly(score_fn=make_analytical_scorer(...))`
  via the new factory.
- All autotune2 tests updated: `score_fn=fn` → `sim_manager=AnalyticalOnly(fn)`,
  `make_score_fn=...` → `make_sim_manager=lambda t: AnalyticalOnly(...)`.

**PR2 is landed and verified.** Adds `RustAll`, a per-pass wall-clock
`TimeBudget`, `start_pass(...)` on the `SimulationManager` protocol,
CLI plumbing, calibration writes when rust runs, and a `cycle_source`-aware
fast path in `promote_top_k`.

PR2 changes:
- `src/autotune2/sim_manager.py`
  - `TimeBudget(total_seconds=None)`: async-safe via `asyncio.Lock`,
    fixed-size deque for `recent_avg_seconds` (window=8), `reset(...)`
    re-arms for the next pass. `remaining_seconds` returns `inf` for
    unlimited so the standard `<= 0` cutoff is uniform.
  - `RustAll(score_fn, rust_evaluate_fn, time_budget, calibration_store,
    sources_dir, kernel, preset, hw_config_hash, compute_bw, run_id)`:
    runs analytical first (for `on_chip` + breakdown + budget-exhausted
    fallback), then offloads the rust call to `asyncio.to_thread`,
    consumes the budget, and appends a `CalibrationRecord` to the
    store. Composed sources written to `<sources_dir>/<sha256>.py`
    (content-addressed so duplicates de-dup naturally and concurrent
    writers race benignly).
  - `start_pass(time_budget_seconds)` on the Protocol; `AnalyticalOnly`
    no-ops it, `RustAll` resets the shared budget. Wiring lives in
    `search._search_node`, which awaits `node_sim_manager.start_pass(...)`
    on every node task — idempotent because every RustAll node-manager
    shares one TimeBudget via factory closure.
- `src/autotune2/search.py` — `SearchConfig.time_limit_seconds` is the
  pass-level wall-clock limit. `_search_node` awaits
  `start_pass(config.time_limit_seconds)` before fanout. `VariantSummary`
  now carries `cycle_source`, threaded in by
  `render_library_as_variant_summaries`.
- `src/autotune2/prompts.py` — `VariantSummary.cycle_source` rendered
  inline as `cycles=N (analytical|rust)`; `render_accepted_summary`
  does the same; `build_autotune2_system_prompt` appends a
  `_MIXED_SOURCE_CAVEAT` paragraph so the LLM knows mixed-source
  numbers aren't directly comparable.
- `src/autotune2/runtime.py` — `promote_top_k` now partitions picks:
  entries with `cycle_source == "rust"` reuse `entry.cycles` (no
  redundant rust rerun, `rust_dur_ms = 0.0` to signal cached); the
  rest are dispatched to the rust pool concurrently as before.
- `run_autotune2.py`
  - CLI: `--sim-mode {analytical, rust}` (default analytical),
    `--sim-calibration-path` (default `<ckpt>/autotune2/calibration.jsonl`).
  - Pass-spec knob: `time_limit_seconds`.
    Registered in `_PASS_KNOBS` + `_PASS_SPEC_DEFAULTS`, forwarded via
    `SearchConfig.time_limit_seconds`.
  - `make_sim_manager` factory now constructs `RustAll` (with shared
    `TimeBudget`, shared `CalibrationStore`, content-addressed sources
    dir) when `--sim-mode=rust`, else `AnalyticalOnly` as before.
  - `rust_evaluate = build_rust_evaluate_fn(...)` is now constructed
    before the autotune loop (RustAll captures it; `promote_top_k`
    still uses the same closure after).
  - `base_stamp_extra["sim_mode"]` folded into the per-node stamp so
    toggling modes invalidates cached libraries (analytical and rust
    numbers aren't comparable).
- New tests: `tests/test_autotune2_sim_manager.py` (16 tests covering
  TimeBudget arithmetic + concurrent consume safety + reset, RustAll
  success/budget-exhausted/analytical-failure paths, source de-dup,
  calibration round-trip via JSONL) plus 2 added in
  `tests/test_autotune2_runtime.py` for the new `cycle_source == "rust"`
  fast path in `promote_top_k`.

**PR3 is landed and verified.** Adds the `final_pick(k=1)` stage that
replaces `promote_top_k(k=3)` as the default root-pick path, plus the
CLI seam and summary-schema tag so downstream consumers can tell which
path produced the reported number.

PR3 changes:
- `src/autotune2/runtime.py`
  - `final_pick(root_library, rust_evaluate_fn, strategy="min_cycles")`:
    walks the root's full Pareto front (cycles, on_chip), restricts to
    rust-sourced entries when the library is mixed (HANDOFF design
    decision #5 — analytical + rust cycles are incommensurable), then
    picks lowest cycles with on_chip as tiebreaker. Picked entry whose
    `cycle_source == "rust"` reuses `entry.cycles` (no fresh rust call,
    `rust_dur_ms = 0.0`); otherwise the rust evaluator runs exactly
    once. Returns a single-element `list[RustPromotionResult]` so
    `write_autotune2_summary` keeps working unchanged. PR4 will add
    `strategy="agent"` for the `FinalPickAgent`.
  - `write_autotune2_summary(..., root_pick_strategy: str | None =
    None)`: new optional kwarg surfaced in the JSON payload as
    `root_pick_strategy` (e.g. `"final_pick:min_cycles"` or
    `"top_k:3"`). Additive — old callers omitting it get a `null`
    field. No `format_version` bump (no existing field changed
    meaning).
- `run_autotune2.py` — CLI `--root-pick {final_pick, top_k}` (default
  `final_pick`); `--top-k N` retained for back-compat and only
  consulted when `--root-pick=top_k`. The chosen path's tag is
  forwarded as `root_pick_strategy` to `write_autotune2_summary`.
- New tests: `tests/test_autotune2_final_pick.py` (13 tests covering
  min_cycles selection, Pareto-front restriction to rust-sourced
  entries on mixed libraries, cached-cycles reuse for rust-source
  picks, fresh rust call for analytical picks, descendant composition,
  surface-equivalence with `promote_top_k(k=1)` for analytical-only
  libraries, strategy/empty-library assertions, and the summary
  `root_pick_strategy` payload field).

**PR4 is landed and verified.** Adds `DeterministicSplit` (rule-driven
mix, today only `"rust_baselines"`) and `AgentManager` (two-step
in-loop LLM decision: curation agent ranks past `(analytical, rust)`
records, decision agent picks rust-vs-analytical per variant).
`FinalPickAgent` is intentionally deferred to PR5 — the PR4 scope
discussion confirmed shipping the in-loop path first.

PR4 changes:
- `src/autotune2/prompts.py`
  - `build_curation_system_prompt()` / `build_sim_decision_system_prompt()`
    — static system prompts. CurationAgent ranks records by predictive
    signal (matching DSL ops, similar tile shapes, divergent
    analytical/rust pairs). SimDecisionAgent decides rust-vs-
    analytical with explicit lean-toward heuristics on baselines, large
    divergence, and `remaining_seconds < 2 * recent_rust_avg_sec`.
  - `CurationCandidate` dataclass — what the curation agent ranks
    (record_id + composed source + cycle pair + kernel/preset).
  - `build_curation_user_prompt(target_source, candidates, k)` and
    `build_sim_decision_user_prompt(...)` — per-call user prompts.
    Both fail-loud on bad inputs (whitespace in record_id, k out of
    range, unknown variant_kind, duplicate record_id).
  - `parse_curation_response` / `parse_sim_decision_response` — fenced
    JSON parsers, strict (assertion-based). Cardinality, enum
    membership, candidate-set membership all checked.
- `src/agents.py` — `make_curation_agent(llm_config)` and
  `make_sim_decision_agent(llm_config)` factories. Same SDK
  `Agent`/`ReasoningAwareModel`/`_build_model_settings` pattern as the
  existing autotune2 agents. Production callers can swap in a cheaper
  llm_config profile for curation (HANDOFF #7: cheaper-LLM curation);
  for first deployment the same profile is fine.
- `src/autotune2/sim_manager.py`
  - `DeterministicSplit(score_fn, rust_evaluate_fn, time_budget,
    calibration_store, sources_dir, kernel, preset, hw_config_hash,
    compute_bw, run_id, rule="rust_baselines")` — analytical for
    variants, rust for baselines, same budget/calibration plumbing as
    `RustAll`. Single supported rule today; the `_should_use_rust`
    dispatch makes adding new rules a one-line switch.
  - `AgentManager(...)` with the same constructor shape plus
    `curation_agent_fn`, `decision_agent_fn`, `fetch_candidates_fn`,
    `log_warning`. Flow: analytical first → budget check → curation
    (skipped on cold start) → decision agent → execute (`rust`
    consumes budget + appends calibration record; `analytical`
    short-circuits). On *any* exception inside the curation/decision
    path (RPC error, parse failure, candidate fetch crash), the outer
    `_decide` catches, logs via `log_warning`, and degrades to
    analytical. The fenced-JSON parsers themselves stay fail-loud.
    Class constants `MAX_CURATION_CANDIDATES=50` and `CURATION_K=4`
    bound prompt size — promote to ctor args once we have telemetry.
  - `_write_calibration_source` / `_append_calibration` — module-scope
    helpers refactored out of `RustAll`. `RustAll`, `DeterministicSplit`,
    and `AgentManager` all share the persistence path so a future fix
    only happens once.
  - `_extract_agent_text(reply)` helper — mirrors search's
    `_coerce_agent_response` locally so sim_manager doesn't import
    search (avoids a new cycle).
- `run_autotune2.py`
  - `--sim-mode` extended to `{analytical, rust, deterministic-split,
    agent}`; the choices assertion and help text follow.
  - `make_sim_manager` dispatches all four modes. For `agent`, the
    runner up-front constructs the two new agents via the existing
    `make_curation_agent` / `make_sim_decision_agent` factories,
    wraps each in a thin `Runner.run`-backed `AgentFn` (mirrors
    `build_real_agent_fn`'s pattern), and supplies a
    `fetch_candidates_fn` closure that iterates the shared
    `CalibrationStore` filtered by `hw_config_hash`, caps at
    `AgentManager.MAX_CURATION_CANDIDATES`, reads each
    `composed_source_path` from disk, and yields `CurationCandidate`
    rows. Records whose source file got cleaned up are skipped
    (the file may have been GC'd by the user's sources_dir cleanup).
- Tests added:
  - `tests/test_autotune2_sim_manager.py`: 15 new tests (6 for
    DeterministicSplit, 9 for AgentManager). Covers baseline-rust /
    variant-analytical split, budget exhaustion at both decision
    points, analytical-failure passthrough, curation success path,
    fallbacks on curation parse failure, decision parse failure,
    RPC failure, and start_pass budget reset.
  - `tests/test_autotune2_prompts.py`: 17 new tests covering both
    builder and parser surfaces of the new prompts, including
    `float("inf")` budget rendering, empty-curated cold-start
    handling, and every parser fail-loud branch.

**PR5 is landed and verified.** Adds the `FinalPickAgent` for end-of-run
root-Pareto selection (HANDOFF design decision #6 + #8) — the missing
agent from the three-LLM lineup. PR4 intentionally deferred this; PR5
ships it.

PR5 changes:
- `src/autotune2/prompts.py`
  - `build_final_pick_system_prompt()` — static system prompt with the
    lean-toward / avoid heuristics (favor rust-sourced candidates;
    flag analytical entries whose curated records show the model
    under-predicts on similar code) and the fenced-JSON output
    protocol.
  - `FinalPickCandidate` dataclass — one root-Pareto entry rendered to
    the agent (variant_index + cycles + on_chip + cycle_source +
    composed_source + per-candidate curated calibration rows).
  - `build_final_pick_user_prompt(root_path, kernel, preset,
    candidates)` — renders one block per candidate with its composed
    source + curation evidence; asserts indices match list position so
    the agent's `variant_index` reply lands in `[0, N-1]` by
    construction.
  - `parse_final_pick_response(text, num_candidates)` — strict fenced-
    JSON parser; rejects missing fence, non-int / boolean
    `variant_index` (note: `True`/`False` are `int` subclasses in
    Python, so the boolean check is explicit), out-of-range index, and
    missing-key responses.
- `src/agents.py` — `make_final_pick_agent(llm_config)` factory, same
  SDK `Agent`/`ReasoningAwareModel`/`_build_model_settings` shape as
  the curation + sim-decision agents.
- `src/autotune2/runtime.py`
  - `final_pick_agent(...)` — new async function (separate from sync
    `final_pick`; the curation + final-pick LLM calls have to be
    awaited, and converting `final_pick` to async would force every
    existing call site through `asyncio.run`). Walks the root Pareto,
    runs curation per candidate (skipped on cold start), feeds the
    full set to the final-pick agent, then either reuses
    `entry.cycles` (when `cycle_source == "rust"`) or invokes
    `rust_evaluate_fn` exactly once. Returns the same single-element
    `list[RustPromotionResult]` shape so `write_autotune2_summary` is
    path-independent.
  - Fallback policy mirrors `AgentManager`: any exception inside the
    agent flow degrades to `min_cycles` (with the mixed-source
    rust-only restriction preserved), logs via `log_warning`, and
    the run still produces a reportable number. Strict parsers stay
    fail-loud; only the outer wrapper degrades.
  - Single-Pareto short-circuit — when only one non-dominated entry
    survives, promote it directly without any agent round trip.
  - `_root_pareto_entries` and `_build_promotion_for_pick` helpers
    extracted from `final_pick` so the agent path doesn't duplicate
    the Pareto walk or the rust-cycle-reuse logic.
  - `final_pick(strategy="agent")` (sync) now asserts with a
    redirect to `final_pick_agent` instead of the PR3 placeholder
    "PR4 will add 'agent'" message.
- `run_autotune2.py`
  - `--root-pick` choices extended to `{final_pick, top_k, agent}`.
  - The up-front `make_curation_agent` / `_make_agent_call` /
    `fetch_candidates_fn` block, previously only built when
    `--sim-mode=agent`, is now also built when `--root-pick=agent`
    (the two modes share the curation agent + candidate fetcher).
    The sim-decision agent and final-pick agent are constructed
    only when their respective modes are active.
  - The final-pick dispatch `if args.root_pick == "final_pick"` /
    `elif "agent"` / `else "top_k"` awaits
    `final_pick_agent(...)` and tags the summary as
    `"final_pick:agent"`.
- Tests added (`tests/test_autotune2_final_pick.py` + `_prompts.py`):
  - **prompts (10 new)**: static-system-prompt shape, full
    candidate+curation rendering, cold-start empty-curated rendering,
    builder fail-loud on empty / misindexed / unknown-cycle-source
    inputs, parser happy path, parser fail-loud on missing fence /
    out-of-range / non-int / boolean / missing-keys.
  - **runtime (8 new + 1 modified)**: happy path (curation per
    candidate + final pick + rust call), rust-source pick reuses
    cached cycles, single-Pareto short-circuit (no agent calls),
    cold-start path (curation skipped, pick still runs), parser-
    failure fallback to min_cycles, out-of-range-index fallback, RPC-
    failure fallback, mixed-source fallback respects the rust-only
    restriction. The PR3 `test_final_pick_rejects_unknown_strategy`
    is split into `_redirects_agent_strategy_to_final_pick_agent`
    (new assertion message) + an `unknown_strategy` test for
    `"bogus"`.

**PR6 is landed and verified.** Adds the agent-decision telemetry
sink (HANDOFF "Next steps" item #2) and promotes the capacity knobs
(`MAX_CURATION_CANDIDATES`, `CURATION_K`) from class constants to
ctor + CLI args (item #4). The remaining "Next steps" items
(additional `DeterministicSplit` rules, cheaper-model curation
profile) are still open.

PR6 changes:
- `src/autotune2/agent_telemetry.py` (new) — `AgentDecisionRecord` +
  `AgentDecisionStore`. Same atomic-append JSONL shape as
  `CalibrationStore` (single `write()` per line under the 4096-byte
  POSIX limit, asserted at append time, fail-loud on overflow).
  Composed-source body is NOT stored inline; the record carries the
  sha256 hash so a telemetry row joins to a calibration row by
  `composed_source_hash` equality whenever a rust call followed.
  Stage discriminator `"sim_decision" | "final_pick"` so both call
  sites write to one file. Decision tags: `rust`, `analytical`,
  `analytical_budget_exhausted`, `fallback`, `picked`,
  `pareto_short_circuit`. `curation_dur_ms` and `decision_dur_ms`
  are split (different model profiles often back the two calls, so
  prompt-drift on one is easier to isolate); `-1.0` means "call did
  not happen".
- `src/autotune2/sim_manager.py`
  - `AgentManager.__init__` takes `max_curation_candidates`,
    `curation_k`, `telemetry_store` keyword args (defaults preserve
    PR4/PR5 behavior). Asserts
    `1 <= curation_k <= max_curation_candidates` at construction so
    bad CLI flags fail before the first agent call.
  - The old class constants are renamed to
    `DEFAULT_MAX_CURATION_CANDIDATES` / `DEFAULT_CURATION_K` and used
    only as ctor defaults.
  - `_decide` returns a `DecisionOutcome` dataclass carrying
    `decision`, `reason`, `curated_ids`,
    `num_candidates_available`, `curation_dur_ms`,
    `decision_dur_ms`. `score()` writes one telemetry row per call —
    including the budget-exhausted skip (decision tag
    `analytical_budget_exhausted`) and the fallback (decision tag
    `fallback`). Telemetry-store writes are fail-loud (no
    try/except) for parity with `CalibrationStore.append` — a disk
    full now blows up the run rather than silently losing the audit
    log.
- `src/autotune2/runtime.py`
  - `final_pick_agent(...)` takes `curation_max_candidates`,
    `telemetry_store`, `hw_config_hash`, `run_id` kwargs (existing
    `curation_k` retained). Times each curation call and the single
    final-pick call; writes one telemetry row per invocation tagged
    `picked`, `fallback`, or `pareto_short_circuit`.
    `curated_record_ids` on the row reflects the curated set for
    the *picked* variant so "what evidence did the agent see for
    the variant it chose?" is one grep away.
- `run_autotune2.py`
  - CLI flags: `--curation-max-candidates` (default 50),
    `--curation-k` (default 4). Both are forwarded into the
    `AgentManager` ctor AND the `fetch_candidates_fn` closure
    (which previously hard-referenced
    `AgentManager.MAX_CURATION_CANDIDATES`) AND the
    `final_pick_agent(...)` call site. Up-front assertion enforces
    `1 <= curation_k <= max_curation_candidates`.
  - `AgentDecisionStore` constructed at
    `<ckpt>/autotune2/agent_decisions.jsonl` when
    `--sim-mode=agent` or `--root-pick=agent`; sibling to
    `calibration.jsonl` and joinable on `composed_source_hash`.
- New tests (19):
  - `tests/test_autotune2_agent_telemetry.py` (6): JSONL append +
    iter round-trip + parent-dir creation + oversized-record
    fail-loud + 64-way concurrent-append non-interleaving.
  - `tests/test_autotune2_sim_manager.py` (8 new):
    capacity-knob ctor assertions, `max_curation_candidates` cap
    actually limits the prompt input, `curation_k` cardinality
    plumbing, telemetry rows for each of {rust decision,
    analytical decision, budget-exhausted, fallback} plus the
    `telemetry_store=None` opt-out.
  - `tests/test_autotune2_final_pick.py` (5 new):
    telemetry rows for {picked, single-Pareto short-circuit,
    parser-fallback}, capacity-knob assertions, and
    `curation_max_candidates` cap.

**Test status (post-PR6)**: 291 pass / 6 fail. The same 6 pre-existing
failures from PR1/PR2/PR3/PR4/PR5's notes (verified by failure signature —
identical `max_abs_err=1.00e+00` correctness-check mismatches and the
`Your declared input_contracts...non-on-chip args ['x']` wrapper-build
shape mismatch in the score-fn-raise test):
- `tests/test_autotune2_search.py::test_search_leaf_seeds_baseline_and_admits_llm_proposals`
- `tests/test_autotune2_search.py::test_search_leaf_writes_pass1_baseline_and_per_turn_score_artifacts`
- `tests/test_autotune2_search.py::test_search_leaf_appends_feedback_on_parse_fail_then_recovers`
- `tests/test_autotune2_search.py::test_search_leaf_score_fn_raise_becomes_user_feedback`
- `tests/test_autotune2_runtime.py::test_verifier_factory_non_root_passes_on_clean_translate_and_build`
- `tests/test_autotune2_runtime.py::test_verifier_surfaces_derived_output_contracts`

## Design decisions locked in (user confirmed)

1. **The simulation manager only picks which estimate to record.**
   `on_chip` always comes from analytical (rust doesn't surface bytes
   today). Admission to the Pareto library is unchanged — sort on
   `(cycles, on_chip)`, ignore source.
2. **Agent sees the full composed source** per variant, not summaries.
3. **The baseline-scoring sites also go through the manager** (so the
   agent can choose for the baseline too). In `AnalyticalOnly` this is a
   no-op.
4. **Budget is per-pass TIME, not per-call count.** Rust runtime is
   unpredictable (seconds → hours). The agent sees `remaining_seconds`
   and `recent_rust_avg_sec`; the manager hard-cuts off when
   `remaining_seconds <= 0`. Rust calls that started before the cutoff
   run to completion (no mid-call kill).
5. **Single-source result display.** Wherever cycles surface to any LLM,
   render `(cycles, cycle_source)`. Never both side-by-side, even when
   both ran internally. This is encoded structurally via
   `DesignEntry.cycle_source`.
6. **Final-stage fairness**: at run end, one variant from the root Pareto
   is picked (deterministic for non-agent modes: min cycles; agent-picked
   for `AgentManager`). That single variant is rust-evaluated. The result
   is the experiment's reported number — comparable across all configs.
   This replaces today's `promote_top_k`.
7. **Calibration library is shared cross-kernel.** Curation agent (a
   cheaper LLM, e.g. Haiku-class) handles relevance filtering; we don't
   pre-split by kernel.
8. **Three LLM agents will exist** (plus the existing variant-generation
   agent that's untouched):
   - `CurationAgent` — picks relevant calibration records given a target
     composed source. Called as a step by the other two.
   - `SimulationManagerAgent` — in-loop per-variant decision.
   - `FinalPickAgent` — end-of-run selection; **separate from** the
     in-loop manager (different judgment, different prompt).
9. **Variant prompts will explain mixed sources.** Add a one-paragraph
   caveat in the variant-generation system prompt that
   different `cycle_source` tags are not directly comparable.

Agent prompt outlines (system prompt sections, per-call user prompt
structure, output protocol) are documented in the conversation log
preceding this handoff. They are NOT in code yet — they're for PR3+.

## What worked

- **Async Protocol** for `SimulationManager.score`. `AnalyticalOnly` is
  effectively sync but the protocol must be async for future LLM-backed
  managers. The cost (one needless `await`) is negligible.
- **Moving `_safe_score` exception handling into `AnalyticalOnly`.** The
  feedback string is byte-identical to the legacy version; the search
  loop just checks `result.error_feedback is not None` instead of the
  old `score_err is not None`. Same control flow.
- **Backward-compat snapshot load** for the new `cycle_source` field via
  `data.get("cycle_source", "analytical")`. Avoided a format_version bump
  and the resulting forced re-runs of in-flight checkpoints.
- **JSONL `CalibrationStore` with path-by-reference for composed sources.**
  Initial draft inlined the source text, which (a) blew past POSIX
  `PIPE_BUF` so concurrent appends would tear, and (b) needed a
  `try/except` to skip malformed lines. Storing a path reference keeps
  lines under 4096 bytes (asserted at append time), so no `try/except`
  and concurrent writes from multiple processes are safe.
- **Verifying "no behavior change" by git stash + re-run.** Cheap way to
  separate pre-existing failures from regressions; saved time debugging a
  test that turned out to be already broken.

## What didn't work

- **Inlining `composed_source` in `CalibrationRecord`.** Rejected — see
  above. Path reference is what shipped.
- **Initial attempt to gate fallbacks with `try/except`** for malformed
  calibration JSONL lines and for missing `.breakdown` attributes.
  Project style (CLAUDE.md) is hard no on speculative `try/except`; both
  cases were rewritten with assertions for fail-loud behavior.

## Next steps (PR7+: more DeterministicSplit rules + curation model swap)

- Additional `DeterministicSplit` rules — `"rust_better_than_baseline"`
  (rust baselines + variants whose analytical cycles beat the
  node's baseline by >threshold) needs per-node baseline tracking
  inside the manager. The HANDOFF "e.g." example. CLI surface:
  `--sim-deterministic-rule {rust_baselines, rust_better_than_baseline}`
  + a `--rust-improvement-threshold` knob (default 0.20).
- Curation model swap — production should run CurationAgent on a
  cheaper profile (HANDOFF #7). Plumb a separate `--curation-model`
  CLI flag and route through `load_llm_config(...)` independently.
  Now also a good time: with PR6 splitting `curation_dur_ms` and
  `decision_dur_ms` in the telemetry, the latency win from a
  smaller curation model is directly measurable post-hoc.

PR6 added agent-decision telemetry to
`<ckpt>/autotune2/agent_decisions.jsonl` (one row per
`AgentManager._decide` call and per `final_pick_agent` invocation,
joinable to `calibration.jsonl` via `composed_source_hash`) — use it
when tuning prompts. Capacity knobs are now CLI flags
(`--curation-max-candidates`, `--curation-k`); promote further to
per-pass spec knobs only after telemetry shows real variance across
passes.

## Manual smoke checklist (PR2)

Before PR3, run one end-to-end sanity check that PR2's wiring actually
works in a real autotune2 invocation (the test suite covers the unit
behavior but doesn't exercise the StepDB rust subprocess):

- `python run_autotune2.py <existing outer_dir> --sim-mode rust` with a
  small `time_limit_seconds` (e.g. `60`) on the smallest available
  kernel. Confirm `<ckpt>/autotune2/calibration.jsonl` accumulates
  records and `<ckpt>/autotune2/calibration_sources/` fills with
  sha256-named files.
- Same run with the budget set to `0`: every variant should fall back
  to analytical and the JSONL should stay empty.
- `--sim-calibration-path /tmp/shared.jsonl` to verify the override
  works (subsequent runs append to the same file, the sources dir
  lives alongside it).

## Quick reference

- Primary working dir: `/workspace/NextStep/StepGenFlow12`
- Test command:
  `bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate testenv && python -m pytest tests/test_autotune2_*.py --tb=line"`
- Pre-existing failures to ignore when evaluating PR3/PR4/PR5: see
  "Test status" section above (same 6, signatures unchanged since PR1).
- The conversation that produced PR1 contains the full agent-prompt
  outlines (system prompts + per-call user prompt sections + output
  protocols) for `CurationAgent`, `SimulationManagerAgent`, and
  `FinalPickAgent`. Pull those out when starting PR4.
- PR2-relevant code surfaces to inspect when starting PR3:
  - `src/autotune2/sim_manager.py` — `RustAll`, `TimeBudget`, the
    `start_pass` Protocol method.
  - `src/autotune2/runtime.py::promote_top_k` — the partition
    pattern (rust-source entries reuse cached cycles) is the same
    shape `final_pick(k=1)` should take.
  - `src/autotune2/calibration.py` — already wired by RustAll;
    `final_pick` and the future `CurationAgent` read records from
    the same store (filter by `hw_config_hash` per node).
