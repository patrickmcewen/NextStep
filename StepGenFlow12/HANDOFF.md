# Multi-pass autotune2 pipeline — in progress

## Goal

Add a configurable multi-pass pipeline to autotune2, the analog of
autotune1's `passes` list in `autotune_config.json`. Each pass branches
off entries in the prior pass's library, exploring multiple variants per
branch. Canonical use case: a tiling pass followed by a parallelism
pass, where each tiling variant becomes its own starting point for the
parallel pass.

## Design decisions (already locked with the user)

1. **Branching expansion** (not augmentation). Pass N runs once per
   selected entry in pass N-1's library; each branch can admit multiple
   new variants. Final library is the union across branches.

2. **Per-pass cap + selection strategy** to control LLM cost:
   `max_baselines_per_node` and `baseline_selection`. Strategies: `all`,
   `min_cycles`, `min_on_chip`, `pareto_diverse` (default — k-medoids
   over normalized (cycles, on_chip) Pareto front via greedy
   farthest-point sampling).

3. **Per-branch context**: each branch's LLM sees only its own baseline
   + admissions inside that branch's conversation. Branches do NOT see
   each other's results. Encourages divergent exploration.

4. **Pass-1 baseline always seeded**: every pass adds `pass1_baseline`
   to its baseline list (alongside selected library entries). Insurance
   against pass N-1 missing fertile starting shapes.

5. **Node-major on-disk layout**:
   `<ckpt>/autotune2/<node>/pass_0_tiling/library.json`,
   `<node>/pass_1_parallel/library.json`. Groups by node, which is the
   debug-time access pattern.

6. **Per-pass stamps with cascade**: stamp pass i =
   `hash(pass_specs[0..i] + plan + extra)`. Editing pass 1's spec
   invalidates pass 1 and 2; editing pass 2's spec only invalidates
   pass 2.

7. **Per-pass overridable knobs**: `fewshot`,
   `max_baselines_per_node`, `baseline_selection`, `attempt_budgets`,
   `max_turns_per_attempt`. Top-level keys are defaults.

8. **No halting** except on exception. Empty-pass outputs are fine — a
   different fewshot might rescue what the prior pass missed.

## Final config schema (target)

```json
{
  "hw_config": {...},
  "max_on_chip_memory": 16777216,
  "attempt_budgets": [null, 1.0, 0.5, 0.1],
  "passes": [
    {
      "name": "tiling",
      "fewshot": "tile_shrink",
      "max_baselines_per_node": 4,
      "baseline_selection": "pareto_diverse"
    },
    {
      "name": "parallel",
      "fewshot": "parallel",
      "max_baselines_per_node": 4,
      "baseline_selection": "pareto_diverse"
    }
  ]
}
```

`passes` absent ⇒ single-pass spec synthesized from CLI args (current
behavior preserved).

## Implementation chunks

| # | Status | Description |
|---|---|---|
| 0 | Done ✓ | `fewshot` param wired through `build_autotune2_system_prompt`; parallel fewshot guide + parallel system-prompt template variant; CLI `--fewshot` flag |
| 1 | Done ✓ | `select_baselines()` helper + 22 unit tests |
| 2 | Done ✓ | `pass1_dsl` → `baseline_dsl` rename inside `_run_*_attempt` + prompt builder + template placeholder; `DesignEntry.breakdown` cached for per-branch breakdown rendering |
| 3 | Done ✓ | `initial_baselines: list[DesignEntry] \| None = None` on `search_leaf` / `search_parent`; per-branch `(baseline, budget)` fan-out; `baseline_{b_idx}_attempt_{a_idx}_b{label}` dir naming; provenance includes branch index; per-branch `baseline_accepted=[b]` isolation |
| 4 | Done ✓ | `autotune(initial_libraries=..., max_baselines_per_node=4, baseline_selection="pareto_diverse", pass_subdir=None)`; per-node `select_baselines` call excludes prior library's pass-1 entry by identity; `pass_subdir` appends `<ckpt>/autotune2/<node>/<pass_subdir>/` when set |
| 5 | Done ✓ | Config parsing (`_resolve_pass_specs`) + per-pass loop in `_run_autotune2`. Per-pass system prompts via `_build_system_prompts`; per-pass stamps cascade via `pass_specs[0..i]` slice in `extra`; node-major layout under `<ckpt>/autotune2/<node>/pass_<i>_<name>/`. `passes` absent ⇒ legacy single-pass layout preserved |
| **6** | **Next** | Top-K promotion + summary writing on the final pass's libraries; integration tests |

## Files changed so far

### Chunk 0 (foundation — done in prior session before this work)

- `prompts/autotune_parallel_fewshot.txt` (new) — skeleton guide for
  shared vs. independent parallelism; includes GEMM M-axis indep
  example transcribed into DSL.
- `prompts/autotune2_system_parallel.txt` (new) — sibling of
  `autotune2_system.txt` with `{parallel_fewshot}` placeholder.
- `src/autotune2/prompts.py` — added `fewshot: str = "tile_shrink"`
  param to `build_autotune2_system_prompt`; validates against
  `("tile_shrink", "parallel")`.
- `run_autotune2.py` — `--fewshot` CLI flag (default `tile_shrink`)
  threaded through `_load_pass1_state`.

### Chunk 1

- `src/autotune2/baseline_selection.py` (new) —
  `select_baselines(lib, k, strategy, exclude)`, `STRATEGIES`
  constant, `_flatten`, `_pareto_front`, `_farthest_point_sample`
  helpers. Excludes by object identity (`is`) so callers can omit
  the pass-1 baseline they re-seed separately.
- `tests/test_autotune2_baseline_selection.py` (new) — 22 tests
  covering all 4 strategies, edge cases (empty lib, k=0, identity vs
  value exclude, multi-cell libraries, ties on front, dominated-entry
  pruning, degenerate zero-span fronts, invalid args).

### Chunk 2

- `src/autotune2/contracts.py` — added `DesignEntry.breakdown: str = ""`
  field, populated on creation so multi-pass branches can render a
  starting-design memory breakdown into the LLM prompt without
  re-running `compose_source` + `score_fn.breakdown`.
- `src/autotune2/search.py` — populated `breakdown` on baseline
  (`_seed_baseline`, `_seed_root_baseline`) and on LLM-admission
  (`_run_leaf_attempt`, `_run_parent_attempt`). Renamed `pass1_dsl` →
  `baseline_dsl` in `_run_*_attempt` parameter and inside the
  `build_autotune2_user_prompt` call. The call sites in `search_leaf`
  / `search_parent` pass `baseline_dsl=pass1_dsl` (the public API name
  stays `pass1_dsl` until chunk 3 reshapes it).
- `src/autotune2/prompts.py` — `build_autotune2_user_prompt` parameter
  `pass1_dsl` → `baseline_dsl`; `{pass1_dsl}` placeholder in
  `_USER_PROMPT_TEMPLATE` → `{baseline_dsl}`. Heading "### Pass-1
  verified design" intentionally kept (accurate in single-pass mode;
  rename to "### Starting design" when chunk 3 actually feeds
  non-pass-1 baselines).
- `tests/test_autotune2_prompts.py` — `_PROMPT_KWARGS` kwarg renamed
  `pass1_dsl=` → `baseline_dsl=` to match the new builder signature.

### Chunks 4 + 5

- `src/autotune2/search.py`:
  - `autotune()` gained `initial_libraries`,
    `max_baselines_per_node` (default 4), `baseline_selection`
    (default `"pareto_diverse"`), and `pass_subdir` kwargs. Per-node,
    `select_baselines()` is called against
    `initial_libraries[node.path]` with the prior library's pass-1
    entry excluded by `id()`. The resulting list is passed as
    `initial_baselines=` into `search_leaf` / `search_parent`.
    `pass_subdir` (when set) is appended to `node_ckpt` so artifacts
    live at `<ckpt>/autotune2/<node>/<pass_subdir>/`.
  - Fixed chunk-3 oversight: `_NODE_OWNED_DIR_PREFIXES` was still
    `("attempt_",)`; updated to `("baseline_",)` so stale-snapshot
    wipes match the new `baseline_*` dir names.
- `run_autotune2.py`:
  - `_resolve_attempt_budgets(...)`: extracted byte-resolution helper
    so it can run per-pass.
  - `_resolve_pass_specs(autotune_config, cli_args, source)`:
    normalizes the optional `passes:[...]` block into a list of
    fully-resolved specs. Resolution order per knob: per-pass key >
    top-level config key > CLI arg (where applicable) > library
    default. Returns `(specs, is_multi_pass)`. Validates per-pass
    keys against `_PASS_KNOBS` + `name`.
  - `_build_system_prompts(tree, dsl_code, fewshot)`: extracted so
    per-pass fewshot overrides can rebuild the system-prompt dict.
  - `_load_pass1_state`: dropped the `fewshot` kwarg; now takes
    `cli_args` and returns `pass_specs`, `is_multi_pass`, `dsl_code`
    in addition to the existing fields. System prompts are no longer
    built here.
  - `_run_autotune2`: pass loop. Per pass `i`, computes pass-specific
    `node_stamps` via `compute_plan_stamps(extra={..., "pass_specs":
    pass_specs[:i+1]})` so editing pass `j` invalidates pass `j+`
    cascadingly. Builds per-pass system prompts, sets `pass_subdir =
    f"pass_{i}_{spec['name']}"` only when the config had a `passes`
    key (legacy single-pass configs keep the flat layout). Threads
    `initial_libraries=prior_pass.libraries` into each pass after
    pass 0.
- `autotune_config_2.json`: switched to the multi-pass schema —
  tiling + parallel passes inheriting top-level
  `max_baselines_per_node=4` / `baseline_selection="pareto_diverse"`.
- `tests/test_autotune2_multipass.py` (new): 10 tests — 4 over
  `autotune(initial_libraries=...)` plumbing + `pass_subdir` layout,
  and 6 over `_resolve_pass_specs` (single-pass synth, top-level
  defaults, per-pass overrides, per-pass `attempt_budgets`, unknown-
  key assertion, missing-name assertion, CLI fallback).

### Chunk 3

- `src/autotune2/search.py`:
  - `_run_leaf_attempt` + `_run_parent_attempt`: added
    `baseline_index: int` (keyword-only). Provenance string is now
    `f"llm_baseline_{baseline_index}_attempt_{attempt_index}_b{blabel}_turn_{turn}"`
    so admitted entries are traceable to their originating branch.
  - `search_leaf` + `search_parent`: added
    `initial_baselines: list[DesignEntry] | None = None`. Internally
    builds `baselines = [pass1_baseline, *(initial_baselines or [])]`
    and replaces the per-budget comprehension with a double loop over
    `(baseline_index, attempt_index)`. Each branch passes its own
    `baseline_dsl=b.dsl`, `baseline_breakdown=b.breakdown`, and
    `baseline_accepted=[b]` — branch isolation per HANDOFF design
    decision 3.
  - Attempt dir naming changed unconditionally:
    `attempt_{i}_b{label}` → `baseline_{b_idx}_attempt_{a_idx}_b{label}`.
    Single-pass mode now writes `baseline_0_attempt_0_b…/`. Chunk 5
    will reshape the parent dir layout anyway
    (`<ckpt>/autotune2/<node>/pass_<i>_<name>/`), so this rename is
    cheap to land now.
- `tests/test_autotune2_search.py`:
  - 3 new tests: `test_search_leaf_default_initial_baselines_matches_current_behavior`,
    `test_search_leaf_extra_baselines_spawn_additional_attempts`,
    `test_search_leaf_per_branch_attempt_dir_naming`.
  - Updated `test_search_leaf_writes_pass1_baseline_and_per_turn_score_artifacts`'s
    expected attempt-dir and provenance strings to the new format.
    (This test remains in the pre-existing-failure set — same root
    cause as before, unrelated to chunk 3.)

### Deliberately NOT changed (deferred to later chunks)

- `_seed_baseline` / `_seed_root_baseline` keep their `pass1_dsl: str`
  parameter — these helpers really do seed the pass-1 baseline.
- `autotune()` input `pass1_dsls: dict[node_path, str]` — that's the
  pass-1 verified DSLs from `_load_pass1_state`, name is correct.
- `autotune()` does not yet thread `initial_baselines` through. That
  is chunk 4: take an `initial_libraries: dict[node_path, NodeLibrary]
  | None`, run `select_baselines()` per node, and pass to the
  per-node `search_leaf` / `search_parent` call.
- The "### Pass-1 verified design" prompt heading text.
- Whether `initial_baselines` entries should be re-inserted into the
  new pass's library (so prior-pass entries survive even if no branch
  admits anything new). Currently they are NOT — only the pass-1
  baseline seed plus this pass's admissions land in the lib. Chunk 4
  owns this call when wiring `initial_libraries`.

## Pre-existing test failures (NOT caused by this work)

```
FAILED tests/test_autotune2_runtime.py::test_verifier_factory_non_root_passes_on_clean_translate_and_build
FAILED tests/test_autotune2_runtime.py::test_verifier_surfaces_derived_output_contracts
FAILED tests/test_autotune2_search.py::test_search_leaf_seeds_baseline_and_admits_llm_proposals
FAILED tests/test_autotune2_search.py::test_search_leaf_writes_pass1_baseline_and_per_turn_score_artifacts
FAILED tests/test_autotune2_search.py::test_search_leaf_appends_feedback_on_parse_fail_then_recovers
FAILED tests/test_autotune2_search.py::test_search_leaf_score_fn_raise_becomes_user_feedback
```

Confirmed pre-existing by `git stash && pytest ... && git stash pop` —
same 6 failures with my changes stashed. Failures are about Pareto /
scorer messaging mismatches, unrelated to multi-pass work. Don't try to
fix these as part of this initiative.

Baseline: 179 passed, 6 failed after chunks 4 + 5 (chunk 3 added 3
tests, chunks 4 + 5 added 10 in `test_autotune2_multipass.py`). Same
6 pre-existing failures, no regressions.

## Next steps

### Chunk 6 (concrete)

Goal: ensure top-K promotion + summary writing operate against the
**final pass's** libraries (which is already what
`_run_autotune2` does after the chunks 4+5 refactor — `result` is
reassigned each pass), and add an integration test exercising a real
multi-pass run end-to-end.

Mechanical considerations:

1. `promote_top_k(root_library=result.root_library(), ...)` is fed the
   final pass's root library, so promotions reflect the last pass's
   admissions. This may be empty if the final pass admitted nothing
   non-Pareto-dominated by the pass-1 baseline. Consider whether the
   summary should also surface earlier-pass admissions as a "history"
   block. Open design question.

2. `write_autotune2_summary` currently consumes only `autotune_result`
   + `rust_promotions`. For multi-pass, decide whether the summary
   should also include per-pass library sizes / per-pass admission
   counts (helpful for debugging "did pass 2 do anything?").

3. Integration test: a tiny 2-node tree, single LLM stub that returns
   a slightly different DSL on each call, run with a 2-pass config,
   assert that pass 1 sees `initial_baselines` derived from pass 0's
   admissions. Layout: `<ckpt>/autotune2/<node>/pass_0_*/` and
   `<ckpt>/autotune2/<node>/pass_1_*/` both populated.

### Verification

```bash
cd /workspace/NextStep/StepGenFlow12
source /root/miniconda3/etc/profile.d/conda.sh && conda activate testenv

# Targeted: chunks 4 + 5
python -m pytest tests/test_autotune2_multipass.py -q

# Full suite — expect 179 passed, 6 failed (pre-existing)
python -m pytest tests/test_autotune2_*.py -q
```

## Project constraints from CLAUDE.md

- Minimal code, no try-except, use assertions
- DO NOT mindlessly agree with the user
- OpenAI Codex reviews code — senior-engineer standard
- `conda activate testenv` to run code in NextStep/
- Keep memory files up to date

## Open follow-up items (not blocking chunk 6)

- `_wipe_node_run_artifacts` in `autotune()` is invoked with the
  per-pass `node_ckpt` (after `pass_subdir` is appended), so wiping
  one pass doesn't blanket-remove sibling pass dirs. But the
  function still uses the `_NODE_OWNED_DIR_PREFIXES = ("baseline_",)`
  + `_NODE_OWNED_FILES` allow-list, which silently skips anything
  unexpected. If the per-pass dir contains items outside that
  allow-list (e.g. a future per-pass debug log), wipe will leak
  them. Tighten if needed.
- `promote_top_k` and `write_autotune2_summary` operate on the final
  pass's library (which already contains the pass-1 seed + that
  pass's admissions). They do NOT see prior passes' admissions unless
  the final pass also re-derived them (via the
  `select_baselines→initial_baselines` path → the LLM, with no
  guarantee it'll re-admit). Chunk 6 owns the call about whether to
  carry prior-pass admissions forward into the final library or
  surface them only in the summary as history.
- `_resolve_pass_specs` validates per-pass keys but not types beyond
  `name`. A future tightening pass could assert
  `isinstance(spec["max_baselines_per_node"], int)` etc. — currently
  the downstream `select_baselines` / `SearchConfig` asserts cover
  most invalid values, but the error message is less helpful than a
  config-level one.
