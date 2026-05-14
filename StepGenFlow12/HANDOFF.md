# StepGenFlow12 — Bottom-up Tree-DP Autotuner (autotune2) — Handoff

## Goal

Build a new autotuner that attaches after Pass-1 + Pass-2 in `StepGenFlow12`,
using **bottom-up tree DP** over the planner tree with **per-node Pareto
libraries**. Each library entry is a (cycles, on_chip)-Pareto-optimal DSL
variant of that node. The parent autotuner picks among its children's
variants (and proposes glue) to build its own library, recursively up to the
root. Top-K root entries are then promoted from the cheap analytical model to
the expensive rust simulator.

Lives in `src/autotune2/` (the existing flat `src/autotune.py` is kept as a
comparison baseline; do not modify it).

## TL;DR — where things stand

**Algorithmic core, production wire-up, and tests are all complete.** 112
autotune2 tests pass (`tests/test_autotune2_*.py`, ~2s). The end-to-end CLI
(`run_autotune2.py`) successfully resolves all state from a saved outer-dir
checkpoint produced by a fresh pass-1 run.

What remains is **iterative debugging of the real-world execution path** —
the user is running the autotune2 driver against actual checkpoints and
surfacing edge cases (e.g. multi-output node wrapping, rank-0 stream
outputs); these are fixed one-by-one as they appear.

## Run the tests

```bash
cd /workspace/NextStep/StepGenFlow12
PY=/root/miniconda3/envs/testenv/bin/python
$PY -m pytest tests/test_autotune2_*.py    # 112 tests, ~2s
```

Note: `tests/test_blackbox_stub.py` has **6 pre-existing failures** unrelated
to autotune2 (`torch.equal` on `StepTensor`). Leave alone unless asked.

## Run the autotuner end-to-end

```bash
$PY run_autotune2.py <outer_dir> \
  --kernel <kernel_name> --preset <preset> \
  --autotune-config <path/to/autotune_config_*.json> \
  [--model gpt-oss-120b] [--config <llm_profile.json>] \
  [--max-turns-per-attempt 3] [--max-attempts 5] \
  [--check-order correctness-first|compliance-first|always-both] \
  [--top-k 3]
```

`--max-attempts 0` runs only the pass-1 baseline (no LLM calls; useful smoke
test). Requires a fresh pass-1 run that wrote per-node `contract.pkl` files
(see "Pass-1 contract dump" below).

## File layout

```
src/autotune2/
  __init__.py
  contracts.py        TensorContract, DesignEntry, NodeLibrary, freeze_contracts
  pareto.py           dominates, insert_pareto, cull_top_T, dsl_dedup_hash
  stubs.py            invert_input_contract, apply_output_contract,
                      make_variant_stub, emit_variants_module
  compose.py          compose_source, cartesian_compose, compose_into_library,
                      make_analytical_scorer
  prompts.py          SELF-CONTAINED system/user prompt builders +
                      render_accepted_summary + parse_autotune2_response
  search.py           NodePromptInputs, search_leaf, search_parent,
                      autotune driver (per-node AgentFn via agent_factory),
                      build_synthetic_wrapper_for_node (multi-output aware)
  runtime.py          pick_top_k_pareto_entries, promote_top_k,
                      build_real_agent_fn, build_real_verifier_fn,
                      build_rust_evaluate_fn

tests/                test_autotune2_{contracts,stubs,compose,prompts,search,runtime}.py
run_autotune2.py      CLI entry point (parallel to run.py)
```

## Locked Design

### Architecture
- Bottom-up tree DP over the pass-1 plan tree. Post-order traversal via
  `plan_tree.iter_topological()`.
- Each `PlanNode` has a `NodeLibrary` indexed by `(input_contracts →
  output_contracts → Pareto front of DesignEntry)`.
- Pass-1 original design is always seeded in the library as the rollback
  baseline.

### Contracts
- A `TensorContract` is `(reshape, permutation)` applied to the vanilla
  PyTorch tensor: `x_in = x_vanilla.reshape(reshape).permute(*permutation)`.
- Contracts apply only to on-chip stream args. RAW args are always vanilla
  (no variant axis). RAW classification is pass-1-frozen via
  `Contract.arg_is_raw`.
- Outputs are always on-chip streams — every output has a contract.

### Per-node Contract pickle dump (pass-1 → autotune2 bridge)
- Pass-1 (`orchestrator._pass1_walk`) pickles each stamped non-root
  `Contract` to `<outer_dir>/pass1/iteration_<MAX>/<child.path>/contract.pkl`
  after the contract-harvest loop. Pickle (not JSON) because `tiled_values`
  carries `torch.Tensor` / `list[int]` payloads.
- `run_autotune2._load_pass1_state` reads these back. Outer-dirs created
  before this dump was added will fail loudly with a clear actionable
  message; re-run pass-1 to populate.

### LLM ↔ autotuner boundary
- LLM owns: tile sizes, parallelism, retile patterns, contract choices,
  glue DSL.
- Autotuner owns: Cartesian product over child Pareto points within
  LLM-chosen contracts, analytical scoring, Pareto culling, top-K rust
  promotion at root.
- LLM never sees stub bodies — only declarative contract metadata expressed
  as `vanilla.reshape(...).permute(...)`.

### Conversational agent + multi-attempt search (decided this session)
- `AgentFn = Callable[[list[dict]], Awaitable[str]]` — takes the
  conversation list, returns the next assistant message. The system prompt
  is baked into the underlying SDK `Agent` (built by `make_autotune2_agent`)
  so each node gets its own agent via `agent_factory(system_prompt)`.
- `SearchConfig`: `max_turns_per_attempt` (default 3), `max_attempts`
  (default 5), `check_order` (mirrors pass-1's 3 modes).
- Search loop pattern (option (c) from design discussion — hybrid):
  - **Outer**: `for attempt in range(max_attempts)`
  - **Inner**: `for turn in range(max_turns_per_attempt)` — failure feedback
    accumulates within an attempt (parse error → user msg; verify failure →
    gate feedback as user msg). The conversation persists across turns.
  - **On success**: admit to library, break inner, start fresh attempt.
  - **Fresh attempts**: render the current Pareto front via
    `render_accepted_summary` into the opening user prompt so the LLM
    targets gaps (not duplicates).

### Self-contained autotune2 prompts (rewritten this session)
- `build_autotune2_system_prompt(is_leaf, dsl_code)` and
  `build_autotune2_user_prompt(...)` produce fully independent prompts —
  pass-1's prompts are **not** referenced at the text level.
- Shared content: only the literal contents of `step_dsl.py` (read from
  `_STEP_DSL_PY` the same way pass-1 does), so the DSL surface description
  stays in sync.
- Framing differs: pass-1 is first-time lowering, autotune2 is variant
  generation against an already-verified design.

### 4-gate verifier cascade
- `build_real_verifier_fn` wires `_gate_correctness` + `_gate_compliance` +
  `_gate_judge` + `_gate_post_validator` from `orchestrator.py` exactly as
  pass-1 does, with `check_order` honoring the same 3 modes
  (correctness-first / compliance-first / always-both).
- Per-gate artifacts (the gate's own `correctness_result.txt`,
  `translate_check/`, etc.) still go to ephemeral `tempfile.mkdtemp()`
  dirs. The search loop writes its own per-turn checkpoint at
  `ckpt_dir/attempt_<N>/turn_<M>/` with: `user_prompt.txt` (latest user
  message before the LLM call), `response.txt`, `reasoning.txt` (when
  the model returned reasoning), `tokens.json` (when usage was reported),
  `extracted_code.py` (parsed DSL on parse-success), `composed_source.py`
  (parent + descendants the verifier saw), `verify_result.txt`
  (`"PASS"` or full gate feedback), and `status.txt`. The node-level
  `<ckpt_dir>/system_prompt.txt` is written once per node.

### Synthetic `tiled_reference` wrapper (`build_synthetic_wrapper_for_node`)
- For non-root nodes scored in isolation. Phase-5-v1 limit: every TensorArg
  input must be RAW (asserts loudly).
- **Multi-output nodes** are destructured and each output is stored:
  ```python
  _out_0, _out_1, _out_2 = node(...)
  offchip_store(promote_outer(_out_0))
  offchip_store(promote_outer(_out_1))
  return offchip_store(promote_outer(_out_2))
  ```
- Every output flows through `promote_outer` before `offchip_store` to
  satisfy the Rust simulator's stream-rank-≥1 startup constraint
  (`step-perf/src/memory/offchip_store.rs:99`).
- The "exactly one offchip_store at root" compliance rule doesn't apply to
  this wrapper — it exists only for analytical scoring + isolated node
  verification, not as a production root.

### Output protocol (LLM emission)
Fenced YAML block immediately above the DSL python block:
```yaml
child_picks:                  # parent prompts only
  attention_block: 3
parent_input_contracts:       # empty when no input is on-chip
  Q: {reshape: [8, 8, 64], permutation: [1, 0, 2]}
parent_output_contracts:
  out_0: {reshape: [16, 4, 512], permutation: [1, 0, 2]}
```
Parsed by `parse_autotune2_response` with strict schema assertions.

### Top-K rust promotion
Flatten root's cells → drop dominated → keep K lowest-cycles non-dominated
entries. Default K=3. Rust evaluator: `Callable[[composed_source], (cycles,
dur_ms)]` — dependency-injected via `build_rust_evaluate_fn`.

## What Worked

- **Reusing pass-1's `Runner.run` + gate cascade**. The user's clarification
  pushed us away from a direct-Anthropic-call bridge and toward reusing
  pass-1's SDK Agent infrastructure (via a thin `make_autotune2_agent`
  factory). This gave us a consistent OpenAI-compatible endpoint, gate
  feedback, token accounting, and turn-dir layout for free.
- **Self-contained prompts**. Forking the prompt text (sharing only the
  step_dsl reference) means autotune2 can evolve its framing without
  rippling into pass-1.
- **Dependency-injected `AgentFn` / `VerifierFn` / `ScoreFn`**. Kept the
  search loop unit-testable without an LLM or STeP runtime.
- **Per-node agent_factory pattern**. The autotune driver builds one
  AgentFn per node (system prompt baked in) — Tests inject a trivial
  `agent_factory = lambda _sys: my_mock_agent` for full control.
- **Assertion-first error messages**. The `contract.pkl missing`
  assertion in `_load_pass1_state` points users directly at the fix
  (re-run pass-1). The `make_client` API key fallback uses `"None"` as
  a placeholder to match the existing gpt-oss-120b config convention.

## What Didn't Work (corrections during this session)

- **Original plan: addendum-knob on `make_pass1_agent`**. The user pushed
  back: autotune2 should reuse pass-1's *machinery* (multi-turn loop, gate
  cascade, code extraction, judge agent) but have its own
  framing/prompts. Pivoted to self-contained prompts + new
  `make_autotune2_agent` factory.
- **Original plan: single-shot AgentFn `(system, user) -> str`**. Failed
  to surface gate feedback to the LLM across turns. Pivoted to
  conversational `(conversation) -> str` so failure feedback accumulates
  within an attempt, exactly like pass-1's `_run_pass_agent`.
- **Original plan: "v1: gates 1+4 only" verifier**. The user asked for
  the full 4-gate cascade with `check_order` flexibility. Pivoted; all
  four gates are now wired.
- **Stripped llm_config from checkpoint config.json directly**. The
  checkpoint's embedded `llm_config` lacks `api_key`. Fixed by
  resolving the llm config through `load_llm_config(args.config,
  args.model)` (mirrors `run.py`) — the canonical profile loader fills
  in the api_key (or its placeholder) from `<NextStep>/configs/`.
- **Synthetic wrapper returning a tuple to `offchip_store`**. The
  original single-output template broke on multi-output nodes
  (`pre_attention` returns Q, K, V). Fixed by destructuring +
  per-output `offchip_store`. Then surfaced a second issue: rank-0
  stream outputs panic the Rust simulator on startup — fixed by
  wrapping each store input in `promote_outer`.

## Next Steps — debugging the real execution path

The wire-up is structurally complete; the remaining work is whack-a-mole as
real execution surfaces edge cases. Approach for the next agent:

1. **Re-run the autotune2 invocation against a fresh pass-1 checkpoint** —
   the user has been running `prefill_transformer_simple/outer_1` from
   `checkpoints/2026-05-14-174755/`. Each iteration fixes one issue.

2. **Likely classes of issues to expect**:
   - More synthetic-wrapper / DSL composition mismatches at the analytical
     scorer (the `tiled_reference` we generate must translate cleanly via
     `dsl_to_step` and execute under `build_graph`).
   - Verifier feedback shape mismatches between pass-1's `_GateResult` and
     autotune2's `VerifyResult` (currently we just concatenate `feedback`
     strings — might need preserved structure for the LLM to act on).
   - LLM responses that parse but produce DSL the translator rejects —
     ensure parse / verify / score failures all route through clean user
     feedback in the conversation.

3. **Don't suppress real errors with try/except**. Per CLAUDE.md, fix root
   causes; the current code uses assertions throughout for exactly this
   reason. When the user hits an error, look at the failing assert/exception
   and fix the underlying mismatch — don't catch and silently continue.

4. **Test as you go**. Run `pytest tests/test_autotune2_*.py` after every
   change; the test suite catches signature drift in seconds.

5. **Eventual smoke target**: a `--max-attempts 0` baseline-only run should
   produce a root rust-cycle count matching the pre-existing pass-2 rust
   cycle count for the same kernel. That validates the analytical scorer +
   rust promotion path without any LLM in the loop. Once this works, the
   only remaining variable is LLM quality.

## Pass-1 → autotune2 prerequisites checklist

For a checkpoint to be usable by `run_autotune2.py`:

- [x] `<outer_dir>/plan/iteration_<MAX>/tree.json` exists (planner output).
- [x] `<outer_dir>/pass1/iteration_<MAX>/<node_path>/.../extracted_code.py`
      with `status.txt` containing `PASS` (one per node).
- [x] **`<outer_dir>/pass1/iteration_<MAX>/<node_path>/contract.pkl`** —
      added by the orchestrator's contract dump in this session. Old
      checkpoints (pre-dump) fail loudly here; re-run pass-1 to populate.
- [x] `<outer_dir>/../config.json` (the parent kernel-grouping dir's
      run config — provides `dims`).
- [x] `--autotune-config` JSON with `hw_config` block (same JSON
      `run.py --autotune-config` consumes).
- [x] `<NextStep>/configs/<model>.json` for the LLM profile (default
      `gpt-oss-120b`; pass `--config` for an explicit path).

## Code Style Reminders (from CLAUDE.md)

- **NO try/except** unless absolutely necessary. Use assertions.
- Keep code additions minimal. Avoid unnecessary control flow.
- Don't mindlessly agree — challenge user assumptions when warranted.
- Codex will review the code; write to senior-engineer standard.
- Keep an eye on portability: assume other modules can change; don't
  hard-couple to current pass-1 internals where reasonable.

## Existing Infrastructure to Integrate With

- [src/contract.py](src/contract.py) — `Contract` dataclass with
  `arg_is_raw: tuple[bool, ...]`. RAW classification source of truth.
- [src/blackbox_stub.py](src/blackbox_stub.py) — existing pass-1 stub
  generator (pure reshape, no permute). Variant stubs extend it with
  permute on both sides.
- [src/orchestrator.py](src/orchestrator.py) — pass-1 verification +
  child stub injection. Key sites:
  - [2166-2240](src/orchestrator.py#L2166): pass-1 per-node verification
  - [2517-2559](src/orchestrator.py#L2517): parallel child dispatch
  - [2575-2630](src/orchestrator.py#L2575): contract harvest + **pickle
    dump for autotune2** (this session)
  - [2686-ish](src/orchestrator.py#L2686): pass-2 compose success
  - [1293](src/orchestrator.py#L1293): `_gate_correctness`
  - [1377](src/orchestrator.py#L1377): `_gate_compliance`
  - [1434](src/orchestrator.py#L1434): `_gate_judge`
  - [1469](src/orchestrator.py#L1469): `_gate_post_validator`
  - [3026](src/orchestrator.py#L3026): `_load_tree_from_dir` (reused by
    autotune2 loader)
  - [3085](src/orchestrator.py#L3085): `_load_verified_dsls` (reused)
- [src/agents.py:294](src/agents.py#L294) — `make_client` (now tolerant
  of missing `api_key`).
- [src/agents.py](src/agents.py) — `make_autotune2_agent(llm_config,
  system_prompt)` factory added this session.
- [src/planner.py:39](src/planner.py#L39) — `Tree.iter_topological()`
  (post-order), `Tree.find(path)`.
- [src/config_loader.py:9](src/config_loader.py#L9) — `load_llm_config`
  (canonical profile loader; reused by `run_autotune2`).
- [/workspace/NextStep/StepDB/evaluate.py](../StepDB/evaluate.py) — rust
  simulator entry. `evaluate_kernel(kernel, preset, work_dir, timing_only)`
  returns `EvalResult(cycles, duration_ms, ...)`. Cost: **minutes to hours
  per call** — top-K only.
- [/workspace/NextStep/step_tl/src/timing_and_emulator/timing.py:165](../step_tl/src/timing_and_emulator/timing.py#L165) —
  `analyze_timing(graph, hw_config)`. Cost ~10-100ms per call.
- [src/autotune.py](src/autotune.py) — legacy flat autotuner. **Do not
  modify; comparison baseline.**

## Target Kernel

[/workspace/checkpoints/2026-05-14-174755/prefill_transformer_simple/outer_1](../../checkpoints/2026-05-14-174755/prefill_transformer_simple/outer_1)
is the freshest pass-1 output the user has been driving against. Plan tree:

- `tiled_reference` (root) → `pre_attention`, `attention_o_proj`, `moe`
- `moe` → `rms_norm`, `moe_dispatch__root_moe_moe_dispatch`
- (etc — see the saved `tree.json`)

`pre_attention` is a multi-output node (Q, K, V) — its successful scoring
under the synthetic wrapper was the most recent fix in this session.
