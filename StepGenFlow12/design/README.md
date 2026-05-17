# StepGenFlow design notes

Reference documentation for the inner kernel-refactoring flow that turns a
PyTorch reference kernel into a verified STeP graph. These describe **what
StepGenFlow is** as a system — its phases, contracts, and observable
behaviors — at a level abstract enough that an implementer can satisfy them
without prescription about specific code organization.

## What StepGenFlow does

For one `(kernel, preset)` pair, StepGenFlow produces a
`build_graph(dims, tensors)` function that defines a STeP IR graph whose
emulated output matches a PyTorch reference within floating-point tolerance.
The kernel name selects a PyTorch reference module from `StepDB`; the preset
selects a concrete `dims` dict; an external `precompute` step builds the
input `tensors` dict. The flow is LLM-driven on the parts that need code
synthesis (the decomposition planner, the refactor pass, optionally the
translation pass, and the optional autotuner) and deterministic on the
parts that don't (gold comparison, AST-level translation, regex compliance,
the timing model).

```
   PyTorch ref + dims + tensors          (per kernel × preset)
              │
              ▼
       ┌──────────────┐  Phase 0
       │  planner     │  LLM decomposes the kernel into a tree of
       │  (loop)      │  sub-Models; refactor walks post-order
       └──────┬───────┘  (skipped under --no-plan or bundle mode)
              │ tree
              ▼
       ┌──────────────┐  Phase 1
       │  refactor    │  LLM rewrites each tree node into DSL form,
       │  (loop)      │  parents using verified children as few-shot
       └──────┬───────┘  gated on tiled-DSL correctness vs gold
              │ dsl_code.py (root DSL)
              ▼
       ┌──────────────┐  Phase 2
       │  translate   │  DSL → STeP IR (deterministic AST rewrite,
       │              │  or LLM pass under --translator=llm)
       └──────┬───────┘  gated on emulator output vs gold
              │ build_graph
              ▼
        verified graph
              │
              ▼ (optional)
       ┌──────────────┐
       │  autotune    │  Bottom-up tree-DP over the planner tree:
       │  (tree DP)   │  per-node Pareto libraries on (cycles,
       └──────────────┘  on_chip); top-K promoted through rust
```

`max_outer` independent attempts at the implementer pipeline run in parallel
per kernel; the first to succeed wins. The autotuner is a **separate,
standalone post-pipeline pass** invoked via `run_autotune2.py` against a
finished implementer checkpoint; it walks the planner tree bottom-up,
builds a Pareto-front library of variant DSL implementations at each node,
and promotes the top-K root entries through the rust simulator. A
separate batch driver (`run_regression.py`) fans out across many
`(kernel, preset)` jobs as subprocesses with a parallelism cap.

## Two operating modes

- **Standalone mode.** The flow uses a hard-coded DSL surface
  (`step_dsl.py`), a deterministic DSL→STeP translator, a multi-pass
  refactor sequence with cumulative compliance tables keyed by pass name,
  and per-pass judge prompts.
- **Bundle mode.** The flow accepts an external bundle directory
  (`abstraction.py`, `transpiler.py`, `refactor_system.txt`,
  `manifest.json`). The bundle's abstraction module is mounted under the
  name `step_dsl`, the bundle's transpiler replaces the deterministic
  one, the bundle's system prompt replaces the refactor prompt, and the
  bundle's compliance config drives both the regex check and the judge
  template. The pipeline collapses to a single refactor pass plus the
  bundle's own deterministic translate. Bundle mode is what the
  AbstractionOpt outer flow invokes; see `flowv2/design/inner_flow_integration.md`
  for the bundle-side contract.

## Documentation index

- [pipeline.md](pipeline.md) — invocation, the standard / direct / bundle
  pipeline shapes, the three-phase structure, parallel outer iterations,
  the `--translator` switch, resume.
- [planner.md](planner.md) — Phase 0 decomposition: tree decomposition
  contract, planner LLM pass and guards, post-order walk with sibling
  parallelism, per-node refactor with relaxed gates for non-root,
  replanning, resume-planner.
- [pass_loop.md](pass_loop.md) — the LLM-driven per-pass turn loop:
  prompt assembly, executor types, correctness, compliance, judge,
  post-validator, gate ordering, feedback channels.
- [bundle_mode.md](bundle_mode.md) — what `--bundle-dir` swaps in, how
  the abstraction is mounted, how the judge is templated, how the
  pipeline collapses.
- [regression_runner.md](regression_runner.md) — multi-kernel batch
  driver: subset selection, subprocess model, parallelism cap, summary.
- [autotuner.md](autotuner.md) — performance-tuning subsystem: bottom-up
  tree-DP over the planner tree, per-node Pareto-front libraries keyed
  by `(input_contracts, output_contracts)`, synthetic isolation
  wrapper for non-root scoring, top-K rust promotion, run-summary
  schema.
- [logging.md](logging.md) — the on-disk checkpoint tree, per-turn
  artifacts, `result.json`, regression summary.
