# Bundle mode

Bundle mode is the integration surface used by the AbstractionOpt outer
flow. It replaces the hard-coded DSL surface, refactor system prompt,
deterministic translator, and compliance vocabulary with artifacts
supplied by an external **bundle directory**, leaving the per-pass turn
loop and the executor machinery unchanged.

The bundle-side contract — what a bundle promises and what its on-disk
layout looks like — is documented in `flowv2/design/bundle_contract.md`
and `flowv2/design/inner_flow_integration.md`. This document is the
*inner* side of the same contract: what StepGenFlow does when
`--bundle-dir` is set.

## Activation

Bundle mode activates when `--bundle-dir <path>` is passed. The path
must point at a directory containing the four bundle artifacts; this is
asserted up front. Bundle mode forces:

- the pipeline shape collapses to a single `refactor_final` pass plus
  one deterministic translate;
- the translator selection collapses to `auto` (the bundle's own
  transpiler is the only translator that knows the abstraction's
  vocabulary);
- the refactor pass uses the `dsl` correctness executor against the
  bundle's abstraction (which the bundle is required to make directly
  runnable; see below);
- the standard step-DSL judge agent is replaced with a bundle-templated
  one;
- Phase 0 (the decomposition planner) is asserted disabled. The
  planner's `compose-equivalent` guard and few-shot composition logic
  assume the standalone DSL surface; mixing them with a bundle's
  invented vocabulary would silently break either the guard or the
  refactor prompt. Bundle runs must be invoked with `--no-plan`.

Standalone-mode features that bundle mode keeps unchanged: parallel
outer iterations, the per-turn loop and feedback channels, resume from a
saved DSL checkpoint, few-shot examples, and the on-disk checkpoint
layout. Bundle mode is structurally a special configuration of the same
pipeline, not a separate driver.

## What the bundle replaces

| component | standalone source | bundle source |
|---|---|---|
| DSL surface available to the refactor pass | `src/step_dsl.py` | `<bundle>/abstraction.py`, mounted as `step_dsl` |
| refactor system prompt | `prompts/refactor_final_system.txt` | `<bundle>/refactor_system.txt` |
| DSL→STeP translator | `src/dsl_to_step.translate` | `<bundle>/transpiler.translate` |
| compliance vocabulary | hard-coded per-pass tables | `<bundle>/manifest.json["compliance"]` |
| refactor judge prompt | `prompts/refactor_final_judge_system.txt` | `prompts/bundle_refactor_judge_system.txt` templated from compliance |

The user-prompt assembly, executor wiring, judge dispatch, and
post-validator plumbing are all unchanged.

## Mounting the abstraction

The bundle's `abstraction.py` is loaded under the import name `step_dsl`
so that LLM-authored kernel code that says `import step_dsl` (the name
the bundle's own system prompt is expected to use) resolves to *this*
bundle's surface. Loading is a fresh import per invocation:

- if a `step_dsl` module is already cached in `sys.modules` from a prior
  bundle, it is evicted before this one is loaded;
- the bundle's directory is prepended to `sys.path` so the bundle's
  `transpiler` (which may itself `import step_dsl`) resolves against the
  same module object;
- if a `transpiler` module is already cached from a prior bundle, it is
  evicted before this one is loaded.

This eviction discipline matters because the outer flow can run multiple
bundles back-to-back inside a single Python process — a stale module
would silently smuggle the previous bundle's vocabulary into the next.

The mounted abstraction must be **directly runnable**: the `dsl`
executor calls `tiled_reference(dims, tensors)` against it on every turn
to get a real correctness signal independent of the bundle's transpiler.
A bundle whose abstraction is only a vocabulary stub is not supported by
this flow.

## Pipeline collapse

Bundle mode runs exactly one LLM pass (`refactor_final`) followed by one
deterministic translate. Several things collapse:

- **No phase ordering.** There is no separate translate pass; the
  bundle's `transpiler.translate` is invoked directly after the refactor
  pass succeeds.
- **Two real gates, in the standard places.** The `dsl` executor runs
  the bundle's abstraction directly and compares against gold — this
  catches abstraction-level bugs independent of the transpiler. The
  post-validator runs the bundle's `transpiler.translate`, dispatches
  the resulting graph on the simulator, and compares against gold — this
  catches transpiler bugs separately. Splitting these two responsibilities
  cleanly is what makes the bundle author's surface (the abstraction) and
  the orchestrator's ground truth (IR semantics) testable in isolation.
- **Compliance is bundle-driven.** Compliance is one allowlist plus one
  banned-pattern list plus one required-ops list, exactly as the manifest
  declares — there are no per-pass tables to inherit, since bundle mode
  runs only one pass.

## Compliance config

The compliance block in `manifest.json` has a fixed schema:

```
{
  "allowed_ops":     [string, ...],
  "banned_patterns": [{"pattern": string, "fix": string}, ...],
  "required_ops":    [string, ...]
}
```

Each field drives both the regex check and the judge:

- **`allowed_ops`** — every `torch.X(...)` and `F.X(...)` call in the
  function body whose suffix is not in this list is flagged. Empty list
  disables the allowlist branch (any callable is permitted).
- **`banned_patterns`** — each `pattern` substring is searched for; if
  present, a violation line is emitted that quotes the paired `fix`
  hint back to the model.
- **`required_ops`** — each name must appear textually in the function
  body. Useful for forcing the inclusion of source/sink ops the
  abstraction defines (e.g., the bundle's analogue of `offchip_load`
  / `offchip_store`). Empty list disables the requirement.

The same three fields are rendered into the judge's system prompt as
bullet lists at template-fill time, so the judge enforces the same
vocabulary as the regex check without any ad-hoc per-bundle prompt
authoring.

A compliance config is a hard-fail input: a missing `compliance` key, a
missing field, or a malformed entry aborts the run loud rather than
falling back to a default.

## Per-turn feedback in bundle mode

The per-turn feedback channels (correctness, regex, judge,
post-validator, shape trace, enhanced tracebacks) all behave the same
as in standalone mode, with the substitutions listed above. In
particular:

- the correctness block reports a real PASS / FAIL from running the
  bundle's abstraction against gold, so a refactor that breaks the
  algorithm fails fast at the abstraction layer rather than waiting for
  the transpiler;
- the post-validator's translator-failure / graph-execution-failure /
  graph-mismatch branches signal failures that survived the abstraction
  gate — i.e., bugs in the bundle's transpiler or in the LLM's use of
  vocabulary that the abstraction tolerates but the IR rejects;
- the shape trace only appears if the bundle's abstraction emits the
  expected trace prints (the standalone DSL does; a bundle's
  abstraction need not, but is encouraged to).
