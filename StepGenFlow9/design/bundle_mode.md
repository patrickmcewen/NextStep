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
- the refactor pass uses the `passthrough` correctness executor — see
  below for why;
- the standard step-DSL judge agent is replaced with a bundle-templated
  one.

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

## Pipeline collapse

Bundle mode runs exactly one LLM pass (`refactor_final`) followed by one
deterministic translate. Several things collapse:

- **No phase ordering.** There is no separate translate pass; the
  bundle's `transpiler.translate` is invoked directly after the refactor
  pass succeeds.
- **The post-validator is the gate.** Because the executor for the
  refactor pass is `passthrough` (it always reports `match=True`), the
  *only* gate that proves the candidate is correct is the post-validator
  — which runs the bundle's transpiler, runs the resulting graph on the
  simulator, and compares against gold. Why passthrough rather than
  `dsl`? The DSL surface is the bundle's invention; the orchestrator
  cannot know how to call into it from outside the bundle's prompt
  contract, so it does not try. The bundle author owns DSL-level
  semantics; the orchestrator owns IR-level ground truth.
- **No cumulative-table compliance.** Compliance is one allowlist plus
  one banned-pattern list plus one required-ops list, exactly as the
  manifest declares.

> Note. In a future iteration this could be split into two real gates —
> a DSL-level executor that runs the abstraction directly to get a
> correctness signal independent of the transpiler, and an IR-level
> executor that catches transpiler bugs separately. The flowv2
> outer-flow design documents this as the v1→v2 difference; the inner
> flow currently exposes only the IR-level gate. Bundles whose
> abstractions are directly runnable would benefit from the split.

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

- the correctness block always reports PASS for the refactor pass
  (passthrough), so it does not contribute information;
- the post-validator's translator-failure / graph-execution-failure /
  graph-mismatch branches are the only place an actual semantic
  failure can be signaled;
- the shape trace only appears if the bundle's abstraction emits the
  expected trace prints (the standalone DSL does; a bundle's
  abstraction need not).

Bundle authors who want richer correctness-side feedback should make
their abstraction directly runnable and have the orchestrator invoke it
via a non-passthrough executor. That extension is straightforward but
not currently wired.
