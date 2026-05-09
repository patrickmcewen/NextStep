# Pipeline

The implementer pipeline is the per-kernel driver: it owns CLI invocation,
mode selection, the phase ordering, parallel outer iterations, and resume.
The per-pass LLM loop is described separately in [pass_loop.md](pass_loop.md);
this document is everything that wraps it.

## Invocation

A single CLI entry point runs one `(kernel, preset)` job. User-facing
arguments:

| flag | role |
|---|---|
| `kernel` (positional) | kernel name from StepDB's `bench_config.yaml` |
| `preset` (positional) | preset name selecting a concrete `dims` dict |
| `--model` / `--config` | LLM profile name (loads `configs/<name>.json`) or explicit JSON path |
| `--max-outer` | number of independent outer attempts to run in parallel |
| `--max-turns` | cap on turns per LLM pass loop |
| `--results-dir` | top-level directory for run logs |
| `--experience-dir` | directory for successful implementations (legacy / external use) |
| `--checkpoint-dir` | override the default timestamped checkpoint root |
| `--pipeline` | `standard` (lowering + translate), `direct`, or `direct_no_functional` |
| `--translator` | `auto` (deterministic AST rewrite, default) or `llm` |
| `--resume` | path to a saved DSL checkpoint; skips lowering |
| `--resume-planner` | path to a crashed outer's directory; resumes a planner walk |
| `--few-shot` | example program paths shown to the refactor agent |
| `--bundle-dir` | bundle directory; activates bundle mode |
| `--check-order` | `correctness-first` (default) or `compliance-first` — see [pass_loop.md](pass_loop.md) |
| `--no-plan` | disable Phase 0 (decomposition planner); fall back to single-shot refactor |
| `--max-replans` | global re-plan budget on Phase 1 failure |
| `--node-attempts` | per-tree-node parallel refactor attempts |
| `--max-plan-depth` | maximum recursion depth of the planner tree |
| `--non-root-sequential` / `--no-non-root-sequential` | with `--node-attempts > 1`, run non-root attempts sequentially with early-exit (default) or in parallel |
| `--stateless-refactor` | discard refactor chat history; rebuild the prompt each turn from (orig + last failed code + last feedback) |
| `--autotune` | run the autotuner on each outer's verified DSL (off by default) |
| `--autotune-config` | path to `autotune_config.json` (loaded only when `--autotune` is set) |
| `--autotune-max-turns` | override `max_turns` from the autotune config — only used when the config has no `passes` list |
| `--autotune-agent` | autotune agent variant — only used when the config has no `passes` list |

LLM configuration (provider URL, API key, model id, optional reasoning
effort) is loaded from JSON profiles under a repo-root `configs/` directory
keyed by model name.

`--resume` and `--resume-planner` are mutually exclusive: `--resume` skips
lowering entirely (a verified `dsl_code.py` already exists);
`--resume-planner` re-runs lowering using a saved decomposition tree plus
any per-node DSLs that were verified before the crash. See
[planner.md](planner.md) for the resume-planner contract.

## Pipelines

The flow can be configured into one of three named pipeline shapes plus
bundle mode:

| pipeline | phase 1 (lowering) | phase 2 (translation) |
|---|---|---|
| `standard` | LLM `refactor_final` pass — PyTorch → DSL form, optionally driven by the planner | deterministic AST rewrite (`auto`) **or** LLM `translate` pass (`llm`) |
| `direct` | (none) | LLM `translate_full` — PyTorch → STeP graph in one pass |
| `direct_no_functional` | (none) | LLM `translate_full_no_functional` — variant prompt that excludes the `functional.py` reference |
| bundle | LLM `refactor_final` with bundle-supplied prompt | bundle's own deterministic transpiler |

`standard` + `auto` is the canonical configuration: the refactor pass is the
only LLM call on the lowering side, and translation is a pure AST rewrite.
The `direct` shapes exist as ablations that skip the DSL intermediate.
Bundle mode is structurally the same as `standard` + `auto`, with the
bundle replacing the hard-coded DSL surface, the refactor prompt, and the
translator.

Phase-2 translate passes have their own per-pass LLM prompt name — the
canonical case is `translate`; the `direct` shapes use `translate_full`
and `translate_full_no_functional` (different system prompts, different
compliance tables).

## Three-phase structure

For pipelines that have a lowering phase, the flow runs **phase 0 → phase
1 → phase 2** in order, each gated separately.

1. **Phase 0 — decomposition planner** (`standard` only, on by default).
   The planner LLM decomposes the kernel into a tree of sub-Models. The
   planner's per-node guard checks (`compose-equivalent`,
   `anti-passthrough`, etc.) and the tree walk are documented in
   [planner.md](planner.md). With `--no-plan`, this phase collapses to a
   single-leaf tree equivalent to the legacy single-shot path.
2. **Phase 1 — lowering.** For each tree node in post-order, the LLM is
   given the node's PyTorch reference, the dims, the precomputed tensor
   descriptions, any verified child DSLs as few-shot context, and a
   system prompt that describes the DSL surface. It must emit a
   `tiled_reference(dims, tensors)` (or, for non-root nodes, the node's
   own `Model.forward` lifted to DSL — possibly tuple-returning) that
   calls only DSL operators. Each turn is gated on **tiled-DSL
   correctness** — the candidate code is exec'd against the precomputed
   tensors and its output compared to gold (see [pass_loop.md](pass_loop.md)
   for the executors). On success the verified DSL source is persisted;
   the root node's DSL is the run's `dsl_code.py`.
3. **Phase 2 — translation.** Under `--translator=auto` the DSL source
   is fed through a deterministic AST translator that emits a
   `build_graph(dims, tensors)` returning `(graph, output_op)`; the graph
   is executed on the STeP simulator and compared to gold. Under
   `--translator=llm` a separate LLM `translate` pass produces the same
   shape, gated on simulator correctness. Either way, phase 2 is the
   gate that decides whether the outer attempt succeeded.

When phase 1 already produces something that translates and runs cleanly,
the deterministic translator is invisible to the LLM. When it doesn't, the
translator's failure is surfaced back to the refactor pass via the
post-validator (see [pass_loop.md](pass_loop.md)) — the LLM keeps fixing the
DSL until it lowers cleanly. This is what makes `--translator=auto` viable:
translator-side constraints get enforced inside the refactor loop instead of
in a separate downstream pass.

## Parallel outer iterations

A single invocation runs `max_outer` independent attempts at the pipeline
in parallel under `asyncio.gather`. Each attempt has its own checkpoint
subdirectory (`outer_<i>/`) and its own log file. The outcomes are
independent: a crash in one attempt does not affect the others (exceptions
are normalized into failure-result records), and the call returns the
**first successful attempt** if any exists, otherwise the last failure.

Outer attempts share precomputed gold tensors (memoized per `(kernel, dims)`
to avoid re-allocating multi-GiB references) but otherwise have no shared
state.

Under `--autotune`, each outer that produces a verified `dsl_code.py`
runs the autotuner inline before its coroutine returns. Because each
outer is its own task, an outer's autotune runs concurrently with
whatever the other outers are still doing on the functional pipeline.
The autotuner writes under `outer_<i>/autotune/`, and an autotune
crash is trapped at the boundary so the outer's functional success is
preserved. The autotune step itself can be a chain of passes (see
[autotuner.md](autotuner.md) for the chain schema and per-outer schema).

## Resume

Two resume modes serve different recovery scenarios:

- **`--resume`** loads a verified `dsl_code.py` and skips Phase 0 + Phase 1
  entirely. The resumed run starts at Phase 2. Use this when iterating on
  the translator without re-paying for the refactor pass, or when feeding
  a previously-verified DSL form into a new translate run. Accepts a
  `dsl_code.py` file, an `outer_<N>/` directory containing one, or a
  checkpoint root, in which case the flow searches for
  `<kernel_name>/outer_*/dsl_code.py`.
- **`--resume-planner`** loads a crashed outer's saved tree plus any
  per-node DSLs that were verified before the crash, and re-runs only the
  non-verified nodes. Use this when an outer crashed mid-Phase-1 and the
  invested per-node refactor work is worth recovering. See
  [planner.md](planner.md) for the resume contract and on-disk layout.

Both modes are stateless re-entries: the resumed process writes a new
checkpoint directory; the prior one is left untouched. There is no
analogue of an outer-flow `state.json` here — each StepGenFlow invocation
is treated as a fresh job.

When `--resume` is set, every outer attempt loads the same DSL code,
skips phases 0 and 1, and starts at phase 2.

## Few-shot examples

The lowering prompt supports an optional list of resolved example pairs —
each a `(PyTorch reference, DSL form)` lifted from a previously successful
checkpoint of *another* kernel. Examples are rendered into the
`refactor_final` system prompt under `{few_shot_examples}` and are intended
for kernel-family priming (e.g., showing one transformer kernel's lowering
to seed another's). Resolution rules are the same as `--resume`: a
`dsl_code.py`, an `outer_<N>/` directory, or a checkpoint root. The kernel
name is recovered from the resolved checkpoint's `config.json` so the
matching PyTorch reference can be loaded from StepDB.

Few-shot examples are distinct from the planner's own
"verified sub-task DSLs" block, which is rendered automatically into a
parent node's user prompt from its just-verified children.

## Run-time invariants

- The kernel and preset must exist in StepDB's `bench_config.yaml` before
  the run starts; assertion failure here aborts loud.
- `precompute_tensors(kernel, dims)` is run once per outer-iteration group
  and produces the `tensors` dict that flows through every executor and
  every prompt. The flow does not let the LLM construct its own tensors —
  the system prompts forbid `torch.randn`, `torch.zeros`, etc., and the
  judge enforces this lexically.
- Gold tensors are computed once per `(kernel, dims)` and memoized for the
  process lifetime. Gold computation is deterministic (fixed seeds inside
  the reference).
- `--translator=auto` is incompatible with the `direct` pipeline shapes
  (the AST translator consumes DSL output, which the direct shapes never
  produce). Bundle mode forces `auto`.
- Phase 0 (the planner) is incompatible with `--bundle-dir`, with
  `--pipeline != standard`, and with `--resume`. See [planner.md](planner.md)
  for the rationale.
