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

Performance autotuning runs as a **separate post-pipeline pass**
(`run_autotune2.py`); the implementer pipeline does not invoke it
inline. See [autotuner.md](autotuner.md).

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
2. **Phase 1 — lowering.** Split into two sub-passes:

   ```
   Phase 1: Pass 1 (pre-order, LLM)  →  Pass 2 (post-order, deterministic, no LLM)
   ```

   **Pass 1** walks root → leaves. Each non-leaf node refactors with its
   children represented as auto-generated blackbox stubs; the parent
   declares the call-site contract (input shapes, output shape/permutation)
   that each child must satisfy. The LLM is given the node's PyTorch
   reference, a parent-declared contract block, and the blackbox signatures
   for its children. Each turn is gated on tiled-DSL correctness — the
   candidate code is exec'd with the stubs providing child outputs. On
   success the verified DSL and the captured child contracts are persisted.
   Sibling subtrees fan out in parallel once their parent's Pass 1 succeeds.

   **Pass 2** walks leaves → root deterministically. For each non-leaf
   node, each blackbox name is rebound to the verified child DSL function;
   the parent's frozen Pass-1 source is re-executed against kernel-level
   gold. No LLM is invoked. In v1 only the root is verified at Pass 2
   (the root subsumes intermediate compositions); per-level verification
   is a future refinement. Failure escalates to the existing replan loop.

   `--no-plan` collapses Phase 1 to a single root refactor with no children
   (degenerate Pass 1, no Pass 2), behaving identically to the legacy
   single-shot path. Bundle mode is pinned to `--no-plan` and is unaffected
   by the two-pass structure.

   On success the verified root DSL is persisted as the run's `dsl_code.py`.
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

Performance autotuning is **out of band**: the implementer pipeline is
done as soon as some outer produces a verified `dsl_code.py`. The
autotuner ([autotuner.md](autotuner.md)) is invoked separately via
`run_autotune2.py` against a chosen `outer_<N>/` checkpoint and writes
under its own snapshot of that outer.

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
  per-node DSLs verified before the crash, and re-runs only the
  non-verified nodes. A resume can land in mid-Pass-1 (some nodes
  complete, some not), at the boundary "Pass 1 done, Pass 2 not started",
  or mid-Pass-2. Use this when an outer crashed during Phase 1 and the
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

Few-shot examples are distinct from the Pass-1 prompt's contract block
and child-blackbox signatures, which are generated per-node from the
planner tree and injected automatically.

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
