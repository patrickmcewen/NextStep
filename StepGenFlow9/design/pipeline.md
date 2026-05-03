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
| `--pipeline` | `standard` (lowering + translate), `direct`, or `direct_no_functional` |
| `--translator` | `auto` (deterministic AST rewrite, default) or `llm` |
| `--resume` | path to a saved DSL checkpoint; skips lowering |
| `--few-shot` | example program paths shown to the refactor agent |
| `--bundle-dir` | bundle directory; activates bundle mode |
| `--checkpoint-dir` | override the default timestamped checkpoint root |
| `--autotune` | run the autotuner on each outer's verified `build_graph` (off by default) |
| `--autotune-config` | path to `autotune_config.json` (loaded only when `--autotune` is set) |
| `--autotune-max-turns` | override `max_turns` from the autotune config |
| `--autotune-agent` | autotune agent variant (`general` or `parallel`) |

LLM configuration (provider URL, API key, model id, optional reasoning
effort) is loaded from JSON profiles under a repo-root `configs/` directory
keyed by model name.

## Pipelines

The flow can be configured into one of three named pipeline shapes plus
bundle mode:

| pipeline | phase 1 (lowering) | phase 2 (translation) |
|---|---|---|
| `standard` | LLM `refactor_final` pass — PyTorch → DSL form | deterministic AST rewrite (`auto`) **or** LLM `translate` pass (`llm`) |
| `direct` | (none) | LLM `translate_full` — PyTorch → STeP graph in one pass |
| `direct_no_functional` | (none) | LLM `translate_full_no_functional` — variant prompt that excludes the `functional.py` reference |
| bundle | LLM `refactor_final` with bundle-supplied prompt | bundle's own deterministic transpiler |

`standard` + `auto` is the canonical configuration: the refactor pass is the
only LLM call, and translation is a pure AST rewrite. The `direct` shapes
exist as ablations that skip the DSL intermediate. Bundle mode is structurally
the same as `standard` + `auto`, with the bundle replacing the hard-coded
DSL surface, the refactor prompt, and the translator.

## Two-phase structure

For pipelines that have a lowering phase, the flow runs **phase 1 to
verified DSL form** before starting phase 2.

1. **Phase 1 — lowering.** The LLM is given the PyTorch reference, the
   dims, the precomputed tensor descriptions, and a system prompt that
   describes the DSL surface. It must emit a `tiled_reference(dims, tensors)`
   that calls only DSL operators. Each turn is gated on **tiled-DSL
   correctness** — the candidate code is exec'd against the precomputed
   tensors and its output compared to gold (see [pass_loop.md](pass_loop.md)
   for the executors). On success the verified DSL source is persisted as
   the run's `dsl_code.py`.
2. **Phase 2 — translation.** Under `--translator=auto` the DSL source
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

Under `--autotune`, each outer that produces a verified `build_graph`
runs the autotuner inline before its coroutine returns. Because each
outer is its own task, an outer's autotune runs concurrently with
whatever the other outers are still doing on the functional pipeline.
The autotuner writes under `outer_<i>/autotune/`, and an autotune
crash is trapped at the boundary so the outer's functional success is
preserved. See [autotuner.md](autotuner.md) for the per-outer schema.

## Resume

A run can be resumed from a saved DSL checkpoint to skip the lowering phase.
The `--resume` argument accepts:

- a path to a `dsl_code.py` file directly,
- a path to an `outer_<N>/` directory containing `dsl_code.py`,
- a path to a checkpoint root, in which case the flow searches for
  `<kernel_name>/outer_*/dsl_code.py`.

When resuming, every outer attempt loads the same DSL code, skips phase 1,
and starts at phase 2. This is for two cases: iterating on the translator
without re-paying for the refactor pass, and feeding a previously-verified
DSL form into a new LLM translate run.

Resume is a stateless re-entry: the resumed process writes a new checkpoint
directory; the prior one is left untouched and is the source from which
the DSL code was lifted. There is no analogue of the outer-flow
`state.json` here — each StepGenFlow invocation is treated as a fresh job.

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
