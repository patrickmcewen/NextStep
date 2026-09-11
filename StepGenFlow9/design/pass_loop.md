# Per-pass turn loop

Each LLM pass — `refactor_final`, `translate`, the `direct` variants —
runs a turn-based loop with a maximum turn budget. The loop is the same
shape across passes; what differs between passes is the *executor*, the
*compliance rules*, the *judge prompt*, and whether a *post-validator* is
attached. This document describes the shape and the contracts of each
gate.

## Per-turn shape

Per turn:

1. The accumulated conversation (system prompt + alternating user/assistant
   turns + appended feedback) is sent to the LLM.
2. The assistant's response is parsed for a Python code block
   (` ```python ... ``` `, bare ` ``` ... ```, or, as a fallback, a whole
   response that parses as valid Python).
3. The extracted code runs through up to four sequential gates:
   correctness → regex compliance → judge → post-validator.
4. If every gate passes, the turn succeeds and the loop exits.
5. Otherwise a feedback string is built from whichever gate failed and
   appended as the next user turn. The model retries on the next turn.

Per-turn artifacts (user prompt, response text, reasoning, extracted code,
correctness output, shape trace, judge response, status) are written to
disk; see [logging.md](logging.md).

If `max_turns` rounds elapse without success, the pass returns failure and
its outer attempt's pipeline halts. Other outer attempts continue
independently.

A response that yields no extractable code is treated as a recoverable
protocol error: the loop appends a "no code block" reminder to the
conversation and burns one turn rather than aborting.

## User-prompt assembly

The first turn's prompt carries the kernel context the model needs:

- the kernel name,
- the original PyTorch reference (verbatim from StepDB),
- the dims dict (JSON-rendered),
- a description of every entry in the precomputed `tensors` dict
  (per-key shape + dtype) plus the source of the precompute function
  (lifted from `StepDB/precompute.py` by AST search) so the model knows
  exactly how each tensor was built,
- the previous pass's verified output, when present (refactor passes
  receive PyTorch; translate passes receive the DSL form),
- the function signature the pass must produce (`tiled_reference` for
  refactor, `build_graph` for translate),
- a hard prohibition on creating new torch tensors (no `torch.randn`,
  no `torch.zeros`, no `@`).

Subsequent turns' prompts carry only the feedback string for the failure
that ended the prior turn; the conversation history retains the full
context.

## Executors

Correctness is run by one of two executor functions, selected per pass.
Each takes the extracted code and returns either a "match=True" string
or a structured failure description:

| executor | role |
|---|---|
| `dsl` | exec'd as `tiled_reference(dims, tensors)` with the DSL surface injected into the namespace. Validates a refactor pass — the DSL is directly runnable, so a refactor-pass output gets a real correctness signal before the translator ever runs. |
| `graph` | exec'd as `build_graph(dims, tensors)`, run through the STeP simulator. Validates a translate pass — the lowered IR graph is dispatched and its output compared to gold. |

Each executor compares the result tensor's shape against gold, then
computes max-absolute and relative error and accepts on `rel_err < 1e-5`.
On mismatch, the worst-error index is reported alongside the gold and
candidate values at that index. Gold is memoized per `(kernel, dims)` for
the process lifetime.

The `dsl` executor prepends a small import scaffold to the user code
(`torch`, `torch.nn.functional`, the DSL module, etc.) so the model never
needs to write `import` statements; the system prompts forbid imports
explicitly. The `graph` executor uses an analogous scaffold sourced from
StepDB so the same imports are in scope across the implementer flow,
StepDB validation, and the autotuner.

Because the DSL is directly runnable, the executor split lines up exactly
with the phase split: phase 1 (refactor) is gated by `dsl`, phase 2
(translate) is gated by `graph`. There is no executor mode that mixes
the two scopes or skips correctness; every LLM-emitted candidate is
exec'd against gold before the next gate runs. In bundle mode the
abstraction takes the place of the standalone DSL surface (see
[bundle_mode.md](bundle_mode.md)), and the same `dsl` executor runs
against it.

## Compliance check

After correctness passes, the candidate code runs through a regex compliance
check. Compliance is **lexical** — it does not parse the code, just searches
the function body for forbidden or required tokens. The check produces a
list of violation strings; an empty list means compliant.

Two backends, switched by mode:

- **Standalone mode.** Compliance rules are a hard-coded set of per-pass
  tables: an `allowed_torch` allowlist, an `allowed_F` allowlist, a
  `banned_patterns` list of `(substring, fix-hint)` pairs, and a
  `required_ops` list. Tables for the refactor passes accumulate
  cumulatively across the pass sequence — a `refactor_final` check
  enforces every prior refactor pass's rules in addition to its own.
  Translate passes share an analogous cumulative group.
- **Bundle mode.** A single check driven by the bundle manifest's
  `compliance` block: `allowed_ops`, `banned_patterns`, `required_ops`.
  Empty allowlist disables the allowlist branch. Banned-pattern entries
  carry a `fix` hint that gets shown back to the model.

For translate passes the check is scoped to the body of `build_graph`
only — scaffold helpers and DSL function definitions are expected to
contain `torch.*` calls and shouldn't be flagged.

When the check finds violations on otherwise-correct code, the next-turn
feedback explicitly says "your output is correct, but you used these
disallowed operations" — distinguishing compliance from correctness
matters because the model's repair strategy is different in each case.

## Judge

For passes that have a judge agent, a second LLM is run after the regex
check passes (and, for selected refactor passes, *also* on regex-rejected
turns to give richer line-specific feedback alongside the regex output).

The judge sees the candidate code plus a short context block:

- For correctness-passing turns: an explicit "this code's output already
  matches the reference" notice, so the judge knows to evaluate structure
  only and not "looks wrong mathematically" intuitions.
- For non-compliant turns where the judge runs anyway: the tensors-dict
  description, so the judge can flag tensor-shape laundering.

The judge returns either `VERDICT: PASS` or `VERDICT: REJECT` with a
`VIOLATIONS:` block. Ambiguous responses are treated as reject.

The judge prompts are templates. In standalone mode each pass has a
hand-tuned template baked into a prompt file. In bundle mode the template
is a parameterized one whose `{allowed_ops_block}` /
`{banned_patterns_block}` / `{required_ops_block}` placeholders are
filled from the bundle's compliance config — so the judge speaks the
abstraction's invented vocabulary rather than the hard-coded DSL one.

## Post-validator

A pass may have a post-validator: an arbitrary callable
`(code, turn_dir) -> str | None` that runs after correctness, regex, and
judge all pass. Returning `None` means the turn succeeds; returning a
string treats the turn as failed and uses that string as feedback for
the next turn.

The post-validator's primary role is the deterministic-translation gate
on `refactor_final`: under `--translator=auto`, the validator runs the
DSL-to-STeP translator on the candidate DSL and runs the resulting graph
on the simulator. If translation throws, or the graph throws, or the
graph's output doesn't match gold, the validator returns a feedback
string that distinguishes the three cases. This is how translator-side
constraints get fixed inside the refactor loop rather than failing later
in a separate pass.

In bundle mode the post-validator wraps the bundle's own
`transpiler.translate` callable, so the bundle's invented translation
rules become the gate.

## Feedback channels

Beyond the per-gate failure messages, the loop appends two diagnostic
channels to the next-turn prompt when relevant:

### Per-op shape trace

The DSL operators print one line per call describing input/output stream
and tile shapes (gated by an environment variable so the trace is only
captured in the orchestrator's stdout-redirect block). On any correctness
exception, the captured trace up to the failing op is appended to the
feedback under a "STeP DSL shape trace" block. The trace is tail-truncated
to bound context cost.

### Enhanced tracebacks

Two layers of error enhancement run before feedback is built:

- **User-code line mapping.** When a frame in the traceback is from
  `<string>` (the exec'd user code), the line number is mapped back to
  the user's code, three lines of surrounding context are quoted, and
  any tensor-shape locals at that frame are listed.
- **Emulator context.** When the simulator raises while dispatching a
  STeP node, the traceback is walked to recover which node was being
  dispatched, where it was constructed in the user code, and what the
  failing functional-py frame's locals looked like. Shape and stream
  metadata of the failing node are reported alongside the user-code
  match.

A handful of common error families also get follow-up hint blocks
appended:

- `ModuleNotFoundError` / `ImportError` triggers a "do not include
  `import` statements" reminder.
- `missing 1 required positional argument` triggers a "most STeP ops
  require `graph` as the first arg; source ops do not" reminder.
- `FlatPartition ... not subscriptable` triggers a "FlatPartition returns
  a single node; access branches with `(partitioned, i)` tuples" reminder.

These are deliberate scaffolding for failure modes the LLM repeatedly
fell into; they're cheap to maintain because the orchestrator has the
exception text in hand at feedback-build time.
