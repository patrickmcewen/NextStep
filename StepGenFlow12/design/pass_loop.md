# Per-pass turn loop

Each LLM pass — `refactor_final`, `translate`, the `direct` variants —
runs a turn-based loop with a maximum turn budget. The loop is the same
shape across passes; what differs between passes is the *executor*, the
*compliance rules*, the *judge prompt*, and whether a *post-validator*
is attached. This document describes the shape and the contracts of
each gate.

## Per-turn shape

Per turn:

1. The accumulated conversation (system prompt + alternating user/assistant
   turns + appended feedback) is sent to the LLM. Under
   `--stateless-refactor`, the conversation is collapsed each turn to
   `(original user prompt + last failed code + last feedback)` — see
   below.
2. The assistant's response is parsed for a Python code block
   (` ```python ... ``` `, bare ` ``` ... ```, or, as a fallback, a whole
   response that parses as valid Python).
3. The extracted code runs through up to four sequential gates:
   correctness, regex compliance, judge, post-validator (gate ordering
   is configurable — see "Gate ordering" below).
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

A model-side `LLM_BAD_REQUEST` (provider returned a structured error such
as a context-length overflow) is also recorded as a single turn's
status and burns the turn rather than aborting; the loop continues with
the same conversation.

## User-prompt assembly

The first turn's prompt carries the kernel context the model needs:

- the kernel name,
- the original PyTorch reference (verbatim from StepDB, or the planner
  node's `reference_code` for non-root nodes),
- the dims dict (JSON-rendered),
- a description of every entry in the precomputed `tensors` dict
  (per-key shape + dtype) plus the source of the precompute function
  (lifted from `StepDB/precompute.py` by AST search) so the model knows
  exactly how each tensor was built,
- the previous pass's verified output, when present (refactor passes
  receive PyTorch; translate passes receive the DSL form),
- for planner non-root nodes, the parent-declared contract block and
  child blackbox signatures (Pass-1 prompt extensions — see below),
- the function signature the pass must produce (`tiled_reference` for
  refactor, `build_graph` for translate),
- a hard prohibition on creating new torch tensors (no `torch.randn`,
  no `torch.zeros`, no `@`).

Subsequent turns' prompts carry only the feedback string for the failure
that ended the prior turn; the conversation history retains the full
context.

### Stateless mode

Under `--stateless-refactor`, the per-turn conversation is rebuilt each
turn as `(original user prompt + latest failed extracted_code +
latest feedback string)` rather than being grown by appending. The
prompt-cache prefix stays stable across turns and per-turn context is
capped at O(1). This is a refactor-pass-only mode; translate passes
keep growing their conversation regardless. Off by default —
accumulating chat history is the existing behavior.

## Pass-1 prompt extensions

When the orchestrator runs Pass 1 (pre-order refactor with blackbox children),
the per-node user prompt carries two additions beyond the baseline assembly
described above:

**Contract block.** For every non-root node, the prompt includes the
parent-declared contract: the actual tiled input tensor shapes and values
that the parent passed at the child's call site, and the `out_shape` /
`out_perm` the parent requested back. This is the concrete tiled interface
the child must satisfy — not abstract planner-level shapes.

**Child-blackbox signature block.** For every non-leaf node, the prompt
lists the auto-generated blackbox stubs available to call, one entry per
planner child:

```
<child_name>(*tiled_args, out_shapes, out_perms=None) -> Tensor | tuple
```

The model is told these names resolve to callable subroutines whose
semantics match the corresponding PyTorch reference; it chooses the
input tiling and the requested output shape per call site.

**Reshape-only rule.** Any tensor flowing into a blackbox call must be
transformed by pure reshape only (no `permute`, `transpose`, or slicing
before the call). Compliance enforces this lexically as an extension of
the existing `banned_patterns` mechanism; the `required_ops` list is
computed per-node from the planner tree's child names rather than from a
hard-coded table.

## Pass-2 deterministic gate

Pass 2 runs once after all Pass-1 nodes are verified. It is post-order,
deterministic, and invokes no LLM.

**Name rebinding.** For each non-leaf node (deepest first, then root),
the orchestrator builds a namespace where each blackbox name is bound to
the verified child DSL function instead of the auto-generated stub.
Because the verified child DSL has exactly the same call shape as the stub
(positional tiled args followed by keyword-only `out_shapes`/`out_perms=None`),
call sites resolve without any AST modification to the parent's source.

**Re-execution against gold.** The parent's frozen Pass-1 source is
exec'd with the rebound namespace against kernel-level gold
(`rel_err < 1e-5`). No candidate code is generated; the parent's source
is never rewritten.

**Scope in v1.** Only the root is verified at Pass 2. The root's
re-execution subsumes all intermediate compositions; per-level
verification is a future refinement.

**Failure.** If the root's Pass-2 executor fails (typically due to
numerical drift from DSL-level precision differences between the stub
and the real child DSL), the entire outer attempt fails and escalates to
the existing replan loop. There is no LLM repair at Pass 2.

## Executors

Correctness is run by one of two executor functions, selected per pass.
Each takes the extracted code and returns either a "match=True" string
or a structured failure description:

| executor | role |
|---|---|
| `dsl` | exec'd as `tiled_reference(dims, tensors)` with the DSL surface injected into the namespace. Validates a refactor pass — the DSL is directly runnable, so a refactor-pass output gets a real correctness signal before the translator ever runs. |
| `graph` | exec'd as `build_graph(dims, tensors)`, run through the STeP simulator. Validates a translate pass — the lowered IR graph is dispatched and its output compared to gold. |

Each executor compares the result against gold:

- Comparison is **always shape-relaxed**: both tensors are flattened in
  row-major order and compared element-wise with `rel_err < 1e-5`,
  requiring only that `numel` matches. Layout-changing bugs (transposes,
  scrambles) still fail because they place different element values at
  the same flat positions; unit-dim differences (e.g., vanilla 2D gold
  vs the equivalent stream-shaped output) pass since they share flat
  memory order. This is the natural correctness check for a flow whose
  outputs are tile-streams of rank ≥ 3 even when the reference's vanilla
  output is lower rank.
- The executor accepts a tuple/list result alongside a tuple/list gold
  (per-output comparison) for planner non-root nodes whose forward
  returns multiple tensors.

On mismatch, the worst-error flat index is reported alongside the gold
and candidate values at that index. Gold is memoized per `(kernel, dims)`
for the process lifetime.

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

A regex compliance check runs the candidate code through a per-pass
table of allowed / banned / required tokens. Compliance is **lexical**
— it does not parse the code, just searches the function body for
forbidden or required tokens. The check produces a list of violation
strings; an empty list means compliant.

Two backends, switched by mode:

- **Standalone mode.** Compliance rules are a hard-coded set of per-pass
  tables: an `allowed_torch` allowlist, an `allowed_F` allowlist, a
  `banned_patterns` list of `(substring, fix-hint)` pairs, and a
  `required_ops` list. Each pass's table is independent — there is no
  cumulative inheritance across passes.
- **Bundle mode.** A single check driven by the bundle manifest's
  `compliance` block: `allowed_ops`, `banned_patterns`, `required_ops`.
  Empty allowlist disables the allowlist branch. Banned-pattern entries
  carry a `fix` hint that gets shown back to the model.

For translate passes the check is scoped to the body of `build_graph`
only — scaffold helpers and DSL function definitions are expected to
contain `torch.*` calls and shouldn't be flagged.

For planner non-root refactor passes (`is_root=False`), the rules
relax: sink-op requirements (`offchip_store`) drop, since non-root
DSLs don't terminate at off-chip — they hand a stream up to their
parent. The root, in contrast, must always include `offchip_store`
even when its body is a pure blackbox-orchestrator: the kernel's
external output is written off-chip exactly once, at the root.

When the check finds violations on otherwise-correct code, the next-turn
feedback explicitly says "your output is correct, but you used these
disallowed operations" — distinguishing compliance from correctness
matters because the model's repair strategy is different in each case.

## Judge

For passes that have a judge agent, a second LLM is run after the regex
check passes (and, for the refactor pass, *also* on regex-rejected
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

For planner non-root nodes, the post-validator is plumbed with
`is_root=False` and runs the same translate-and-execute cycle, but the
output contract is relaxed alongside the executor's relaxation.

In bundle mode the post-validator wraps the bundle's own
`transpiler.translate` callable, so the bundle's invented translation
rules become the gate.

## Gate ordering

`--check-order` chooses between two orderings:

- **`correctness-first`** (default): `correctness → regex → judge →
  post-validator`. The candidate code is exec'd first; only correct
  proposals pay the structural-review cost.
- **`compliance-first`**: `regex → judge → correctness → post-validator`.
  Structural rejection happens before exec, so a proposal that's
  obviously non-canonical is rejected without paying for execution.
  Useful when correctness is expensive (large tiles) and the model is
  repeatedly emitting structurally-broken code.

The two orderings change which `status.txt` value lands when an early
gate rejects. Under `correctness-first`, the noncompliant /
judge-rejected statuses imply correctness already passed; under
`compliance-first`, they don't.

## Status vocabulary

`status.txt` carries one of:

| value | meaning |
|---|---|
| `PASS` | every gate passed; pass loop exited successfully |
| `NO_CODE_EXTRACTED` | response had no parseable code block |
| `LLM_BAD_REQUEST: <exc>` | provider returned a structured error (e.g., context-length overflow); turn burned, loop continues |
| `FAIL: <error head>` | correctness check rejected the candidate |
| `CORRECT_BUT_NONCOMPLIANT` | correctness PASS, regex compliance found violations |
| `NONCOMPLIANT` | regex compliance found violations under `compliance-first` (correctness not yet checked) |
| `CORRECT_BUT_JUDGE_REJECTED` | correctness + regex PASS, judge said REJECT |
| `JUDGE_REJECTED` | judge rejected under `compliance-first` (correctness not yet checked) |
| `CORRECT_BUT_POST_VALIDATOR_REJECTED` | correctness + regex + judge PASS, post-validator rejected |

The `CORRECT_BUT_*` and bare-name variants exist as a pair so the same
gate's verdict can be distinguished by whether correctness has already
been verified at the time it fired.

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
