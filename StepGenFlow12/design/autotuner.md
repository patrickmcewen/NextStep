# Autotuner

The autotuner takes a kernel whose implementer pipeline (Pass 0 / 1 / 2)
has already completed — i.e., a planner-decomposed tree of verified-correct
per-node DSL functions — and searches for performance-superior variants.
The search is **bottom-up tree DP** over the planner tree: each node
acquires a *library* of `(input_contracts, output_contracts)`-indexed
Pareto-front entries, parents compose their own entries by enumerating
the Cartesian product of their children's libraries, and the root's
library is the final search output. The top-K root entries on the
analytical Pareto front are re-scored through the rust simulator to
catch analytical-model errors.

Correctness is enforced at the root only — a non-root node's outputs
are free to take any layout the LLM proposes, because functional
equivalence is recovered when the parent composes the chosen child
variant with its own DSL and the root verifies the full chain against
`compute_gold(kernel, dims, tensors)`. Per-node verification is
reduced to a graph-build smoke test plus a DSL eager-exec smoke test
so the LLM gets actionable feedback before contract clashes leak into
the analytical scorer.

## How autotune2 is invoked

Autotune2 is a **separate, standalone CLI** — it does not run inline
inside `run.py`. The entry point is `run_autotune2.py`, which takes
an existing `outer_<N>/` checkpoint as its positional argument:

```
python run_autotune2.py /path/to/checkpoints/<ts>/<kernel>/outer_N \
    [--autotune-config autotune_configs.yaml] \
    [--autotune-config-name autotune_config_2] \
    [--model gpt-oss-120b] [--config <llm_config.json>] \
    [--max-attempts 5] [--max-turns-per-attempt 16] \
    [--check-order correctness-first] \
    [--top-k 3] [--compute-bw 100000] \
    [--checkpoint-dir <base>] [--include-sources]
```

The chosen `outer_<N>/` must contain:

- `pass1/iteration_<MAX>/tree.json` — the planner-decided tree;
- `pass1/iteration_<MAX>/<node_path>/contract.pkl` per non-root node —
  the recorded `Contract` written by Pass 1 (the post-Pass-1 contract
  dump);
- the per-node verified DSL artifacts that
  `orchestrator._load_verified_dsls` can recover;
- an outer-level `<ts>/config.json` two directories up, used for
  `dims`, and (when `--kernel` / `--preset` are not passed on the
  CLI) for the StepDB `kernel` / `preset` names.

`--autotune-config` points at a JSON or YAML file containing one required
key `hw_config` (HBM channels, channel latency, PMU buffer sizes,
`max_compute_bw`, `max_par_dispatch`, etc. — the same shape the
implementer's timing model consumes). The default path is
`autotune_configs.yaml` at the repo root, with
`--autotune-config-name autotune_config_2`.

### Checkpoint isolation

Before any work begins the source checkpoint is copied into a fresh
timestamped directory under `--checkpoint-dir` (default: the parent
of the source `<ts>/` directory). The chosen `outer_<N>/` is
preserved; sibling `outer_*` directories are excluded from the copy
so the snapshot stays small. All autotune2 reads and writes happen
against the copy; the original outer is never modified. Behavior
mirrors `run.py --resume-after-pass1`.

## What gets searched

Each plan-tree node is searched independently, with the constraint
that parents wait for their children's libraries before composing.
The unit of search is a **`DesignEntry`**:

```
DesignEntry:
  dsl:               str                       # standalone function source for this node
  input_contracts:   {arg_name -> TensorContract}    # boundary layout of each on-chip TensorArg
  output_contracts:  {out_<i>  -> TensorContract}    # boundary layout of each output
  cycles:            int                       # analytical timing-model cycle count
  on_chip:           int                       # analytical on-chip-memory bytes
  provenance:        str                       # "pass1_baseline" or "llm_attempt_<i>_turn_<j>"
  children_picks:    {child_path -> DesignEntry}     # for parents: chosen child entry per child
```

`children_picks` is the back-pointer that lets a root `DesignEntry`
reconstruct the entire composed source by walking
`gather_descendants_postorder` — direct object references mean
Pareto culling at a child level cannot orphan a parent's
reproducibility chain (Python GC keeps culled-but-referenced entries
alive).

### TensorContract

A `TensorContract` is the declarative description of a boundary
layout — a `(reshape, permutation)` pair applied to the vanilla
PyTorch shape:

```python
@dataclass(frozen=True)
class TensorContract:
    reshape:     tuple[int, ...]    # factorization of the vanilla shape (numel must match)
    permutation: tuple[int, ...]    # permutation of range(len(reshape))
```

Two equivalent ways to read a contract:

- mechanically: `x_in_contract = x_vanilla.reshape(reshape).permute(*permutation)`;
- declaratively: "the LLM is laying out this tensor as the
  post-permute shape, and the parent must hand it that layout."

The **identity contract** is `reshape == vanilla_shape` and identity
permutation; it produces a tensor identical to vanilla. Contracts
apply only to on-chip `TensorArg`s — RAW args (loaded fresh inside
the leaf via `offchip_load`) and list args carry no contract.

`reshape` is unconstrained beyond element-count preservation: rank
may change, dims may be split or merged. The wrapper-build path
(`_offchip_load_args_for_contract`) imposes one additional
realizability constraint: the contract's permutation must keep the
last two reshape axes in place (`permutation[-2:] == (n-2, n-1)`)
because the synthetic `offchip_load` wrapper tiles the last two dims
and walks the leading stream prefix; permutations that pull stream
dims into the tile aren't expressible as a single strided
`offchip_load`. This is asserted at wrapper-build time so the LLM
gets DSL-level feedback ("pick contracts that keep the tile =
vanilla[-2:]") rather than a deep IR mismatch later.

### NodeLibrary

A node's library is a two-level nested dict:

```
NodeLibrary = {
    input_contracts_key  -> {
        output_contracts_key -> [DesignEntry, DesignEntry, ...]   # Pareto front
    }
}
```

Cells are keyed by the **frozen** form of the contracts dict —
`freeze_contracts({arg: contract, ...})` sorts by arg name so two
dicts with the same contents produce the same key. Each cell holds
a Pareto front of `DesignEntry`s on `(cycles, on_chip)`:
non-dominated insertion + dominated-member culling happens in place
via `insert_pareto`.

The pass-1 baseline is always seeded with identity contracts on
every on-chip arg and output. This is the rollback guarantee: even
if every LLM turn fails, the library still contains the pass-1
design that already verified correct against the kernel reference,
so the root's `pick_top_k_pareto_entries` never returns empty.

## Per-node search

Three entry points in `src/autotune2/search.py`:

- `search_leaf(node, ...)` — populate one leaf node's library.
- `search_parent(node, ...)` — populate one parent node's library,
  composing its DSL with each child's library entries.
- `autotune(plan_tree, ...)` — bottom-up driver that walks the tree
  post-order and dispatches each node's search.

### Synthetic `tiled_reference` wrapper

Non-root nodes are scored in **isolation**: the analytical timing
model is fed a self-contained `tiled_reference(dims, tensors)` so
critical-path analysis sees realistic input streams without
threading the whole kernel through.

`build_synthetic_wrapper_for_node` constructs this wrapper for a
single variant. For each on-chip `TensorArg`, it emits:

```python
_Q_in = offchip_load(tensors["Q"], stride=..., out_shape_tiled=..., tile_row=..., tile_col=...)
_Q_in = flatten(_Q_in, min_rank=..., max_rank=...)
```

with `(stride, out_shape_tiled, tile_row, tile_col)` derived from
the variant's `input_contracts[name]` and the vanilla shape (see
`_offchip_load_args_for_contract` for the closed-form derivation).
The `flatten` step removes `offchip_load`'s leading singleton so
the stream rank matches the contract's `post_permute_shape()[:-2]`.

RAW `TensorArg`s pass through as `tensors[name]` directly — the
leaf is responsible for loading those itself, and `tensors` arrives
wrapped as `StepRawTensor` per `_wrap_input_tensors` so the bare
reference satisfies every DSL source op's `_assert_raw` gate.

Because strides are contract-dependent, the wrapper is **rebuilt
per variant**: identity contracts for the baseline seed, and
`parsed.input_contracts` for each LLM response. The wrapper
terminates with `offchip_store(promote_outer(result))` so the
translator's root-return-wrapping logic has a stream to wrap and
OffChipStore's stream-rank≥1 startup constraint is satisfied. For
multi-output nodes every output is stored, with the last one
returned.

The root node is not wrapped: its DSL is already
`tiled_reference(dims, tensors)` and consumes the kernel's input
tensors directly.

### Per-node tensors

`build_node_tensors_dict(parent_contract)` builds the `tensors`
dict the wrapper's `offchip_load` references resolve against. For
each `TensorArg`, the dict carries the **recorded
`parent_contract.tiled_values[i]`** reshaped to the vanilla
`spec.shape`. Using zero tensors would break for leaves whose
functional executor inspects values, not just shapes — e.g.
`moe_dispatch` reads `expert_onehot` to compute per-expert active
token counts and reshapes the resulting ragged buffer accordingly;
an all-zero one-hot collapses every expert's bucket and
`FlatReassemble` panics on the invalid reshape.

For list args (`ListOfTensorArg`, `ListOfIntArg`) the recorded
`tiled_values` entry passes through unchanged.

The root falls back to the kernel-level `root_tensors` dict
(precomputed via `precompute_tensors(kernel, dims)`).

## Per-turn loop

Each autotune2 pass runs under one shared `time_limit_seconds`
deadline. Per-node attempts keep proposing parse-and-verify turns while
the pass deadline allows new turns. When ACE context refresh is enabled,
an attempt is split into `session_<N>/turn_<M>` windows; the shared
curator refreshes the playbook between windows and the lane starts a new
multi-turn conversation with the refreshed context.

### Prompt assembly

The system prompt (`prompts/autotune/tile_shrink/autotune2_system.txt`
or `prompts/autotune/parallel/autotune2_system_parallel.txt`) is built
per node from a template with three placeholders:

- `{step_dsl_code}` — the literal contents of `src/step_dsl.py`,
  the DSL surface reference;
- `{memory_notes}` — `prompts/autotune/shared/dsl_memory_notes.txt` with
  `{step_dsl_memory_code}` substituted to the contents of
  `src/step_dsl_memory.py` (the on-chip / off-chip memory shim);
- `{output_protocol}` — one of
  `prompts/autotune/shared/autotune2_output_protocol_{leaf,parent}.txt`,
  selected by `node.is_leaf`.

The system prompt is baked into the agent at construction (per
node, via `agent_factory(system_prompt)`); the user prompt is
rebuilt per attempt to carry the running Pareto-front summary so
the LLM targets gaps.

The user prompt (`build_autotune2_user_prompt`) carries:

- the node name, function signature, and PyTorch reference;
- the dims block and tensors description;
- the pass-1 verified DSL as a starting point;
- for parent nodes: a per-child variant table
  (`render_variant_block`) showing each child's arg classification
  (RAW vs on-chip), per-variant Pareto coordinates, and
  per-variant input/output contracts in human-readable
  `vanilla.reshape(...).permute(...)` form;
- for second-and-later attempts: a `render_accepted_summary` block
  of already-admitted entries with their Pareto coordinates and
  contracts so the LLM can target gaps.

### Output protocol

The LLM emits two fenced blocks: a YAML block declaring the
boundary contracts (and, for parents, the per-child variant pick)
and a python block carrying the DSL function body.

```yaml
child_picks:                            # parent prompts only
  attention_block: 3
parent_input_contracts:                 # empty/None when no input is on-chip
  Q: {reshape: [8, 8, 64], permutation: [1, 0, 2]}
parent_output_contracts:
  out_0: {reshape: [16, 4, 512], permutation: [1, 0, 2]}
```

```python
def attention_block(Q, K, V, *, out_shapes):
    ...
```

`parse_autotune2_response` splits the response into an
`AutotuneResponse(child_picks, input_contracts, output_contracts,
dsl)`. Schema violations raise `AssertionError` with the offending
fragment in the message; the search driver routes the assertion
text back into the conversation as feedback.

### Turn-level gate cascade

Per turn, in `correctness-first` order (the default; see
`SearchConfig.check_order` for the other two modes):

1. **Parse** the response. Failure → "no parseable YAML/python
   block" feedback, burn one turn.
2. **Build the synthetic wrapper** with the variant's
   `input_contracts`. Failure (a non-realizable permutation, a
   tile that doesn't divide the vanilla shape) → "contracts must
   keep tile = vanilla[-2:]" feedback.
3. **Verify** the composed source via the per-node verifier (see
   below). Failure → the verifier's feedback string back to the
   LLM.
4. **Score** via `_safe_score`. The analytical scorer runs
   `translate → _exec_build_graph → analyze_timing` (with
   `_rescale_compute_bw` normalizing the per-op compute_bw to the
   global budget). Any exception inside `analyze_timing` is
   converted to a feedback string (the smoke tests upstream catch
   most of these, so this is a last-resort net).
5. **Admit** the entry into the library via `insert_pareto`. A
   successful turn breaks out of the inner loop to start a fresh
   attempt.

For parent nodes, step 5 has an extra fan-out: the parent's DSL
is composed with **every Cartesian combination** of the picked
children's Pareto-front cells (`_iter_parent_compositions`), each
combination is verified and scored independently, and every
non-dominated composition is admitted. The first composed source
and last verify result are logged to the per-turn artifact dir;
fanning out every combo into a per-turn file would explode the
checkpoint.

### Per-node verifier

`build_real_verifier_factory_fn` returns a per-node verifier
factory. The verifier is constructed once per node by the bottom-up
driver and reused across attempts.

**Root verifier** — full 4-gate cascade against
`compute_gold(root_kernel, dims, root_tensors)`:

- **correctness** — `_gate_correctness` runs the composed source as
  `tiled_reference(dims, tensors)` and compares to gold (shape-relaxed,
  `rel_err < 1e-5`);
- **compliance** — `_gate_compliance` with `is_root=True`,
  `extra_required_ops=()`, scoping to the root function body;
- **judge** — `_gate_judge` (when a `judge_agent` is configured) for
  canonical-form review;
- **post_validator** — `_make_translation_post_validator` runs the
  deterministic translator + STeP simulator + gold compare; skipped
  when correctness has not yet verified.

`check_order` swaps gate ordering identically to pass-1's pass
loop.

**Non-root verifier** — *no gold compare*. The LLM is free to
declare new `parent_input_contracts` / `parent_output_contracts`
per node, so the node-local notion of "correct output" decouples
from pass-1's recorded `tiled_outputs`. Numerical correctness is
recovered at composition time. Instead the verifier runs:

- **DSL eager-exec smoke test** (`_dsl_exec_smoke_test`) — invokes
  `_exec_dsl_ref(composed_source, dims, tensors)` and catches
  torch-level runtime errors inside DSL ops (shape mismatches in
  `torch.stack` / `torch.cat`, dtype clashes, asserts inside DSL
  primitives). Failures here would otherwise propagate into the
  timing model's functional executor (`execute_values →
  _exec_flat_reassemble` etc.) and surface as opaque scorer
  crashes; catching them at the DSL surface gives a tighter
  signal.
- **Graph-build smoke test** (`_graph_build_smoke_test`) —
  `dsl_to_step.translate` + `_exec_build_graph(translated, dims,
  tensors)`. Catches STeP frontend assertions like
  `stride × out_shape_tiled exceeds buffer grid` that fire when
  the LLM-proposed input contract clashes with the leaf body's
  internal bufferize / streamify expectations.
- **compliance** + **judge** are scoped to the LLM-emitted
  `def <node_name>` block extracted from the composed source via
  `_extract_node_def_block`. The autotuner-generated wrapper has
  its own `offchip_load` / `offchip_store` calls that would trip
  `is_root=False` compliance rules; only the LLM's code is
  reviewed.

Both smoke tests render their tracebacks into actionable
`_GateResult.feedback`. A DSL-exec failure short-circuits the
graph-build check (no point translating a broken DSL). Variants
that pass the smoke tests are admitted to the Pareto library on
`(cycles, on_chip)` alone.

The non-root post_validator slot is intentionally a no-op — the
root's verifier handles the kernel-level translate-and-execute
check.

### Status vocabulary

Per-turn `status.txt`:

| status | meaning |
|---|---|
| `PARSE_FAIL: <msg>` | YAML/python block parse failed |
| `WRAPPER_BUILD_FAIL: <msg>` | contracts not realizable as a strided offchip_load |
| `BAD_CHILD_PICK: <msg>` | parent's `child_picks` referenced an unknown variant index |
| `VERIFY_FAIL` | the per-node verifier (smoke tests or root 4-gate) rejected the composed source |
| `SCORE_FAIL` | `_safe_score` caught a timing-model crash; routed back as feedback |
| `VERIFY_OR_SCORE_FAIL_ALL_COMPOSITIONS` | parent variant rejected under every Cartesian combination |
| `ACCEPTED` | leaf variant admitted to its cell |
| `ACCEPTED (N entries)` | parent variant admitted N non-dominated entries across compositions |

## Bottom-up driver

`autotune(plan_tree, ...)` schedules one `asyncio.Task` per node.
Each task awaits its children's tasks before starting work, so the
walk is **post-order with sibling parallelism**: all leaves start
concurrently, a parent fires the instant its subtree is done.
`asyncio.gather` over the full task list fans the first failure
out as a cancellation — fail-fast is the right default because a
child failure means the parent can't compose anyway.

Per node, the driver builds:

- the per-node `system_prompt` (from the `system_prompts` dict);
- a per-node `AgentFn` via `agent_factory(system_prompt)` — each
  node gets its own conversation history and reasoning trail;
- the per-node `tensors` dict (root → `root_tensors`, non-root →
  `build_node_tensors_dict(parent_contract)`);
- the per-node `ScoreFn` via `make_score_fn(node_tensors)` —
  closures over the kernel `dims`, `hw_config`, and a global
  `max_total_compute_bw` budget;
- the per-node `VerifierFn` via `make_verifier(node,
  parent_contract, node_tensors)`.

Parent baseline picks come from the Pareto-best of each child's
**first** library cell (which by `_seed_baseline`'s construction is
the identity-contract pass-1 baseline). This ensures the parent's
pass-1 baseline composes against its children's pass-1 baselines,
matching the pre-search-walk state.

## Cycle / memory scoring

`make_analytical_scorer(dims, tensors, hw_config,
max_total_compute_bw)` closes over the kernel context and returns
a one-arg `score(composed_source) -> (cycles, on_chip)`:

1. `dsl_to_step.translate(composed_source)` → `build_graph` source.
2. `_exec_build_graph(translated, dims, tensors)` → live STeP graph.
3. `_rescale_compute_bw(graph, max_total_compute_bw)` — every
   compute op's `compute_bw` is rescaled so the sum equals the
   budget (floor at 1, matching the operator-level minimum in
   `ops.py`). The default budget is 100k cycles, the memory-bound
   regime: per-op BW distribution is vacuous, only the total
   matters.
4. `analyze_timing(graph, hw_config=hw_config)` → `result` dict
   with `total_cycles` and `per_node` info.
5. Sum on-chip bytes across all nodes via
   `node.on_chip_requirement(count_fifos=False)`, with sympy
   `sym_subs` applied when present and free symbols replaced with
   1.

`_safe_score` wraps the scorer call in a try/except and converts
any exception to a feedback string. Belt-and-suspenders behind the
non-root verifier's DSL-exec + graph-build smoke tests, which
catch most timing-model crashes upstream where the LLM feedback
points at the DSL it wrote.

## Top-K rust promotion

After the search completes, the root's library is the analytical
search output. `pick_top_k_pareto_entries(root_library, k)` flattens
every cell into a single list, drops dominated entries, sorts by
`(cycles, on_chip)`, and keeps the top `K`. The K=3 default is set
because the rust simulator is expensive enough that more rarely
pays off.

For each pick, `promote_top_k` reconstructs the full composed
source by walking `gather_descendants_postorder(entry)` (which
follows `children_picks` recursively to the leaves), invokes the
injected `rust_evaluate_fn`, and collects
`RustPromotionResult(entry, rust_cycles, rust_dur_ms,
composed_source)`. Results are returned sorted by `rust_cycles`.

The rust evaluator (`build_rust_evaluate_fn`) writes the composed
source to `<ckpt_dir>/autotune2/_rust_work/step_impl.py` and calls
`StepDB/evaluate.py::evaluate_kernel(kernel_name, preset,
work_dir, timing_only=True, step_impl_source=...)`. A rust
evaluator failure on a promoted entry is a hard assertion — there
is no skip-and-continue path. If the analytical scorer admitted an
entry but the rust simulator rejects it, that's a real signal worth
surfacing.

The autotuner is **non-degrading on the analytical metric** by
construction: the pass-1 baseline is always entry zero in every
node's library and `pick_top_k_pareto_entries` walks the full
Pareto front, so the chosen entries never regress on
`(cycles, on_chip)` versus pass-1. The rust pass can re-order
within the top-K if the analytical model is locally inaccurate;
the analytical-best entry is not necessarily the rust-best.

## Run summary

`write_autotune2_summary(autotune_result, rust_promotions, out_path,
include_sources=False)` emits `autotune2_summary.json` at
`<ckpt_dir>/autotune2_summary.json`:

```json
{
  "root_path":          "attention_o_proj/attention",
  "library_sizes":      {"<node_path>": <num_cells>, ...},
  "root_pareto":        [{"cycles": ..., "on_chip": ..., "provenance": ...}, ...],
  "rust_winners":       [
    {"analytical_cycles": ..., "analytical_on_chip": ...,
     "rust_cycles": ..., "rust_dur_ms": ..., "provenance": ...},
    ...
  ],
  "best_rust_entry":    {<same shape>},
  "best_composed_source": "..."         // only when include_sources=True
}
```

The summary is the canonical handoff to whatever downstream
consumer (a regression-suite scorer, a human, a follow-up
optimizer) wants to know what the autotune2 run produced.
`include_sources=True` embeds the rust-best composed source for
reproducibility at the cost of potentially-megabyte JSON.

## Variants registry

Each node's per-turn checkpoint also emits a declarative
`variants.py` registry at `<ckpt_dir>/autotune2/<node_path>/variants.py`:

```python
from src.autotune2.contracts import TensorContract


variant_registry = {
    0: {
        "input_contracts":  {"Q": TensorContract(reshape=(...), permutation=(...))},
        "output_contracts": {"out_0": TensorContract(reshape=(...), permutation=(...))},
    },
    1: { ... },
    ...
}
```

Variant indices are assigned sequentially per node, one per unique
`(input_contracts, output_contracts)` cell — entries within a cell
share an index because they represent the same boundary contract.
The variant *callable* `<child_name>_<index>` is built at runtime
by `make_variant_stub(ref_module, arg_specs, input_contracts,
output_contracts, ...)`. The stub wraps the child's PyTorch
reference (`nn.Module`) with the input/output adapter pair derived
from the contracts; the LLM only ever sees declarative metadata.

The autotuner overwrites `variants.py` each time a node's library
gains an admitted entry. The registry shape is deliberately
declarative (not per-variant Python function bodies) because the
adapter logic is identical across variants — only contracts differ
— so duplicating bodies into each run dir would multiply bug
surface for no gain.

## Why correctness is enforced at the root only

The implementer pipeline's Pass-1 walk verifies each non-root
node's DSL against its own gold (rebuilt from the node's
`reference_code` per `build_node_tensors`). The autotuner inherits
nothing of that local-gold check — it allows non-root nodes to
declare new `parent_*_contracts` precisely so the LLM can explore
boundary-layout reshufflings that don't preserve any pass-1
intermediate's element layout.

Recovering correctness at the root works because:

- the synthetic wrapper for the *root* doesn't exist — the root
  composes naturally with all descendants via
  `gather_descendants_postorder`;
- the root verifier's `compute_gold` check runs the full kernel
  graph end-to-end, with the LLM-picked variants substituted in.
  If any non-root variant breaks the algorithm under composition,
  the root's correctness gate fails.

This is the **rollback** mechanism that makes per-node speculative
contracts safe: a parent that picks a child variant which doesn't
compose cleanly will fail its own gate cascade (DSL-exec smoke
test, graph build, or — at the root — gold compare) and the parent
falls back to a Cartesian combination it can verify. The pass-1
baseline cell of every child is always available, so the parent
always has at least one combination that works.

## Per-node checkpoint layout

Under `<ckpt_dir>/autotune2/<node_path>/`:

```
system_prompt.txt           # the rendered autotune2 system prompt for this node
pass1_baseline_score.json   # the baseline (cycles, on_chip, provenance) — first thing written
variants.py                 # declarative registry, overwritten on each admission
attempt_<i>/turn_<j>/
    user_prompt.txt
    response.txt
    reasoning.txt           # when the agent returned reasoning summaries
    tokens.json             # when the agent returned a usage object
    extracted_code.py       # the LLM's DSL block (when parse succeeded)
    composed_source.py      # wrapper + leaf + descendants (when verifier was reached)
    verify_result.txt       # "PASS" or the full gate-feedback string
    score.json              # admitted entries' (cycles, on_chip, provenance)
    status.txt              # see status vocabulary above
```

The pass-1 baseline score is written immediately after the baseline
seeds the library so the reference point is recoverable from disk
even when no LLM attempt admits anything.

For parent nodes the `composed_source.py` artifact carries the
*first* Cartesian combination's composed source — its parent DSL
content is identical across combinations; only descendant DSLs
differ. Logging every combination would explode the per-turn dir.

The driver also writes `<ckpt_dir>/autotune2_summary.json` at the
top level when promotion finishes.

## Configuration knobs

`run_autotune2.py` flags:

| flag | default | role |
|---|---|---|
| `--kernel` / `--preset` | from source `<ts>/config.json` | StepDB names |
| `--autotune-config` | `autotune_configs.yaml` | path to JSON/YAML with `hw_config` |
| `--autotune-config-name` | `autotune_config_2` | YAML config name to resolve |
| `--model` / `--config` | `gpt-oss-120b` | LLM profile or explicit config path |
| `--checkpoint-dir` | parent of source `<ts>/` | base for snapshot copy |
| `--max-turns-per-attempt` | 16 | turns within one fresh conversation |
| `--max-attempts` | 5 | fresh-conversation attempts per node |
| `--check-order` | `correctness-first` | gate cascade order; same vocabulary as pass-1 |
| `--top-k` | 3 | root entries promoted to rust |
| `--compute-bw` | 100000 | global `max_total_compute_bw` budget |
| `--include-sources` | false | embed best composed source in summary JSON |

`max_attempts=0` skips the LLM loop entirely and ships only the
pass-1 baselines through promotion (useful for debugging the rust
hop).

## Why bottom-up tree DP

The implementer pipeline's pass-1 walk is **top-down**: parents
refactor first using blackbox stubs for children, then children
refactor under the parent's call-site contract. The autotuner
inverts this for performance: children commit to a Pareto-front
library first, parents enumerate the discrete Cartesian product
of their children's Pareto fronts, and the search space at each
level is the cross-product of locally-good choices below.

This works because:

- the search space at a single LLM call is small (one DSL function
  + its declared contracts) — the LLM can reason about it directly;
- the global combinatorial space (every leaf's variants ×
  cross-products at every parent) is enumerated mechanically, not
  by the LLM;
- contracts are the **declarative interface** that lets a child
  vary its internal DSL freely as long as the boundary layout
  matches what some parent wants to consume.

Compared to a flat "rewrite the whole kernel" autotuner, the
bottom-up walk localises each LLM call to one node-sized
search space and gives the LLM a fresh Pareto-front prompt at each
parent — gaps in the front are immediately visible from the
rendered variant table.

## Module map

| module | role |
|---|---|
| `src/autotune2/contracts.py` | `TensorContract`, `DesignEntry`, `NodeLibrary`, library cell lookup, `vanilla_contract_for`, `validate_reshape` |
| `src/autotune2/pareto.py` | `dominates`, `insert_pareto`, `cull_top_T`, `dsl_dedup_hash` |
| `src/autotune2/compose.py` | `compose_source`, `cartesian_compose`, `compose_into_library`, `make_analytical_scorer`, `_rescale_compute_bw`, `_sum_on_chip_bytes` |
| `src/autotune2/stubs.py` | `make_variant_stub`, `invert_input_contract`, `apply_output_contract`, `emit_variants_module`, `load_variants_module` |
| `src/autotune2/prompts.py` | `build_autotune2_system_prompt`, `build_autotune2_user_prompt`, `render_variant_block`, `render_accepted_summary`, `parse_autotune2_response` |
| `src/autotune2/search.py` | `search_leaf`, `search_parent`, `autotune`, the synthetic wrapper builder, the bottom-up `asyncio` driver, per-turn artifact writer |
| `src/autotune2/runtime.py` | top-K promotion, rust evaluator factory, real agent / verifier factories, `_dsl_exec_smoke_test`, `_graph_build_smoke_test`, summary writer |
| `run_autotune2.py` | CLI: checkpoint snapshot, state loading from pass-1 artifacts, wiring everything together |
