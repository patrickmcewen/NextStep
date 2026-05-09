# Decomposition planner

Phase 0 of the implementer pipeline. Before Phase 1 runs, the
planner decomposes a kernel into a tree of sub-Models that Phase 1's
two-pass refactor then processes. The planner is on by default;
`--no-plan` falls back to the single-shot legacy behavior of running
one `refactor_final` pass on the whole kernel.

The planner exists because the refactor pass scales badly on large
kernels — by the time a transformer-block-sized kernel is rewritten in
DSL, the model has burned its turn budget on local syntactic fixups
without ever reaching the structural rewrite. Decomposing first lets
the refactor pass work at a granularity the LLM can keep in its head.

## What the planner produces

The planner LLM is given the PyTorch reference for a node and is told
to either (a) declare the node a **leaf** — small enough to lower in
one refactor pass — or (b) **split** it into two-or-more child
sub-Models plus a smaller "refactored parent" whose `forward` calls
its children. The split is recursive: a child can itself be split on
the next planner turn. Recursion is bounded by `--max-plan-depth`;
nodes at depth ≥ cap are forced LEAF without an LLM call.

Each tree node carries:

- a `name` and a `path` (slash-delimited, e.g. `root/qkv_proj`),
- a `reference_code` snapshot — the PyTorch the LLM was asked about for
  this node (the original module for the root; a child's `class Model`
  for a sub-node),
- a `refactored_code` snapshot for internal nodes — the parent's
  `forward` rewritten to call its children's `<CamelCase>Model` classes,
- a `children` list (empty for leaves).

The root node always inherits the kernel's externally-observable
signature; children are free to return tuples (e.g. `Q, K, V`) since
they're verified for numerical match against their own reference module
only and are never independently translated.

## Planner LLM pass

The planner agent's response must be one of:

- `DECISION: leaf` — accepted unconditionally; this node will be
  refactored in one shot.
- `DECISION: split` followed by one or more `# child: <snake_name>`
  blocks (each a self-contained `class Model` + `get_inputs(dims)`)
  and a `# refactored parent` block (a smaller `class Model` that
  imports / instantiates the children).

A split is rejected unless it passes five mechanical guards:

| guard | what it checks |
|---|---|
| anti-passthrough | every child's `forward` contains at least one torch op (no trivial wrappers) |
| anti-monolith | refactored parent's `forward` AST node count is strictly smaller than the original's (the split actually decomposed something) |
| no-dead-children | every child is actually called from the parent's `forward` |
| children-runnable | each child's reference module exec's and `Model()(*get_inputs(dims))` runs |
| compose-equivalent | parent stitched against children's `Model` classes reproduces the original's output to `rel_err < 1e-5` |

A guard failure is fed back to the planner LLM as a structured rejection;
the planner has a fixed retry budget (per-node, not per-turn) and on
exhaustion the node falls back to LEAF rather than failing the whole
tree, on the reasoning that a leaf will get its own retry shot in the
refactor pass with proper context.

Function-based references (kernels whose StepDB reference is a
`compute_gold(dims, tensors)` function rather than a `class Model`)
can only sit at the root: the planner detects them and refuses to
split them. LLM-emitted children are always class-based.

## Tree walk

Once the tree is fixed, Phase 1 runs in two passes (see
[pipeline.md](pipeline.md)). **Pass 1** walks root → leaves (pre-order):
the parent refactors first using auto-generated blackbox stubs for its
children, then each child refactors under the contract the parent's
stub captured. Sibling subtrees fan out under `asyncio.gather` once
their parent's Pass 1 completes. **Pass 2** walks leaves → root
(post-order, deterministic): each blackbox name is rebound to the
verified child DSL and the parent is re-executed against gold.

At planner-output time the orchestrator extracts per-node
intermediate-input shapes by running a hook-based forward dry-run on
each internal node's `refactored_code` via `node_signature.extract_signature`.
These shapes are used to auto-generate the child blackbox stubs that
Pass 1 presents to the parent LLM.

`--node-attempts N` controls fan-out *per node*: each node spawns up to
N parallel refactor-loop attempts (each a full `_run_pass_loop` with
its own `--max-turns` budget) and the first success wins. Nodes that
the planner found easy (typical of leaves) usually succeed on the
first attempt, so spawning N up front wastes tokens.
`--non-root-sequential` (default true) handles this: non-root nodes try
attempts one at a time with early exit, while the root always
parallelizes since its failure is the most expensive failure to
retry. `--no-non-root-sequential` forces all nodes to fan out.

The result of Phase 1 is a single root DSL string — the verified
`tiled_reference(dims, tensors)` for the whole kernel — persisted as
the outer's `dsl_code.py`. From phase 2's perspective this is
indistinguishable from the no-plan output. Per-node DSLs are
intermediate scaffolding; only the root is handed off to translation.

## Per-node refactor

Per-node refactoring reuses the same `refactor_final` pass loop as the
no-plan path: same executor (`dsl`), same compliance and judge gates,
same post-validator. The differences are:

- **Reference code**: gold is computed from the node's own
  `reference_code` (the original module), not from any composed-parent
  form. This is what makes the children-first walk safe — each child
  is verified against its own ground truth.
- **`is_root` flag** flows through every gate. When `is_root=False`:
  - DSL output may be a tuple/list of tensors (children with
    multi-tensor `forward`s);
  - shape strictness is relaxed in `_compare_against_gold` (numel must
    still match, but reshapes/flattens are permitted);
  - the regex op-table drops sink-op requirements (`offchip_store`)
    since non-root DSLs don't terminate at off-chip;
  - the deterministic-translate post-validator runs but doesn't enforce
    the kernel-level output contract.
- **Pass-1 prompt additions**: the node's user prompt carries the
  parent-declared contract block (input shapes + values + output
  shape/permutation) and, for non-leaf nodes, the blackbox signatures
  for each child. The LLM picks the call-site contract; children
  refactor under it. See [pass_loop.md](pass_loop.md) for details.

## Replanning

If Phase 1 fails — i.e., some node's Pass-1 refactor burns its budget,
or Pass 2 fails on composition — the planner phase locates the failing
node's parent (its tree owner) and re-invokes the planner LLM on the
parent's `reference_code` with a populated **replan context**:
the failing path, the last few turns of the failing refactor
conversation, and any successfully-verified sibling DSLs. The new
subtree is spliced in place of the old one, the walk retries, and
already-verified siblings short-circuit since their DSLs are cached by
node path. Budget for the global replan loop is `--max-replans`
(default 3).

Replanning is not the same as the per-node retry budget inside the
planner LLM (which handles "the LLM emitted a malformed split") — it
handles "the *planner's decomposition* is the problem" by giving the
planner a chance to subdivide differently.

## Resume

`--resume-planner <OUTER_DIR>` resumes an outer that crashed mid-walk:

- the saved tree is loaded from the highest-numbered
  `<OUTER_DIR>/plan/iteration_*/tree.json` (replanning iterations are
  numbered 0, 1, … and the last one is the live tree),
- per-node verified DSLs are recovered by scanning
  `<OUTER_DIR>/refactor/**/status.txt` for files whose status is
  `PASS` and reading the matching `extracted_code.py`. The result is
  a `{node_path: dsl_code}` cache.

The walk re-runs only the non-verified nodes; verified siblings are
fed in as few-shot context immediately. Files are also copied forward
into the new outer directory so the resumed run is self-contained.

`--resume-planner` is mutually exclusive with `--resume`. The two
serve different purposes: `--resume` skips the lowering phase entirely
because a complete `dsl_code.py` already exists, while
`--resume-planner` resumes lowering itself when no `dsl_code.py`
exists yet.

## Checkpoint layout

Per outer (`outer_<i>/`), the planner phase writes:

```
plan/
└── iteration_<k>/                       # one per replan iteration
    ├── tree.json                        # the tree shape decided this iteration
    ├── <node_path_with_underscores>/
    │   ├── reference.py                 # the PyTorch reference for this node
    │   └── refactored.py                # internal nodes only — parent rewritten to call children
    └── turns/<node_path_with_underscores>/turn_<N>/
        ├── system_prompt.txt
        ├── user_prompt.txt
        ├── response.txt
        ├── reasoning.txt
        └── status.txt
refactor/<node_path>/[attempt_<i>/]refactor_final/turn_<N>/...
dsl_code.py                              # the verified root DSL handed off to phase 2
```

`status.txt` under `plan/.../turns/.../` carries the planner-pass
vocabulary: `LEAF`, `SPLIT_OK: children=[…]`, `MALFORMED_SPLIT: …`,
`NO_DECISION: …`, `GUARD_FAILED: …`, or `LLM_BAD_REQUEST: …`.
Forced-leaf and exhausted-budget nodes drop a marker file
(`MAX_DEPTH_FORCED_LEAF.txt`, `EXHAUSTED_FALLBACK_TO_LEAF.txt`) so the
distinction between "the planner chose leaf" and "the planner gave up"
is recoverable from the on-disk record.

The `refactor/<node_path>/` subtree mirrors the legacy
`refactor_final/turn_<N>/...` layout from the no-plan path. With
`--node-attempts > 1`, an `attempt_<i>/` layer is inserted so each
parallel attempt's turns stay separate.

## Invariants

- `plan_enabled=True` is incompatible with `--bundle-dir`,
  `--pipeline != standard`, and `--resume`. Bundles ship their own
  abstraction surface and refactor system prompt; the planner's
  `compose-equivalent` guard and few-shot composition logic both
  assume the standalone DSL surface, so they cannot be cleanly mixed.
- `--resume` and `--resume-planner` are mutually exclusive.
- The planner's per-node retry budget is fixed (not user-tunable);
  `--max-replans` is the only user-facing replanning knob.
- The planner pass always produces *some* tree, even if every node
  collapses to leaf. A planner phase that "produced nothing" is
  unreachable — the worst-case output is a single-leaf tree
  equivalent to the no-plan path.
