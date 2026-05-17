# FlatReassemble shape-uniformity bug — RESOLVED

## Resolution

The user-reported crash:

```
File "/workspace/NextStep/step_tl/src/timing_and_emulator/functional.py:1179"
RuntimeError: stack expects each tensor to be equal size, but got
              [1, 23, 1, 512] at entry 0 and [1, 19, 1, 512] at entry 1
```

was a real bug in both `_exec_flat_reassemble` (timing-model executor)
and `step_dsl.flat_reassemble` (DSL surface op): the pointer walk
indexed `inputs[i].shape[0]` directly, which broke when a producer
left a leading singleton stream dim in front of the dynamic per-expert
dim (e.g. an LLM variant ending each per-expert path with
`promote_outer`). With `inputs[i].shape = (1, D_i, 1, 512)` the
executor saw `shape[0] == 1`, ate the whole per-expert block as a
single "slot," and the inner `torch.stack(group)` then produced
variable-sized `token_groups` whose outer stack fails.

Fix mirrors what `_exec_flat_partition` already does
([functional.py:1134](../step_tl/src/timing_and_emulator/functional.py#L1134)):
flatten leading stream dims into a single axis before the ptr walk,
so the dynamic dim's position in the input stream prefix doesn't
matter. The IR semantics with `reassemble_rank=0` already mandate
that the entire input stream prefix is discarded — the executor just
needed to honor that.

### Files changed

* [step_tl/src/timing_and_emulator/functional.py:1148-1191](../step_tl/src/timing_and_emulator/functional.py#L1148-L1191)
  — `_exec_flat_reassemble` now builds `flat_inputs = [inp.reshape(-1,
  tile_r, tile_c) for inp in inputs]` and walks the flat list with
  `ptrs`.
* [src/step_dsl.py:948-995](src/step_dsl.py#L948-L995)
  — `flat_reassemble` mirrors the same flatten-first pattern over
  `inputs[i].underlying_tensor`.

### Regression tests

* [tests/test_flat_reassemble_shapes.py](tests/test_flat_reassemble_shapes.py)
  — 4 DSL-surface tests pinning the fix. The
  `_extra_leading_singleton_dynamic_at_axis1` test reproduces the
  user's exact shape signature `(1, D_i, 1, 512)`; the
  `_values_independent_of_leading_singletons` test confirms the fix is
  not just crash-avoidance — output values match the equivalent
  non-singleton layout.
* [tests/test_flat_reassemble_analyze_timing.py](tests/test_flat_reassemble_analyze_timing.py)
  — end-to-end test driving `make_analytical_scorer` → `analyze_timing`
  → `_exec_flat_reassemble` on a minimal DSL program that mirrors the
  failing autotune2 variant's shape topology (8 experts → 4, hidden
  512 → 32 for cheap test execution). Without the executor fix this
  raises with `[1, 3, 1, 32] vs [1, 11, 1, 32]` — same crash family,
  smaller dims.
* [tests/test_autotune2_runtime.py::test_verifier_factory_non_root_surfaces_dsl_exec_failure_as_feedback](tests/test_autotune2_runtime.py)
  — locks in the new DSL eager-exec smoke test (see below). Mocks
  `_exec_dsl_ref` to raise with the exact `[1, 23, 1, 512] vs [1, 19,
  1, 512]` signature and asserts the non-root verifier turns it into
  `## DSL eager-exec smoke test failed` feedback before any IR work
  begins. Also verifies the graph-build path is short-circuited so we
  don't redundantly re-translate after a DSL failure.
* [tests/test_autotune2_search.py::test_search_leaf_score_fn_raise_becomes_user_feedback](tests/test_autotune2_search.py#L661)
  — still in place; locks in the exception-to-LLM-feedback path
  inside the search loop. This is now a belt-and-suspenders test: the
  underlying executor bug is fixed, but if something else inside
  `analyze_timing` raises in the future, the autotuner still won't
  crash.

### Defensive change: DSL eager-exec smoke test in non-root verifier

The non-root verifier ([src/autotune2/runtime.py:537](src/autotune2/runtime.py#L537))
used to skip DSL eager execution entirely — its "correctness" gate
was a graph-build smoke test only, which builds the IR but never
runs `execute_values`. The flat_reassemble bug had a clean escape
route through that gap: graph build succeeded, then `score_fn`
crashed in `analyze_timing → execute_values → _exec_flat_reassemble`.

[src/autotune2/runtime.py:_dsl_exec_smoke_test](src/autotune2/runtime.py)
runs `_exec_dsl_ref(composed_source, dims, tensors)` before the
graph build, captures any exception, and returns it as
`_GateResult(feedback=..., status="DSL_EXEC_FAIL")`. The verifier
short-circuits on DSL-exec failure (no point translating to IR if
the source is broken at the torch level). Failures land in
`VerifyResult.feedback`, which `search_leaf`'s verify-fail branch
([src/autotune2/search.py:929-944](src/autotune2/search.py#L929-L944))
already wires into the next turn's user prompt — same path pass1's
correctness-gate failures take.

Net effect: any future shape-regime bug that would have crashed
`_exec_flat_reassemble` (or any other timing-model executor) on the
*same* torch operations that the DSL surface uses now surfaces at
the verifier with a clearer error message, instead of slipping
through to the score path and getting picked up by `_safe_score`'s
catchall.

### Verification

```bash
cd /workspace/NextStep/StepGenFlow12
PY=/root/miniconda3/envs/testenv/bin/python
$PY -m pytest tests/                              # 207 pass, 8 pre-existing fail
$PY -m pytest tests/test_flat_reassemble_*.py     # 5 pass
$PY -m pytest tests/test_autotune2_*.py           # 126 pass
```

The 8 pre-existing failures (`test_blackbox_stub.py`, `test_exec_extra_globals.py`)
are unrelated — they predate this work and trace back to the recent
`StepRawTensor` wrapping commit (`c03987b wrap torch.tensor as
steprawtensor`). They fail identically with the fix stashed.

### Reproduction (kept for posterity)

```bash
PY=/root/miniconda3/envs/testenv/bin/python
cd /workspace/NextStep/StepGenFlow12
$PY <<'EOF'
import pickle, json, sys
sys.path.insert(0, '.'); sys.path.insert(0, '/workspace/NextStep/step_tl/src')
from src.autotune2.search import (
    build_node_tensors_dict, build_synthetic_wrapper_for_node,
    _identity_input_contracts,
)
from src.autotune2.compose import make_analytical_scorer, compose_source

OUTER = '/workspace/checkpoints/2026-05-17-013132/prefill_transformer_simple/outer_3'
P = f'{OUTER}/pass1/iteration_0/root/moe/moe_dispatch/moe_dispatch__root_moe_moe_dispatch'
c = pickle.load(open(f'{P}/contract.pkl', 'rb'))
import glob, os
leaf = None
for s in sorted(glob.glob(f'{P}/attempt_*/refactor_final/turn_*/status.txt')):
    if open(s).read().strip() == 'PASS':
        leaf = open(os.path.dirname(s) + '/extracted_code.py').read(); break

# Inject promote_outer on each expert_output to recreate the failing
# shape regime (1, D_i, 1, 512) instead of (D_i, 1, 1, 512).
leaf = leaf.replace(
    'expert_outputs.append(weighted)',
    'expert_outputs.append(promote_outer(weighted))')

config = json.load(open('/workspace/checkpoints/2026-05-17-013132/config.json'))
at = json.load(open('autotune_config_2.json'))
wrapper = build_synthetic_wrapper_for_node(
    node_name='moe_dispatch__root_moe_moe_dispatch',
    parent_contract=c, input_contracts=_identity_input_contracts(c))
composed = compose_source(parent_dsl=wrapper + '\n' + leaf,
                          descendant_dsls_postorder=[])
print(make_analytical_scorer(
    dims=config['dims'], tensors=build_node_tensors_dict(c),
    hw_config=at['hw_config'],
    max_total_compute_bw=at['constraints']['max_total_compute_bw'])(composed))
EOF
```

Before fix: `RuntimeError: stack expects each tensor to be equal
size, but got [1, 23, 1, 1, 512] at entry 0 and [1, 19, 1, 1, 512] at
entry 1`. After fix: `(1579109, 88111112)` — cycles match the
non-`promote_outer` baseline (1579109 cycles) since the singleton
doesn't change timing; on-chip differs slightly due to memory
accounting.

## Why the user's framing was misleading

The HANDOFF originally framed this as a "mismatch between
`functional.py` and `step_dsl.py`." A careful side-by-side read shows
the two implementations had **identical** logic and **identical**
bugs. The fix patched both the same way. There was no mismatch to
reconcile — just the same shape-handling oversight, twice. (Future
debugging tip: when "implementation A diverges from B" framing
appears, verify the divergence is real before chasing it. Both sides
sharing a bug is a common and easy-to-miss possibility.)

## What's still in place (defense in depth)

`_safe_score` ([src/autotune2/search.py:704-748](src/autotune2/search.py#L704-L748))
remains around the score path. It's no longer the only thing keeping
the autotuner alive on this specific bug, but it's a useful
catchall: any future shape-regime that the functional executor
mishandles will surface as LLM feedback rather than a process crash.

## Pointers / where things live

* DSL surface op: [src/step_dsl.py:922-1003](src/step_dsl.py#L922-L1003)
  (`flat_reassemble`).
* IR op definition: `/workspace/NextStep/step_tl/src/step_py/ops.py`
  — `class FlatReassemble` at ~line 3204. Output stream shape:
  `control.stream + (new_dyn,) + tail`, where `tail = inputs[0].shape[
  -reassemble_rank:]`. With `reassemble_rank=0` (the DSL default —
  see [src/dsl_to_step.py:1144](src/dsl_to_step.py#L1144)), `tail` is
  empty and the entire input stream prefix is discardable — which is
  what the executor fix relies on.
* Timing-model executor:
  `/workspace/NextStep/step_tl/src/timing_and_emulator/functional.py:1148-1191`
  (`_exec_flat_reassemble`). Mirrors `_exec_flat_partition` (line
  1109) which already flattens leading stream dims correctly.
* Autotune2 score path:
  [src/autotune2/compose.py:191-204](src/autotune2/compose.py#L191-L204)
  (`make_analytical_scorer.score`).
* Crash-tolerant wrapper:
  [src/autotune2/search.py:704-748](src/autotune2/search.py#L704-L748)
  (`_safe_score`).

## Run the tests

```bash
cd /workspace/NextStep/StepGenFlow12
PY=/root/miniconda3/envs/testenv/bin/python
$PY -m pytest tests/test_flat_reassemble_*.py tests/test_autotune2_*.py
```
