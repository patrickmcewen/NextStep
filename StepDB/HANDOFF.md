# Rust simulator vs functional divergence — Handoff

## Goal

Make `python evaluate.py generated_prefill_transformer small` pass correctness
against its PyTorch reference. The kernel itself is correct (functional sim
matches gold to `rel_err=9.3e-7`); the Rust simulator (`step_perf`) was
returning values ~1000× too large, with `max_diff = 2,110,076.75`.

Found and fixed one Rust simulator bug. One downstream divergence remains
(`max_diff = 2,638` after the first fix). The remaining bug is **not in the
graph or functional sim** — only in Rust execution of certain ops in this
kernel.

## Current progress

### Done
1. **Validator now compares by numel.**
   [validate_functional.py:184-201](validate_functional.py#L184-L201) — was
   strict `gold.shape == sim.shape`; replaced with `numel()` check + flat
   reshape. The kernel produces output shape `(32768, 1)` while gold is
   `(64, 512)` — same elements, different layout. With the patch:

   ```
   PASS  generated_prefill_transformer  small  max_abs_err=1.83e-03  rel_err=9.30e-07
   ```

2. **Rust `add_constant` bug fixed.**
   At [step_tl/step-perf/src/functions/map_fn.rs:275](../step_tl/step-perf/src/functions/map_fn.rs#L275),
   `add_constant<T>` was doing `arr1.mapv(|x| x * constant)` (copy-paste from
   `mul_constant`). Changed to `x + constant`. The Python serializer
   ([sim/__init__.py:120-128](../step_tl/src/sim/__init__.py#L120-L128))
   correctly maps `map_fn.AddImmediate(float)` → proto `AddConstant` with
   `constant_float`, so the dispatch is fine — only the function body was
   wrong. `mul_constant` was correct; `sub_constant` has a stale "Multiply..."
   comment but correct body.

3. **Rebuilt step_perf** — see
   [`memory/step_perf_rebuild.md`](/root/.claude/projects/-workspace/memory/step_perf_rebuild.md).
   Critical: `step_perf` is a separate editable maturin crate at
   `step_tl/step-perf/`, NOT rebuilt by `maturin develop` from `step_tl/`.
   Requires `PROTOC=…/torch/bin/protoc` and `unset VIRTUAL_ENV`. Stale `.so`
   produces identical max_diff before/after a "fix" — always verify the .so
   mtime changed.

### After fix
- `max_diff` dropped from **2,110,076.75 → 2,638.46** (~800× reduction)
- BinaryMap_9 (RMS-norm output `normed`) now matches functional to ~1e-7
- Remaining divergence is somewhere downstream of BinaryMap_9

## What worked

- **Tap-based bisection.** [`/tmp/tap_kernel.py`](/tmp/tap_kernel.py) loads the
  failing kernel via `validate_functional.build_graph_from_impl`, then adds an
  extra `OffChipStore(..., store_file_name="tap_output")` on a chosen node
  (selected by class name + index, or by `instance_id`). Crucially it does
  **not trim** the graph — both the original `output` store and the new
  `tap_output` store run, so all consumer channels stay connected. Then it
  runs functional sim and Rust sim and compares values at the tap.

  Usage:
  ```bash
  unset VIRTUAL_ENV
  source /root/miniconda3/etc/profile.d/conda.sh && conda activate testenv
  python /tmp/tap_kernel.py id 9       # tap node with instance_id=9
  python /tmp/tap_kernel.py class Streamify 0   # tap first Streamify
  python /tmp/tap_kernel.py list       # list all nodes with shapes & fn names
  ```

- **Magnitude signatures.** Once we saw Rust = func × 1000 exactly (ratio mean
  1000.000061, std 0.01), recognizing `1/sqrt(eps=1e-6) = 1000` pointed
  directly at the eps step in RMS-norm. Look for arithmetic relationships
  between the residual factor and known constants (eps, head_dim, hidden, etc.)
  before deep-diving.

- **Functional sim as ground truth.** Functional Python emulator is the
  reference. If functional matches PyTorch gold and Rust diverges, the bug is
  necessarily in `step-perf`. Use that asymmetry — don't trust the Rust output
  even when "the test passes" (no test exercised `add_constant` with non-blank
  data; the existing Rust round-trip tests use `Tile::new_blank` which masks
  data bugs).

## What didn't work / don't repeat

- **Trimming the graph to only ancestors of the tap.** Caused
  `DisconnectedReceiver` panics in Rust because some kept nodes (e.g. shared
  `cos_stream`) had downstream consumers we removed → dangling sender channels.
  The "no-trim, dual-store" approach is fine and the extra cost is small.

- **`maturin develop --release` from `step_tl/`.** That builds the `step_tl`
  module but **not** `step_perf` (the simulator). After such a build,
  `evaluate.py` reports the *exact same* `max_diff` as before — silent
  no-op-from-the-user's-perspective. Always rebuild from `step_tl/step-perf/`
  and verify the `.so` mtime at
  `/root/.miniconda3/envs/testenv/lib/python3.12/site-packages/step_perf/*.so`.

- **Conflating Streamify with the bug.** The user's initial lead was
  "Streamify was just added recently". A standalone Streamify repro
  ([`/tmp/streamify_repro.py`](/tmp/streamify_repro.py),
  [`/tmp/streamify_repro2.py`](/tmp/streamify_repro2.py)) confirmed Streamify
  is correct for both `(1,1)` and `(1,32)` tiles. The real divergence was in
  `add_constant` (RMS norm eps step), which happened to first surface inside
  the Q_str = Streamify branch only because that's the first OffChipStore tap
  we tried.

## Next steps

### 1. Bisect the remaining `max_diff=2,638` divergence

Pattern observed at the final output (after `add_constant` fix):
- Magnitudes: `|func|max = 1969.86`, `|rust|max = 1969.86` — same scale ✓
- `idx 0` matches to ~1e-4 ✓
- Many subsequent indices have similar magnitudes but **flipped sign or
  reordered** — e.g. `idx 1174: func=988.19, rust=-1650.27` and
  `idx 21965: func=-1122.9, rust=1451.67`.
- Ratio min/max wildly spread (-517 to +787) — *not* a uniform scaling bug
  like `add_constant`.

This pattern suggests **ordering/dispatch divergence** — values landing at the
wrong output position. Likely candidates, in rough order of suspicion:

- `FlatPartition` / `FlatReassemble` with multihot `SelectGen` (MoE
  dispatch, [step_impl.py:259-280](seed_kernels/transformer_layer/generated_prefill_transformer/step_impl.py#L259-L280)).
  Multi-hot k=2 dispatch where `expert_onehot` has shape `(64, 2, 8)` and each
  row of 2 has exactly one of 8 experts set — Rust may have multihot routing
  bugs.
- `Streamify` with non-trivial buffers — already partially tested but only the
  Q path with stride `(4,1,16)`. The K/V paths use stride `(1,4)` over
  `(4,64)` and weren't tested in isolation.
- `RetileStreamify` / `Accum(RetileRow/RetileCol)` ordering in `compute_qkv`
  and `rotate_half`.
- `StaticReassemble` ordering (used in `rotate_half`).

**How to bisect**: walk the node list (`python /tmp/tap_kernel.py list`) and
tap at strategic milestones. Suggested ordering (each tap is ~30s functional
+ ~2min Rust):

1. **End of attention**, before MoE — tap `attention_o_proj`'s output
   `out = res_add_0`. If matches → bug is in MoE.
2. If matches: tap inside MoE — the `summed = Accum(merged, accum_rank=2)`
   ([step_impl.py:281](seed_kernels/transformer_layer/generated_prefill_transformer/step_impl.py#L281))
   and one of the per-expert `scaled = BinaryMap(down, weight_parts[e], Mul)`.
3. If attention doesn't match: tap `attn_flat = Flatten(attn, ...)` (end of
   `attention_compute`) → bisect attention.
4. If attention matches: tap `Qh`, `Kh`, `Vh` (end of `compute_qkv`) — the
   Bufferize/Streamify reshape.
5. If those match: tap `Q`, `K`, `V` out of `pre_attention` (i.e. `Q_out`,
   `K_out`, `V_out` BinaryMaps in `per_head_norm_and_rope`).

To tap by name when `instance_id` is unknown, add a `name=` print pass that
walks the graph and matches by `type(node).__name__` and a position in the
build flow. The list-mode output already shows shapes which often disambiguate.

### 2. Once bisected, fix the responsible Rust op

Apply the same pattern: read the Rust impl, find the bug, edit, then rebuild
with the PROTOC incantation. Re-verify by re-running the tap (it should now
match functional) and then `evaluate.py` (expect `max_diff` to drop further).

### 3. Run a regression sweep

Once `evaluate.py generated_prefill_transformer small` passes (`max_diff <
1e-3`):

```bash
unset VIRTUAL_ENV
source /root/miniconda3/etc/profile.d/conda.sh && conda activate testenv
cd /workspace/NextStep/StepDB
python validate_functional.py --all
python evaluate.py --all
```

The `add_constant` fix in particular is load-bearing for any kernel using
`AddImmediate` (every RMS-norm-style kernel), so other seed kernels may also
have flipped from broken-to-passing. Confirm.

### 4. (Lower priority) Tighten Rust round-trip tests

The existing Rust unit tests for `add_constant` / `mul_constant` / `streamify`
use `Tile::new_blank` (zero data), so they only verify shape/stop tokens, not
data correctness. That's why this 1000× bug shipped. A follow-up: add a
test that feeds non-zero tile data through `add_constant` and asserts
`out[i] == in[i] + c` for non-trivial `c`. Same for `sub_constant` (whose
stale comment hints the dev may have copy-pasted that body too — body is
correct but worth covering).

## Repro commands (entire flow from cold state)

```bash
# Activate env
unset VIRTUAL_ENV
source /root/miniconda3/etc/profile.d/conda.sh && conda activate testenv
cd /workspace/NextStep/StepDB

# 1. Functional sim sanity check (kernel is logically correct)
python validate_functional.py generated_prefill_transformer small
# → PASS  max_abs_err=1.83e-03  rel_err=9.30e-07

# 2. Rust sim check (currently FAILs with max_diff=2638 after add_constant fix)
python evaluate.py generated_prefill_transformer small
# → FAIL @ correctness ... Output incorrect: max_diff=2638.4561

# 3. Bisect a single node (example)
python /tmp/tap_kernel.py id 9
# → numel ok; max_diff=9.5e-07  (BinaryMap_9 = normed, now matches)

# 4. List nodes
python /tmp/tap_kernel.py list | head -60

# 5. Rebuild step_perf after editing step-perf/src/*
cd /workspace/NextStep/step_tl/step-perf
PROTOC=/root/miniconda3/envs/testenv/lib/python3.12/site-packages/torch/bin/protoc \
    maturin develop --release
# Then re-run step 2/3.
```

## Key files

- [validate_functional.py](validate_functional.py) — patched (numel compare)
- [evaluate.py](evaluate.py) — unchanged; uses Rust sim
- [seed_kernels/.../generated_prefill_transformer/step_impl.py](seed_kernels/transformer_layer/generated_prefill_transformer/step_impl.py) — the failing kernel
- [../step_tl/step-perf/src/functions/map_fn.rs](../step_tl/step-perf/src/functions/map_fn.rs) — `add_constant` patched at line 275
- [../step_tl/step-perf/src/proto_driver/mod.rs](../step_tl/step-perf/src/proto_driver/mod.rs) — dispatch for AddConstant/MulConstant at lines 234, 330
- [../step_tl/src/sim/__init__.py](../step_tl/src/sim/__init__.py) — serialize AddImmediate → proto AddConstant at line 120
- [/tmp/tap_kernel.py](/tmp/tap_kernel.py) — bisection harness
- [/tmp/streamify_repro.py](/tmp/streamify_repro.py) / [/tmp/streamify_repro2.py](/tmp/streamify_repro2.py) — Streamify isolation tests (passed)

## Related memory entries

See `/root/.claude/projects/-workspace/memory/MEMORY.md`. Notably:
- `step_perf_rebuild.md` — how to rebuild the Rust sim
- All the `*_rank.md` and `*_constraint.md` files are from previous sessions
  tightening DSL/IR/functional assertions to catch Rust-vs-Python semantic
  gaps. Following the same pattern, the `add_constant` fix should ideally be
  paired with a Rust unit test (next-steps #4) so it can't regress.
