# HuggingFace model importer for StepDB — Handoff

## Goal

Build a workflow that imports a HuggingFace model and emits a self-contained
StepDB kernel (`reference.py` + `precompute` entry + `bench_config.yaml`
entry) that the StepGenFlow12 planner can decompose. The point is **not**
that the imported reference is the cleanest possible PyTorch — the point is
that the planner sees real PyTorch code it can decompose into STeP, so we
can discover what the LLM is and isn't capable of translating.

User-stated requirement: "instead of showing the inspect source to the LLM
[as planner context], we should inline recursively all the inspect source
outputs of relevant submodules into the parent." The reference.py should
contain the full HF class source inlined, not just import the HF class.

(The previous HANDOFF.md content was about a Rust simulator divergence bug
that was fully resolved in an earlier session. That work is done; this file
replaces it.)

## Current state — decision pending

I've built two iterations of the importer and we're at a fork. **The next
agent should pick one of three paths (see "Decision point" below) and run.**

## Files touched this session

### StepDB (`/workspace/NextStep/StepDB`)

- **`hf_import.py` (new, ~370 lines).** Importer script. Walks
  `AutoModelForCausalLM.from_config(cfg).named_modules()`, collects unique
  non-torch classes, chases free-name references across transformers modules
  for module-level helpers, applies AST surgery to rewrite HF base classes
  to a `_HFBase` shim, emits a self-contained reference.py with stubs for HF
  decorators / dataclasses / framework singletons.

  Run: `python hf_import.py gpt2` → writes `seed_kernels/hf_imported/gpt2/reference.py`.

- **`seed_kernels/hf_imported/gpt2/reference.py` (current state: ~1055
  lines, auto-generated).** Self-contained reference for gpt2 with inlined
  source for GPT2LMHeadModel, GPT2Model, GPT2Block, GPT2Attention, GPT2MLP,
  Conv1D, NewGELUActivation, plus inlined free helpers (create_causal_mask,
  eager_attention_forward, etc.) and stubs. **Currently does NOT successfully
  run compute_gold** — fails at `_preprocess_mask_arguments` because
  `ALL_MASK_ATTENTION_FUNCTIONS._global_mapping` is missing. See "What didn't
  work."

- **`precompute.py` — `_precompute_hf__gpt2` (already in Option A form).**
  Constructs `AutoModelForCausalLM.from_config(cfg)`, dumps `state_dict()`
  (149 entries — includes tied weights and buffers), returns them keyed by
  HF parameter name plus `input_ids`. Seeded with `SEED` for weights and
  `SEED + 1` for input_ids so the two RNGs are decoupled. Aligned with the
  existing StepDB convention that precompute is the single source of truth.

- **`bench_config.yaml` — `hf__gpt2` entry.** Three presets (`tiny`/`small`/
  `medium`), `step_impl: null`, `origin: hf_import`. Confirmed excluded from
  `validate_functional.py`'s default seed-only sweep.

### StepGenFlow12 (`/workspace/NextStep/StepGenFlow12`)

- **`src/planner.py`.** Added `inspect` import, `_extract_hf_module_source(...)`
  helper, and threaded `hf_module_source` through both prompt-builder calls
  in `plan()`. The introspection: if `Model()` has a `self.model` attribute,
  run `inspect.getsource(type(self.model))`. Best-effort with `try/except`.
  **NOTE: user later modified planner.py separately (per system-reminder) —
  the change is intentional, don't revert.** This was the previous "show
  source as context" approach; the user has now pivoted to "inline source
  into the reference" so the planner-side change is mostly superseded but
  harmless to leave in.

- **`src/prompts.py`.** Added `hf_module_source: str | None = None` param to
  both `build_planner_user_prompt` and `build_replan_user_prompt`. Renders
  a `## Wrapped HF module source` section.

- **`prompts/planner_system.txt`.** Added a "HuggingFace-wrapped references"
  section documenting the `self.model` convention for child wrappers. This
  may also be partially obsoleted by the pivot to fully-inlined references.

## What worked

1. **state_dict-from-HF as the precompute source of truth.** Switching
   `_precompute_hf__gpt2` to dump `model.state_dict()` (instead of just
   `input_ids`) aligned the HF import with every other StepDB kernel's
   precompute convention: the dict is named, comprehensive, and consumed by
   both reference and (future) step_impl. Verified: 150 named entries,
   deterministic across calls, tied weights handled (`lm_head.weight` and
   `transformer.wte.weight` both present with identical values).

2. **`inspect.getsource(type(self.model))` planner introspection.** Works
   end-to-end: extracts 3863 chars of GPT2LMHeadModel for gpt2; renders
   cleanly into prompt; falls back to None for non-HF references. Smoke test
   confirmed before the user redirected toward full inlining.

3. **AST base-class rewriting via `_HFBase` shim.** Replacing HF base
   classes (`GPT2PreTrainedModel`, `GenerationMixin`, `GradientCheckpointingLayer`,
   anything ending in `PreTrainedModel`) with a small shim that accepts
   `config` and provides no-op `post_init`/`_init_weights`/`warn_if_*` etc.
   This pattern is sound — keep the shim, expand the no-op method set as
   needed.

4. **Stubbing HF metadata decorators as identity functions.** `@auto_docstring`,
   `@can_return_tuple`, `@deprecate_kwarg`, `@capture_outputs`, etc. all
   work as `_identity_decorator` in the prologue. Important fix: the
   identity decorator must distinguish `@dec` (single function/class arg
   passed as decoration target) from `@dec(...)` (factory pattern) using
   `inspect.isfunction(a[0]) or inspect.isclass(a[0])`. A naive
   `callable(a[0]) and not isinstance(a[0], type)` is wrong because
   classes are types, and silently replaces decorated classes with the
   wrapper function (which we hit and fixed).

## What didn't work

1. **First attempt to chase cross-module helpers via free-name resolution.**
   Walking `_free_names(class_sources)` and pulling `inspect.getsource` for
   every transformers-defined callable that matched produced a 3300-line
   file because `auto_docstring`'s own definition got inlined (and pulled
   in its dozens of helpers — `auto_class_docstring`, `auto_method_docstring`,
   `_get_model_info`, `parse_docstring`, etc.). Fixed by excluding
   `_STRIPPED_DECORATORS` from the resolution pool — they get identity
   stubs in the prologue and never need their real implementation.

2. **Stubbing `merge_with_config_defaults` was necessary but not sufficient.**
   That decorator fills missing kwargs from `config` defaults, which was
   injecting a non-None Cache into `past_key_values`. Stubbing it to identity
   keeps `past_key_values=None` (the source default), letting the forward
   skip the cache codepath. But then masking still calls `create_causal_mask`
   even on `None` past_key_values, which has its own framework dependencies.

3. **`_OpaqueStub.__getattr__` returning callables.** First attempt returned
   `lambda *a, **kw: 0` for unknown attrs so `cache.get_seq_length()` worked.
   But then `hasattr(past_key_values, "is_sliding")` returned True (because
   __getattr__ doesn't raise), and `False in past_key_values.is_sliding`
   tried to iterate a lambda → TypeError. Reverted to raising AttributeError;
   that makes `hasattr` checks return False so forward branches fall through
   to defaults. But it doesn't fix the case where the code calls a method
   without first checking hasattr.

4. **Going down the masking tarpit.** After stubbing
   `ALL_MASK_ATTENTION_FUNCTIONS = {}`, the next error became
   `'dict' object has no attribute '_global_mapping'`. HF's masking utilities
   are deeply tied to attention-function registries that expect specific
   internal structure. Continuing to stub framework internals one at a time
   is a treadmill, and **every new model family will surface more.**

## Decision point — next agent should choose

The fundamental tension: HF model classes and HF framework code (Cache,
masking_utils, attention function registry, decorators that fill defaults)
are tightly coupled. We can't trivially inline one without dragging in the
other. Three options:

### Option 1: Keep stubbing framework names

Add `ALL_MASK_ATTENTION_FUNCTIONS = _OpaqueStub()` or similar, run again,
fix next error, repeat. Estimated 10–20 more iterations for gpt2 alone.
Each new model family adds more. **High maintenance, fragile.**

### Option 2: Reimplement framework helpers ourselves

Replace `create_causal_mask` with a 10-line lower-triangular bias function.
Replace `Cache` with a real tiny no-op cache class. Replace
`_preprocess_mask_arguments` with a minimal version. Estimated ~100–200
lines of hand-written framework code per model family. **More work upfront
but self-contained.**

### Option 3 (recommended): Inline source for visibility, execute via HF

Pivot the importer:
- Keep all inlined class source in the file (planner reads/decomposes it).
- Change the `Model` wrapper to use the **real**
  `transformers.AutoModelForCausalLM.from_config(...)` for `compute_gold`.
- The inlined classes become *what the planner reads*, not *what runs*.

This fully satisfies the user's original goal ("give the planner real
PyTorch code to decompose") while sidestepping all framework-stub work.
The execution path is via transformers (which we know works); the
decomposition path is via inlined source (which the planner sees in the
file). Children the planner emits can either copy from the inlined source
or use HF classes directly — that's the planner's choice, not the
importer's.

**This is the option I was about to take when the conversation paused.**
The user said "yes, go ahead" several iterations earlier but the scope
they signed up for has clearly bloated. Confirm the path with them before
sinking more time into option 1 or 2.

## Next steps

1. **Confirm with user which option (1/2/3).**

2. **If option 3 (recommended):**
   - Edit [hf_import.py](hf_import.py) `_TRAILER_TEMPLATE`: change `Model.__init__`
     to use `from transformers import AutoModelForCausalLM` and
     `AutoModelForCausalLM.from_config(config)` instead of `{top_class}(config)`.
   - The inlined classes stay in the file as code the planner can read.
   - `precompute.py` already extracts state_dict from the real HF class
     (unchanged), so state_dict load is compatible.
   - Re-run `python hf_import.py gpt2`, verify
     `compute_gold` succeeds on all three presets, finite logits, deterministic.

3. **Run StepGenFlow12 planner against `hf__gpt2`.** This is the actual
   integration test the user wanted. Needs an `llm_config` with API
   key + model. The orchestrator entry point is at
   [`/workspace/NextStep/StepGenFlow12/src/orchestrator.py`](../StepGenFlow12/src/orchestrator.py).
   Look at how `prefill_transformer_simple` is invoked and follow the same
   pattern.

4. **Consider whether to revert/keep the planner.py + prompts.py +
   planner_system.txt changes from earlier.** With option 3 the inline-as-
   context approach in planner.py is mostly redundant (the source is
   inlined directly in the reference). But the changes are defensive (no-op
   when no HF wrapper detected) and don't hurt. Probably fine to leave.
   **System-reminder noted that user modified planner.py — don't revert
   their changes.**

## Existing context the next agent should know

- **Project conventions.** Read [`.claude/CLAUDE.md`](.claude/CLAUDE.md) for
  code-style requirements. Notable: avoid unnecessary try/except, use
  assertions, keep code minimal.

- **Existing kernel format.** Look at
  [seed_kernels/transformer_layer/prefill_transformer_simple/reference.py](seed_kernels/transformer_layer/prefill_transformer_simple/reference.py)
  for the "normal" StepDB reference convention: `class Model(nn.Module)` with
  `forward(*tensor_args)`, `get_inputs(dims)`, `get_init_inputs(dims)`,
  `compute_gold(dims)`. The HF import follows the same shape but
  `get_init_inputs` returns `[model_name, config, state_dict]` and forward
  takes input_ids.

- **Python env.** `/root/miniconda3/envs/testenv/bin/python` is the right
  interpreter. Has `torch 2.11.0+cu130` and `transformers 5.8.1`.

- **The DSL target for downstream decomposition.** See
  [/workspace/checkpoints/prefill_transformer_working/prefill_transformer_simple/outer_0/dsl_code.py](/workspace/checkpoints/prefill_transformer_working/prefill_transformer_simple/outer_0/dsl_code.py)
  for an example of what a successful StepGenFlow12 output looks like — this
  is what the planner+refactor agents would produce starting from `hf__gpt2`
  reference.py.

## Quick start for the next agent

```bash
# Verify current state
cd /workspace/NextStep/StepDB
/root/miniconda3/envs/testenv/bin/python -c "
import sys; sys.path.insert(0, '.')
from loader import load_problem, get_dims
mod = load_problem('hf__gpt2')
out = mod.compute_gold(get_dims('hf__gpt2', 'tiny'))
print(out.shape)
"
# Expect: ERROR at ALL_MASK_ATTENTION_FUNCTIONS._global_mapping (the wall).

# Regenerate after edits
/root/miniconda3/envs/testenv/bin/python hf_import.py gpt2
```
