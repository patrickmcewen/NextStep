# DSL-form autotuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the autotuner from rewriting `build_graph(dims, tensors)` to rewriting `tiled_reference(dims, tensors)` (DSL form), with new perf-knob kwargs (`compute_bw`, `par_dispatch`) on the DSL surface that the deterministic translator forwards to STeP nodes.

**Architecture:** Two perf knobs become first-class kwargs on the relevant DSL functions and are forwarded by `dsl_to_step.translate()` into STeP node constructors. The autotuner's per-turn loop expands from one correctness gate to a triple-gate chain (DSL exec → translate → IR sim) plus a timing-model step; each gate failure returns a distinct status so the LLM gets feedback in the surface it controls. The autotune system prompt swaps `ops.py + utility_ops.py + functional.py` for `step_dsl.py`.

**Tech Stack:** Python 3, pytest, torch, the existing StepGenFlow9 orchestrator + autotune subsystem.

**Spec:** [design/dsl_autotuner.md](dsl_autotuner.md)

---

## File Structure

**Modified:**
- `src/step_dsl.py` — add `compute_bw=1` kwarg to 27 compute DSL functions; add `par_dispatch=1` kwarg to 6 memory DSL functions; one assertion per kwarg.
- `src/dsl_to_step.py` — extend three handler factories and four special-case handlers with `compute_bw` extraction; extend six memory handlers + the offchip_store-in-return branch with `par_dispatch` extraction.
- `src/prompts.py` — add `_STEP_DSL_PY` path constant; rework the loader in `build_autotune_system_prompt`; tweak the wording in `build_autotune_user_prompt`.
- `prompts/autotune_system.txt` — placeholder + narrative swap.
- `prompts/autotune_parallel_system.txt` — placeholder + narrative swap.
- `src/autotune.py` — extract `_evaluate_dsl_turn` helper; switch to `_resolve_resume_dsl`; rebuild baseline measurement and main loop; new artifact names; drop dead code.

**Created:**
- `tests/test_step_dsl_knobs.py` — kwarg acceptance + assertion tests for the DSL surface.
- `tests/test_dsl_to_step_knobs.py` — translator forwarding + BC tests.
- `tests/test_autotune_loop.py` — `_evaluate_dsl_turn` gate ladder tests + DSL resume test.

**Unchanged (dependencies relied on):**
- `src/orchestrator.py` — `_run_dsl_correctness` (orchestrator.py:271), `_run_graph_correctness` (:282), `_resolve_resume_dsl` (:921), `_write`, `_extract_code`, `_reasoning_text`. The per-outer integration `_run_outer_autotune` (:164) is untouched and keeps working because the resume-from path it passes (`outer_dir`) already contains a `dsl_code.py`.
- `src/tools.py` — `_exec_dsl_ref` (gate 1 implementation, called via `_run_dsl_correctness`).

---

### Task 1: Expose perf knobs on the DSL surface

**Files:**
- Modify: `src/step_dsl.py` — 33 function signatures.
- Create: `tests/test_step_dsl_knobs.py`.

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_step_dsl_knobs.py`:

```python
"""Knob kwargs on the DSL surface — keyword-only, asserted, inert at eager exec."""

import pytest
import torch

from src.step_dsl import (
    accum_add,
    binary_map_accum,
    binary_matmul,
    offchip_load,
    unary_silu,
)


def test_binary_matmul_compute_bw_kwarg_accepted_and_inert():
    a = torch.randn(2, 4, 4)
    b = torch.randn(2, 4, 4)
    expected = binary_matmul(a, b)
    actual = binary_matmul(a, b, compute_bw=8)
    assert torch.equal(actual, expected)


def test_binary_matmul_compute_bw_zero_raises():
    a = torch.randn(2, 4, 4)
    b = torch.randn(2, 4, 4)
    with pytest.raises(AssertionError, match="compute_bw must be >= 1"):
        binary_matmul(a, b, compute_bw=0)


def test_unary_silu_compute_bw_kwarg_accepted_and_inert():
    x = torch.randn(2, 4, 4)
    expected = unary_silu(x)
    actual = unary_silu(x, compute_bw=4)
    assert torch.equal(actual, expected)


def test_accum_add_compute_bw_kwarg_accepted_and_inert():
    x = torch.randn(3, 4, 4)
    expected = accum_add(x)
    actual = accum_add(x, compute_bw=2)
    assert torch.equal(actual, expected)


def test_binary_map_accum_compute_bw_kwarg_accepted_and_inert():
    a = torch.randn(3, 4, 4)
    b = torch.randn(3, 4, 4)
    expected = binary_map_accum(a, b)
    actual = binary_map_accum(a, b, compute_bw=4)
    assert torch.equal(actual, expected)


def test_offchip_load_par_dispatch_kwarg_accepted_and_inert():
    underlying = torch.randn(8, 8)
    kwargs = dict(stride=(1,), out_shape_tiled=(2,), tile_row=4, tile_col=8)
    expected = offchip_load(underlying, **kwargs)
    actual = offchip_load(underlying, **kwargs, par_dispatch=4)
    assert torch.equal(actual, expected)


def test_offchip_load_par_dispatch_zero_raises():
    underlying = torch.randn(8, 8)
    with pytest.raises(AssertionError, match="par_dispatch must be >= 1"):
        offchip_load(
            underlying,
            stride=(1,),
            out_shape_tiled=(2,),
            tile_row=4,
            tile_col=8,
            par_dispatch=0,
        )
```

- [ ] **Step 1.2: Run tests and verify they fail**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/test_step_dsl_knobs.py -v`
Expected: FAIL — `TypeError: binary_matmul() got an unexpected keyword argument 'compute_bw'` (or similar) on every test.

- [ ] **Step 1.3: Add `compute_bw=1` kwargs to compute DSL functions**

In `src/step_dsl.py`, append `compute_bw=1` as a keyword-only arg (positionally last) to each function listed below, and insert a single assertion at the top of the body. Bodies are otherwise unchanged.

Representative shape (this is exactly the diff for `binary_matmul`):

```python
def binary_matmul(a, b, weight_transposed=False, compute_bw=1):
    assert compute_bw >= 1, f"binary_matmul: compute_bw must be >= 1, got {compute_bw}"
    _assert_float(a, "binary_matmul")
    _assert_float(b, "binary_matmul")
    _assert_stream_match(a, b, "binary_matmul")
    if weight_transposed:
        return torch.matmul(a, b.transpose(-2, -1))
    return torch.matmul(a, b)
```

Apply the same shape (append `compute_bw=1`, prepend `assert compute_bw >= 1, f"<fn_name>: compute_bw must be >= 1, got {compute_bw}"`) to **all 27** compute functions:

| family | functions |
|---|---|
| binary | `binary_matmul`, `binary_mul`, `binary_add`, `binary_div`, `binary_is_equal`, `binary_row_wise_append`, `binary_set_offset`, `binary_cache_write_addr_gen` |
| fused | `binary_map_accum` |
| unary | `unary_silu`, `unary_square`, `unary_exp`, `unary_rsqrt`, `unary_pow2`, `unary_mul_imm`, `unary_add_imm`, `unary_sub_imm`, `unary_rowwise_sum`, `unary_select_to_scalar`, `unary_to_const_int`, `unary_mask_row` |
| accum | `accum_add`, `accum_mul`, `accum_max`, `accum_retile_row`, `accum_retile_col`, `accum_signal_req_all_read` |

- [ ] **Step 1.4: Add `par_dispatch=1` kwargs to memory DSL functions**

Same pattern for the 6 off-chip memory functions: `offchip_load`, `offchip_load_ref`, `dyn_offchip_load`, `random_offchip_load`, `offchip_store`, `random_offchip_store`. Each gains a trailing `par_dispatch=1` kwarg and a leading `assert par_dispatch >= 1, f"<fn_name>: par_dispatch must be >= 1, got {par_dispatch}"`.

Representative shape (`offchip_load`):

```python
def offchip_load(underlying, stride, out_shape_tiled, tile_row, tile_col,
                 transposed=False, par_dispatch=1):
    assert par_dispatch >= 1, f"offchip_load: par_dispatch must be >= 1, got {par_dispatch}"
    assert underlying.dtype in [torch.float32, torch.float16], ...
    # ... rest of body unchanged
```

- [ ] **Step 1.5: Update the module docstring**

In `src/step_dsl.py`, replace the one-line module docstring `"""STeP DSL"""` with:

```python
"""STeP DSL.

Each DSL function whose lowered STeP node carries a perf knob accepts that
knob as a keyword-only argument with default 1:

  - compute DSL calls (binary_*, unary_*, accum_*, binary_map_accum) accept
    ``compute_bw=N``.
  - off-chip DSL calls (offchip_load*, dyn_offchip_load, random_offchip_*,
    offchip_store) accept ``par_dispatch=N``.

The kwarg is asserted (>= 1) but otherwise inert at eager exec time —
the deterministic translator (dsl_to_step.py) reads it back from the AST
and forwards it to the STeP node constructor.
"""
```

- [ ] **Step 1.6: Run tests and verify they pass**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/test_step_dsl_knobs.py -v`
Expected: PASS — 7 tests pass.

- [ ] **Step 1.7: Run the full test suite to confirm no regressions**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/ -x`
Expected: existing tests still pass; only the 7 new tests are added to the count.

- [ ] **Step 1.8: Commit**

```bash
cd /workspace/DEIOpt && git add StepGenFlow9/src/step_dsl.py StepGenFlow9/tests/test_step_dsl_knobs.py
git commit -m "$(cat <<'EOF'
expose compute_bw / par_dispatch kwargs on DSL surface

Adds compute_bw=1 to the 27 DSL functions whose lowered STeP node carries
compute_bw, and par_dispatch=1 to the 6 off-chip memory DSL functions. The
kwargs are asserted (>= 1) but inert at eager exec — they exist so the
deterministic translator can forward them to STeP nodes for the autotuner
to use.
EOF
)"
```

---

### Task 2: Forward perf knobs through the translator

**Files:**
- Modify: `src/dsl_to_step.py` — three factories + four special-case handlers + six memory handlers + offchip_store-in-return branch.
- Create: `tests/test_dsl_to_step_knobs.py`.

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_dsl_to_step_knobs.py`:

```python
"""Translator forwards compute_bw / par_dispatch from DSL kwargs to STeP ctor kwargs."""

import ast

from src.dsl_to_step import translate


def _find_call(tree, fn_name):
    """First Call node whose func.id == fn_name (depth-first)."""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == fn_name):
            return node
    return None


def _kwarg_value(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return ast.unparse(kw.value)
    return None


_BASE_TILED_REF = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4{a_kwargs})
    b = offchip_load(tensors["B"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4)
    c = binary_matmul(a, b{matmul_kwargs})
    return offchip_store(c{store_kwargs})
'''


def _render(a_kwargs="", matmul_kwargs="", store_kwargs=""):
    return _BASE_TILED_REF.format(
        a_kwargs=a_kwargs,
        matmul_kwargs=matmul_kwargs,
        store_kwargs=store_kwargs,
    )


def test_binary_matmul_forwards_compute_bw():
    out = translate(_render(matmul_kwargs=", compute_bw=8"))
    tree = ast.parse(out)
    call = _find_call(tree, "BinaryMap")
    assert call is not None, f"BinaryMap not found in:\n{out}"
    assert _kwarg_value(call, "compute_bw") == "8"


def test_offchip_load_forwards_par_dispatch():
    out = translate(_render(a_kwargs=", par_dispatch=4"))
    tree = ast.parse(out)
    # First LinearOffChipLoad in the AST is the one for tensors["A"].
    call = _find_call(tree, "LinearOffChipLoad")
    assert call is not None, f"LinearOffChipLoad not found in:\n{out}"
    assert _kwarg_value(call, "par_dispatch") == "4"


def test_offchip_store_forwards_par_dispatch():
    out = translate(_render(store_kwargs=", par_dispatch=2"))
    tree = ast.parse(out)
    call = _find_call(tree, "OffChipStore")
    assert call is not None, f"OffChipStore not found in:\n{out}"
    assert _kwarg_value(call, "par_dispatch") == "2"


def test_default_compute_bw_is_one_when_omitted():
    """Backwards compat: DSL without new kwargs translates with default of 1."""
    out = translate(_render())
    tree = ast.parse(out)
    matmul = _find_call(tree, "BinaryMap")
    assert _kwarg_value(matmul, "compute_bw") == "1"
    load = _find_call(tree, "LinearOffChipLoad")
    assert _kwarg_value(load, "par_dispatch") == "1"
    store = _find_call(tree, "OffChipStore")
    assert _kwarg_value(store, "par_dispatch") == "1"


def test_unary_map_forwards_compute_bw():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4)
    b = unary_silu(a, compute_bw=4)
    return offchip_store(b)
'''
    out = translate(src)
    tree = ast.parse(out)
    call = _find_call(tree, "UnaryMap")
    assert _kwarg_value(call, "compute_bw") == "4"


def test_accum_forwards_compute_bw():
    src = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(3,),
                      tile_row=4, tile_col=4)
    b = accum_add(a, rank=1, compute_bw=2)
    return offchip_store(b)
'''
    out = translate(src)
    tree = ast.parse(out)
    call = _find_call(tree, "Accum")
    assert _kwarg_value(call, "compute_bw") == "2"
```

- [ ] **Step 2.2: Run tests and verify they fail**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/test_dsl_to_step_knobs.py -v`
Expected: FAIL — the BC test (`test_default_compute_bw_is_one_when_omitted`) might pass for `par_dispatch=1` because the translator hardcodes that literal today, but `compute_bw` will be missing entirely on `BinaryMap`/`UnaryMap`/`Accum` ctors. Other tests fail because the new kwargs aren't read off the AST.

- [ ] **Step 2.3: Forward `compute_bw` from compute factories and special-case handlers**

In `src/dsl_to_step.py`, edit `_make_binary_map`, `_make_unary_map`, and `_make_accum` to extract `compute_bw` and append it to the ctor string.

`_make_binary_map` (replace the existing function):

```python
def _make_binary_map(map_class):
    def handler(state, target, call):
        a = _src(_arg(call, 0, "a"))
        b = _src(_arg(call, 1, "b"))
        if map_class == "Matmul":
            wt = _arg(call, 2, "weight_transposed")
            wt_s = f"weight_transposed={_src(wt)}" if wt is not None else ""
            fn_str = f"map_fn.Matmul({wt_s})"
            compute_bw = _arg_or_default(call, 3, "compute_bw", "1")
        else:
            fn_str = f"map_fn.{map_class}()"
            compute_bw = _arg_or_default(call, 2, "compute_bw", "1")
        return _block(
            f"{target} = BinaryMap(graph, {a}, {b}, fn={fn_str}, "
            f"write_back_mu=False, compute_bw={compute_bw})\n"
        )
    return handler
```

`_make_unary_map` (replace):

```python
def _make_unary_map(map_class):
    def handler(state, target, call):
        x = _src(_arg(call, 0, "x"))
        if map_class in ("MulImmediate", "AddImmediate", "SubImmediate", "ToConstInt"):
            c = _src(_arg(call, 1, "constant"))
            fn_str = f"map_fn.{map_class}({c})"
            compute_bw = _arg_or_default(call, 2, "compute_bw", "1")
        else:
            fn_str = f"map_fn.{map_class}()"
            compute_bw = _arg_or_default(call, 1, "compute_bw", "1")
        return _block(
            f"{target} = UnaryMap(graph, {x}, fn={fn_str}, "
            f"write_back_mu=False, compute_bw={compute_bw})\n"
        )
    return handler
```

`_make_accum` (replace):

```python
def _make_accum(accum_class, mode):
    def handler(state, target, call):
        x = _src(_arg(call, 0, "x"))
        rank = _arg_or_default(call, 1, "rank", "1")
        compute_bw = _arg_or_default(call, 2, "compute_bw", "1")
        return _block(
            f"{target} = Accum(graph, {x}, "
            f"output_stream_dtype=_dsl2step_out_tile({x}, {mode!r}, {rank}), "
            f"fn=accum_fn.{accum_class}(), init_fn=_dsl2step_init({x}), "
            f"accum_rank={rank}, write_back_mu=False, compute_bw={compute_bw})\n"
        )
    return handler
```

- [ ] **Step 2.4: Forward `compute_bw` from the four special-case compute handlers**

Replace `_h_binary_cache_write_addr_gen`:

```python
def _h_binary_cache_write_addr_gen(state, target, call):
    idx        = _src(_arg(call, 0, "idx"))
    seq_len    = _src(_arg(call, 1, "seq_len"))
    row_offset = _src(_arg(call, 2, "row_offset"))
    compute_bw = _arg_or_default(call, 3, "compute_bw", "1")
    return _block(
        f"{target} = BinaryMap(graph, {idx}, {seq_len}, "
        f"fn=map_fn.CacheWriteAddrGen(row_offset={row_offset}), "
        f"write_back_mu=False, compute_bw={compute_bw})\n"
    )
```

Replace `_h_unary_mask_row`:

```python
def _h_unary_mask_row(state, target, call):
    x = _src(_arg(call, 0, "x"))
    compute_bw = _arg_or_default(call, 1, "compute_bw", "1")
    return _block(
        f"{target} = UnaryMap(graph, {x}, "
        f"fn=map_fn.MaskRow(tile=_dsl2step_in_tile({x})), "
        f"write_back_mu=False, compute_bw={compute_bw})\n"
    )
```

Replace `_h_accum_signal_req_all_read`:

```python
def _h_accum_signal_req_all_read(state, target, call):
    x    = _src(_arg(call, 0, "x"))
    rank = _arg_or_default(call, 1, "rank", "1")
    compute_bw = _arg_or_default(call, 2, "compute_bw", "1")
    return _block(
        f"{target} = Accum(graph, {x}, "
        f"output_stream_dtype=Tile(tile_dtype=Uint64(), shape=(1, 1)), "
        f"fn=accum_fn.SignalReqAllRead(), "
        f"init_fn=Empty(shape=(1, 1), dtype=Uint64()), "
        f"accum_rank={rank}, write_back_mu=False, compute_bw={compute_bw})\n"
    )
```

Replace `_h_binary_map_accum`:

```python
def _h_binary_map_accum(state, target, call):
    a = _src(_arg(call, 0, "a"))
    b = _src(_arg(call, 1, "b"))
    rank = _arg_or_default(call, 2, "rank", "1")
    wt = _arg(call, 3, "weight_transposed")
    wt_s = f"weight_transposed={_src(wt)}" if wt is not None else ""
    compute_bw = _arg_or_default(call, 4, "compute_bw", "1")
    return _block(
        f"{target} = BinaryMapAccum(graph, {a}, {b}, "
        f"fn=map_accum_fn.Matmul({wt_s}), init_fn=_dsl2step_init({a}), "
        f"rank={rank}, write_back_mu=False, compute_bw={compute_bw})\n"
    )
```

- [ ] **Step 2.5: Forward `par_dispatch` from the six memory handlers**

Replace `_h_offchip_load`:

```python
def _h_offchip_load(state, target, call):
    underlying = _src(_arg(call, 0, "underlying"))
    stride     = _src(_arg(call, 1, "stride"))
    out_shape  = _src(_arg(call, 2, "out_shape_tiled"))
    tile_row   = _src(_arg(call, 3, "tile_row"))
    tile_col   = _src(_arg(call, 4, "tile_col"))
    transposed = _arg(call, 5, "transposed")
    par_dispatch = _arg_or_default(call, 6, "par_dispatch", "1")
    extra = f", transposed={_src(transposed)}" if transposed is not None else ""
    return _block(
        f"{target} = LinearOffChipLoad({underlying}, stride={stride}, "
        f"out_shape_tiled={out_shape}, tile_row={tile_row}, tile_col={tile_col}, "
        f"par_dispatch={par_dispatch}{extra})\n"
        f"graph.add_node({target})\n"
    )
```

Replace `_h_offchip_load_ref`:

```python
def _h_offchip_load_ref(state, target, call):
    ref        = _src(_arg(call, 0, "ref"))
    underlying = _src(_arg(call, 1, "underlying"))
    stride     = _src(_arg(call, 2, "stride"))
    out_shape  = _src(_arg(call, 3, "out_shape_tiled"))
    tile_row   = _src(_arg(call, 4, "tile_row"))
    tile_col   = _src(_arg(call, 5, "tile_col"))
    transposed = _arg(call, 6, "transposed")
    par_dispatch = _arg_or_default(call, 7, "par_dispatch", "1")
    extra = f", transposed={_src(transposed)}" if transposed is not None else ""
    return _block(
        f"{target} = LinearOffChipLoadRef(graph, ref={ref}, "
        f"underlying={underlying}, stride={stride}, out_shape_tiled={out_shape}, "
        f"tile_row={tile_row}, tile_col={tile_col}, par_dispatch={par_dispatch}{extra})\n"
    )
```

Replace `_h_random_offchip_load`:

```python
def _h_random_offchip_load(state, target, call):
    underlying     = _src(_arg(call, 0, "underlying"))
    raddr          = _src(_arg(call, 1, "raddr"))
    tile_row       = _src(_arg(call, 2, "tile_row"))
    tile_col       = _src(_arg(call, 3, "tile_col"))
    base_addr_byte = _arg_or_default(call, 4, "base_addr_byte", "0")
    transposed     = _arg(call, 5, "transposed")
    par_dispatch   = _arg_or_default(call, 6, "par_dispatch", "1")
    extra = f", transposed={_src(transposed)}" if transposed is not None else ""
    return _block(
        f"{target} = RandomOffChipLoad(graph, underlying={underlying}, "
        f"raddr={raddr}, tile_row={tile_row}, tile_col={tile_col}, "
        f"base_addr_byte={base_addr_byte}, par_dispatch={par_dispatch}{extra})\n"
    )
```

Replace `_h_dyn_offchip_load`:

```python
def _h_dyn_offchip_load(state, target, call):
    underlying_node = _arg(call, 0, "underlying")
    assert (
        isinstance(underlying_node, ast.Subscript)
        and isinstance(underlying_node.value, ast.Name)
        and underlying_node.value.id == "tensors"
        and isinstance(underlying_node.slice, ast.Constant)
        and isinstance(underlying_node.slice.value, str)
    ), (
        "dyn_offchip_load: underlying must be tensors['<name>'], got "
        f"{ast.dump(underlying_node) if underlying_node is not None else 'None'}"
    )
    name_str = underlying_node.slice.value
    underlying = _src(underlying_node)
    tensor_shape_tiled = _src(_arg(call, 1, "tensor_shape_tiled"))
    tile_row = _src(_arg(call, 2, "tile_row"))
    tile_col = _src(_arg(call, 3, "tile_col"))
    par_dispatch = _arg_or_default(call, 4, "par_dispatch", "1")
    return _block(
        f"{target} = DynLinearOffChipLoad("
        f"input_tensor_name={name_str!r}, "
        f"tensor_shape_tiled={tensor_shape_tiled}, "
        f"dtype={underlying}.dtype, "
        f"tile_row={tile_row}, tile_col={tile_col}, par_dispatch={par_dispatch})\n"
        f"graph.add_node({target})\n"
    )
```

Replace `_h_offchip_store`:

```python
def _h_offchip_store(state, target, call):
    x = _src(_arg(call, 0, "x"))
    par_dispatch = _arg_or_default(call, 1, "par_dispatch", "1")
    return _block(f"{target} = OffChipStore(graph, {x}, par_dispatch={par_dispatch})\n")
```

Replace `_h_random_offchip_store`:

```python
def _h_random_offchip_store(state, target, call):
    underlying     = _src(_arg(call, 0, "underlying"))
    wdata          = _src(_arg(call, 1, "wdata"))
    waddr          = _src(_arg(call, 2, "waddr"))
    tile_row       = _src(_arg(call, 3, "tile_row"))
    tile_col       = _src(_arg(call, 4, "tile_col"))
    base_addr_byte = _arg_or_default(call, 5, "base_addr_byte", "0")
    par_dispatch   = _arg_or_default(call, 6, "par_dispatch", "1")
    return _block(
        f"{target} = RandomOffChipStore(graph, underlying={underlying}, "
        f"wdata={wdata}, waddr={waddr}, tile_row={tile_row}, "
        f"tile_col={tile_col}, base_addr_byte={base_addr_byte}, par_dispatch={par_dispatch})\n"
    )
```

- [ ] **Step 2.6: Forward `par_dispatch` from the offchip_store-in-return branch of `_rewrite_return`**

In `src/dsl_to_step.py`, replace the body of `_State._rewrite_return` (currently around line 530) to read the kwarg off the in-return Call:

```python
def _rewrite_return(self, stmt):
    # Two supported forms at the end of tiled_reference:
    #   return offchip_store(x)               (with optional par_dispatch=N kwarg)
    #   return out                            (out was assigned earlier)
    if (isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == "offchip_store"):
        x = _src(stmt.value.args[0])
        par_dispatch = _arg_or_default(stmt.value, 1, "par_dispatch", "1")
        store_var = self.fresh("store")
        return _block(
            f"{store_var} = OffChipStore(graph, {x}, par_dispatch={par_dispatch})\n"
            f"graph = infer_broadcast(graph)\n"
            f"return graph, {store_var}\n"
        )
    out = _src(stmt.value)
    return _block(
        f"graph = infer_broadcast(graph)\n"
        f"return graph, {out}\n"
    )
```

- [ ] **Step 2.7: Update the module docstring**

In `src/dsl_to_step.py`, replace the existing module docstring with:

```python
"""Deterministic translator: DSL-refactored ``tiled_reference`` -> STeP ``build_graph``.

Replaces the LLM ``translate`` pass. Each DSL call in
``StepGenFlow9/src/step_dsl.py`` maps to one STeP IR node construction.

Perf-knob kwargs on the DSL surface (``compute_bw=N`` on compute calls,
``par_dispatch=N`` on off-chip memory calls) are read off the AST and forwarded
to the STeP node constructor. Missing kwargs default to 1, preserving
byte-for-byte translator output for any DSL source that does not pass them.

Public API:
    translate(dsl_code: str) -> str

The output is a Python source string containing two top-level definitions:
    * a small set of helper functions used by the generated graph builder
    * ``def build_graph(dims, tensors)`` returning ``(graph, output_op)``

The output is intended to be exec'd under ``IMPORT_SCAFFOLD`` and run via
``execute(graph, output_op)`` -- exactly the harness used by
``_run_graph_correctness`` in ``orchestrator.py``.
"""
```

- [ ] **Step 2.8: Run tests and verify they pass**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/test_dsl_to_step_knobs.py -v`
Expected: PASS — 6 tests pass.

- [ ] **Step 2.9: Run the full test suite to confirm no regressions**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/ -x`
Expected: existing tests still pass; only the 6 new tests are added to the count. Crucially, any existing test that exercises `translate()` against a DSL source without new kwargs continues to pass — the BC promise.

- [ ] **Step 2.10: Commit**

```bash
cd /workspace/DEIOpt && git add StepGenFlow9/src/dsl_to_step.py StepGenFlow9/tests/test_dsl_to_step_knobs.py
git commit -m "$(cat <<'EOF'
forward compute_bw / par_dispatch through the DSL translator

Each handler whose STeP ctor carries the knob now extracts it from the DSL
AST via _arg_or_default with a default of 1. Default-of-1 preserves
byte-for-byte translator output for any DSL source that does not pass the
new kwargs.
EOF
)"
```

---

### Task 3: Swap autotune prompts to embed step_dsl.py

**Files:**
- Modify: `src/prompts.py` — path constant + loader changes + user-prompt wording.
- Modify: `prompts/autotune_system.txt` — placeholders + narrative.
- Modify: `prompts/autotune_parallel_system.txt` — placeholders + narrative.

- [ ] **Step 3.1: Add `_STEP_DSL_PY` path constant in `prompts.py`**

In `src/prompts.py`, in the `# Path constants` block (around line 30), add immediately after `_TIMING_PY`:

```python
_STEP_DSL_PY = _PROJECT_ROOT / "src" / "step_dsl.py"
```

- [ ] **Step 3.2: Rework the loader in `build_autotune_system_prompt`**

In `src/prompts.py`, replace the body of `build_autotune_system_prompt` (currently around line 301) with:

```python
def build_autotune_system_prompt(hw_constraints: dict,
                                  template_name: str = "autotune_system.txt") -> str:
    """Build the autotuner agent's system prompt.

    Embeds the DSL surface (step_dsl.py) at {step_dsl_code} and the timing
    model source (timing.py) at {timing_code}, with caller-supplied
    `hw_constraints` (rendered as JSON) at {hw_constraints}. The autotuner
    edits ``tiled_reference(dims, tensors)`` (DSL form) — the deterministic
    translator runs inside its per-turn loop to produce the build_graph the
    timing model scores.

    `template_name` selects which autotune prompt to use (e.g.,
    "autotune_system.txt" for the generalist, "autotune_parallel_system.txt"
    for the parallelism specialist). All templates share the same placeholder
    set so injection logic is identical.
    """
    template_path = _PROMPTS_DIR / template_name
    assert template_path.exists(), f"autotune prompt not found: {template_path}"
    assert _TIMING_PY.exists(), f"timing.py not found: {_TIMING_PY}"
    assert _STEP_DSL_PY.exists(), f"step_dsl.py not found: {_STEP_DSL_PY}"
    template = template_path.read_text()
    replacements = {}
    if "{step_dsl_code}" in template:
        replacements["step_dsl_code"] = _STEP_DSL_PY.read_text()
    if "{timing_code}" in template:
        replacements["timing_code"] = _TIMING_PY.read_text()
    if "{hw_constraints}" in template:
        replacements["hw_constraints"] = json.dumps(hw_constraints, indent=2)
    return template.format(**replacements)
```

- [ ] **Step 3.3: Update `build_autotune_user_prompt` wording**

In `src/prompts.py`, replace the two strings in `build_autotune_user_prompt` (around line 338):

- `"### Current build_graph (correctness verified)"` → `"### Current tiled_reference (correctness verified)"`
- the closing line `"... Output the full updated `build_graph(dims, tensors)` in a single ```python block."` → `"... Output the full updated `tiled_reference(dims, tensors)` in a single ```python block."`

The full replacement:

```python
def build_autotune_user_prompt(kernel_name: str, dims: dict, build_graph_code: str,
                                timing_report: str, baseline_cycles: int,
                                best_cycles: int) -> str:
    """Build the autotuner user prompt for a single turn.

    `build_graph_code` parameter name is kept for back-compat at the call site,
    but the body is now the DSL-form ``tiled_reference`` source.
    `timing_report` is the pretty-printed analyze_timing() output for the
    *translated* build_graph. `baseline_cycles` is the cycle count at the start
    of the tuning run; `best_cycles` is the best we've seen so far.
    """
    return "\n".join([
        f"## Kernel: {kernel_name}",
        "",
        "### Dimensions",
        "",
        "```json",
        json.dumps(dims, indent=2),
        "```",
        "",
        f"### Baseline total_cycles: {baseline_cycles}",
        f"### Best so far:          {best_cycles}",
        "",
        "### Current tiled_reference (correctness verified)",
        "",
        "```python",
        build_graph_code.rstrip(),
        "```",
        "",
        "### Current timing report",
        "",
        "```",
        timing_report.rstrip(),
        "```",
        "",
        "Propose a change that reduces total_cycles. Output the full updated "
        "`tiled_reference(dims, tensors)` in a single ```python block.",
    ])
```

- [ ] **Step 3.4: Rewrite `prompts/autotune_system.txt`**

Replace the entire contents of `prompts/autotune_system.txt` with:

```
You are an autotuner for STeP dataflow graphs.

You will be given a `tiled_reference(dims, tensors)` function written in the
STeP DSL that is already verified correct (its output matches the PyTorch
reference). Your job is to rewrite it so the lowered STeP graph runs faster
on the target hardware, while keeping its output numerically identical. You
can change the tiling scheme, add parallel computations, reallocate compute
resources, reduce memory traffic, and/or make larger structural changes to
the DSL graph.

Each proposal is run through a deterministic translator (DSL ->
`build_graph`) so the analytical timing model can score it; you do not write
`build_graph` directly.

## DSL surface (step_dsl.py)
```python
{step_dsl_code}
```

## Perf-knob convention

Each DSL function whose lowered STeP node carries a perf knob accepts that
knob as a keyword-only argument with default 1. Specifically:

- compute DSL calls (`binary_*`, `unary_*`, `accum_*`, `binary_map_accum`)
  accept `compute_bw=N`.
- off-chip DSL calls (`offchip_load*`, `dyn_offchip_load`, `random_offchip_*`,
  `offchip_store`) accept `par_dispatch=N`.

The kwarg is ignored at DSL eval time and consumed by the deterministic
translator that produces `build_graph` for the timing model. Use these to
express the relative compute share / dispatch parallelism you want each call
to receive.

## Hardware constraints

The target has the following fixed resources. Your configuration must
respect them:

{hw_constraints}

Budget rules:
- After you emit the DSL, the autotuner translates it and runs a
  post-scaling pass that rescales every compute DSL call's `compute_bw` by a
  uniform factor so the sum of `compute_bw` across all compute ops in the
  translated graph equals `max_total_compute_bw` exactly (with each value
  floored at 1). As a consequence, only the *ratios* between per-call
  `compute_bw` values matter — the absolute numbers you write are scaled
  away. Use `compute_bw` to express the relative compute share you want
  each call to receive. The timing report you receive reflects post-scaled
  values, and includes a summary of the rescaling so you can see what each
  call ended up with.
- No off-chip DSL call may have `par_dispatch > max_par_dispatch`.

## Performance model

You will receive, after each proposal, a breakdown from the analytical
timing model below. Read it. It tells you `total_cycles`, per-node start
and end times, OCI (output cycle interval — cycles between produced
tiles), and OTI (per-element rate). The node with the highest `end` is
the current bottleneck. Focus your next change on relieving that
bottleneck; do not globally rewrite. Per-node names in the report
correspond to STeP IR node classes (e.g. BinaryMap, UnaryMap, Accum,
LinearOffChipLoad) — map them back to your DSL calls via the function
names listed in step_dsl.py.

For reference, this is the full source of the timing model — study it to
understand exactly how each knob affects total_cycles:

```python
{timing_code}
```

## Output format

Respond with a short paragraph explaining what bottleneck you are
targeting and which knob(s) / DSL call(s) you are changing, then a single
```python fenced block containing the full updated
`tiled_reference(dims, tensors)` function. Do NOT include import statements
— they are injected.

If you believe the current DSL is already optimal given the constraints,
say so and output the unchanged `tiled_reference` anyway so the loop can
terminate.
```

- [ ] **Step 3.5: Rewrite `prompts/autotune_parallel_system.txt`**

Replace the entire contents of `prompts/autotune_parallel_system.txt` with:

```
You are a parallelism specialist for STeP dataflow graphs.

You will be given a `tiled_reference(dims, tensors)` function written in the
STeP DSL that is already verified correct (its output matches the PyTorch
reference). Your job is to rewrite it so the lowered STeP graph runs faster
by introducing graph-level parallelism, while keeping its output
numerically identical.

You may only do this by inserting, removing, or retuning `parallelize`
calls (which fan a stream out into `n` independent parallel copies) and
their inverse `static_reassemble` calls (which merge the parallel streams
back into one). Do not change any other knob (tiling, `par_dispatch`,
`compute_bw`, `write_back_mu`, etc.) and do not alter the underlying
algorithm.

Each proposal is run through a deterministic translator (DSL ->
`build_graph`) so the analytical timing model can score it; you do not write
`build_graph` directly.

## DSL surface (step_dsl.py)
```python
{step_dsl_code}
```

## Hardware constraints

The target has the following fixed resources. Your configuration must
respect them:

{hw_constraints}

Budget rules:
- After you emit the DSL, the autotuner translates it and runs a
  post-scaling pass that rescales every compute op's `compute_bw` by a
  uniform factor so the sum of `compute_bw` across all compute ops in the
  translated graph equals `max_total_compute_bw` exactly (with each value
  floored at 1). As a consequence, only the *ratios* between per-op
  `compute_bw` values matter — the absolute numbers you write are scaled
  away. Note that parallelizing a subgraph replicates its compute ops `n`
  times, so all replicas participate in the rescaling and shrink each
  replica's share accordingly. The timing report you receive reflects
  post-scaled values, and includes a summary of the rescaling so you can
  see what each op ended up with.
- No off-chip op may have `par_dispatch > max_par_dispatch`.

## Performance model

You will receive, after each proposal, a breakdown from the analytical
timing model below. Read it. It tells you `total_cycles`, per-node start
and end times, OCI (output cycle interval — cycles between produced
tiles), and OTI (per-element rate). The node with the highest `end` is
the current bottleneck. Focus your next change on relieving that
bottleneck; do not globally rewrite. Per-node names in the report
correspond to STeP IR node classes (e.g. BinaryMap, Parallelize,
StaticReassemble) — map them back to your DSL calls via the function
names listed in step_dsl.py.

For reference, this is the full source of the timing model — study it to
understand exactly how each knob affects total_cycles:

```python
{timing_code}
```

## Output format

Respond with a short paragraph explaining what bottleneck you are
targeting and which `parallelize` / `static_reassemble` placement you are
changing, then a single ```python fenced block containing the full updated
`tiled_reference(dims, tensors)` function. Do NOT include import
statements — they are injected.

If you believe the current DSL is already optimal given the constraints,
say so and output the unchanged `tiled_reference` anyway so the loop can
terminate.
```

- [ ] **Step 3.6: Run the full test suite to confirm no regressions**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/ -x`
Expected: existing tests pass (the prompt change does not affect any unit test).

- [ ] **Step 3.7: Commit**

```bash
cd /workspace/DEIOpt && git add \
    StepGenFlow9/src/prompts.py \
    StepGenFlow9/prompts/autotune_system.txt \
    StepGenFlow9/prompts/autotune_parallel_system.txt
git commit -m "$(cat <<'EOF'
swap autotune prompts to embed step_dsl.py instead of ops + functional

Both the general and parallel autotune system prompts now describe the DSL
surface that the autotuner edits. The placeholder set shrinks to
{step_dsl_code} + {timing_code} + {hw_constraints}; ops.py / utility_ops.py
/ functional.py are no longer embedded. The user prompt's headers and
closing instruction are reworded to reference tiled_reference.
EOF
)"
```

---

### Task 4: Add `_evaluate_dsl_turn` helper in autotune.py

**Files:**
- Modify: `src/autotune.py` — add helper, add imports.
- Create: `tests/test_autotune_loop.py` — gate ladder tests.

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_autotune_loop.py`:

```python
"""Per-turn gate ladder for the DSL-form autotuner."""

from pathlib import Path

import pytest

from src import autotune as autotune_mod


# A correct, knob-free DSL source we can use as a baseline. tile sizes are
# kept tiny so the IR sim runs cheaply.
_VALID_DSL = '''
def tiled_reference(dims, tensors):
    a = offchip_load(tensors["A"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4)
    b = offchip_load(tensors["B"], stride=(1,), out_shape_tiled=(2,),
                      tile_row=4, tile_col=4)
    c = binary_matmul(a, b)
    return offchip_store(c)
'''


@pytest.fixture
def stub_gates(monkeypatch):
    """Patch the four gate functions to controllable stubs."""
    state = {
        "dsl_correctness": "match=True",
        "translate_raises": False,
        "graph_correctness": "match=True",
        "measure_raises": False,
        "measure_result": (1234, "TIMING REPORT"),
    }

    def fake_dsl(code, kernel_name, dims, tensors):
        if isinstance(state["dsl_correctness"], Exception):
            raise state["dsl_correctness"]
        return state["dsl_correctness"]

    def fake_translate(code):
        if state["translate_raises"]:
            raise RuntimeError("translate boom")
        return "TRANSLATED:" + code

    def fake_graph(code, kernel_name, dims, tensors):
        if isinstance(state["graph_correctness"], Exception):
            raise state["graph_correctness"]
        return state["graph_correctness"]

    def fake_measure(code, kernel_name, dims, tensors, hw_config, max_total_compute_bw):
        if state["measure_raises"]:
            raise RuntimeError("timing boom")
        return state["measure_result"]

    monkeypatch.setattr(autotune_mod, "_run_dsl_correctness", fake_dsl)
    monkeypatch.setattr(autotune_mod, "translate", fake_translate)
    monkeypatch.setattr(autotune_mod, "_run_graph_correctness", fake_graph)
    monkeypatch.setattr(autotune_mod, "_measure", fake_measure)
    return state


def _evaluate(**kw):
    return autotune_mod._evaluate_dsl_turn(
        dsl_code=_VALID_DSL,
        kernel_name="dummy",
        dims={},
        tensors={},
        hw_config={},
        max_total_compute_bw=128,
    )


def test_clean_pass_status_and_artifacts(stub_gates):
    out = _evaluate()
    assert out["status"] == "PASS"
    assert out["dsl_correctness_text"] == "match=True"
    assert out["translated_code"].startswith("TRANSLATED:")
    assert out["graph_correctness_text"] == "match=True"
    assert out["new_cycles"] == 1234
    assert out["new_report"] == "TIMING REPORT"
    assert out["translate_error_text"] is None
    assert out["timing_error_text"] is None


def test_dsl_fail_short_circuits_at_gate_1(stub_gates):
    stub_gates["dsl_correctness"] = "match=False mismatch=...sample..."
    out = _evaluate()
    assert out["status"] == "DSL_FAIL"
    assert out["dsl_correctness_text"].startswith("match=False")
    # Later gates did not run.
    assert out["translated_code"] is None
    assert out["graph_correctness_text"] is None
    assert out["new_cycles"] is None


def test_dsl_exec_raise_short_circuits_at_gate_1(stub_gates):
    stub_gates["dsl_correctness"] = RuntimeError("boom")
    out = _evaluate()
    assert out["status"] == "DSL_FAIL"
    assert "RuntimeError: boom" in out["dsl_correctness_text"]
    assert out["translated_code"] is None


def test_translate_error_short_circuits_at_gate_2(stub_gates):
    stub_gates["translate_raises"] = True
    out = _evaluate()
    assert out["status"] == "TRANSLATE_ERROR"
    assert out["dsl_correctness_text"] == "match=True"
    assert "translate boom" in out["translate_error_text"]
    assert out["translated_code"] is None
    assert out["graph_correctness_text"] is None


def test_ir_fail_short_circuits_at_gate_3(stub_gates):
    stub_gates["graph_correctness"] = "match=False sim_output=..."
    out = _evaluate()
    assert out["status"] == "IR_FAIL"
    assert out["dsl_correctness_text"] == "match=True"
    assert out["translated_code"].startswith("TRANSLATED:")
    assert out["graph_correctness_text"].startswith("match=False")
    assert out["new_cycles"] is None


def test_timing_error_short_circuits_at_step_4(stub_gates):
    stub_gates["measure_raises"] = True
    out = _evaluate()
    assert out["status"] == "TIMING_ERROR"
    assert "timing boom" in out["timing_error_text"]
    assert out["new_cycles"] is None
```

- [ ] **Step 4.2: Run the test and verify it fails**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/test_autotune_loop.py -v`
Expected: FAIL — `AttributeError: module 'src.autotune' has no attribute '_evaluate_dsl_turn'` (or `_run_dsl_correctness` / `translate` not present in the module).

- [ ] **Step 4.3: Add the helper and required imports**

In `src/autotune.py`:

1. Update the imports block. Replace this line (around line 49):

```python
from src.orchestrator import _run_graph_correctness, _write, _extract_code, _reasoning_text  # noqa: E402
```

with:

```python
from src.orchestrator import (  # noqa: E402
    _run_dsl_correctness,
    _run_graph_correctness,
    _write,
    _extract_code,
    _reasoning_text,
    _resolve_resume_dsl,
)
from src.dsl_to_step import translate  # noqa: E402
```

2. Add the helper. Insert this function **immediately above** `def _measure(` (currently around line 503). It re-uses the imported gate functions and the existing `_measure`:

```python
def _evaluate_dsl_turn(
    dsl_code: str,
    kernel_name: str,
    dims: dict,
    tensors: dict,
    hw_config: dict,
    max_total_compute_bw: int,
) -> dict:
    """Run the triple-gate (DSL -> translate -> IR) chain plus the timing model.

    Each gate is independent; we short-circuit at the first failure so later
    artifacts are absent in that case (which the caller relies on to choose
    feedback for the next turn). Exceptions raised by the gate functions are
    captured into the corresponding text field so the LLM gets a full
    traceback rather than a bare status.
    """
    out = {
        "status": None,
        "dsl_correctness_text": None,
        "translated_code": None,
        "translate_error_text": None,
        "graph_correctness_text": None,
        "timing_error_text": None,
        "new_cycles": None,
        "new_report": None,
    }

    # Gate 1: DSL exec vs gold.
    try:
        dsl_text = _run_dsl_correctness(dsl_code, kernel_name, dims, tensors)
    except Exception:
        dsl_text = "ERROR:\n" + traceback.format_exc()
    out["dsl_correctness_text"] = dsl_text
    if "match=True" not in dsl_text:
        out["status"] = "DSL_FAIL"
        return out

    # Gate 2: deterministic translate.
    try:
        translated = translate(dsl_code)
    except Exception:
        out["translate_error_text"] = traceback.format_exc()
        out["status"] = "TRANSLATE_ERROR"
        return out
    out["translated_code"] = translated

    # Gate 3: IR sim vs gold.
    try:
        graph_text = _run_graph_correctness(translated, kernel_name, dims, tensors)
    except Exception:
        graph_text = "ERROR:\n" + traceback.format_exc()
    out["graph_correctness_text"] = graph_text
    if "match=True" not in graph_text:
        out["status"] = "IR_FAIL"
        return out

    # Step 4: timing model on the translated graph.
    try:
        new_cycles, new_report = _measure(
            translated, kernel_name, dims, tensors, hw_config, max_total_compute_bw,
        )
    except Exception:
        out["timing_error_text"] = traceback.format_exc()
        out["status"] = "TIMING_ERROR"
        return out

    out["new_cycles"] = new_cycles
    out["new_report"] = new_report
    out["status"] = "PASS"
    return out
```

- [ ] **Step 4.4: Run the test and verify it passes**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/test_autotune_loop.py -v`
Expected: PASS — 6 tests pass.

- [ ] **Step 4.5: Run the full test suite**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/ -x`
Expected: existing tests still pass. The helper is additive at this point — `run_autotune` still uses the old loop body.

- [ ] **Step 4.6: Commit**

```bash
cd /workspace/DEIOpt && git add StepGenFlow9/src/autotune.py StepGenFlow9/tests/test_autotune_loop.py
git commit -m "$(cat <<'EOF'
add _evaluate_dsl_turn helper to autotune.py

Pure helper that runs the triple-gate chain (DSL exec -> translate -> IR
sim) plus the timing-model step against a candidate DSL source. Short-
circuits on the first failing gate; returns a dict naming the gate that
failed and the full text of every gate that ran. Will replace the inline
gate code in run_autotune in the next change.
EOF
)"
```

---

### Task 5: Wire `run_autotune` to the triple-gate chain and DSL resume

**Files:**
- Modify: `src/autotune.py` — replace `_resolve_resume_build_graph` with `_resolve_resume_dsl`; rewrite baseline measurement; rewrite the per-turn loop body to call `_evaluate_dsl_turn`; rewrite the artifact writes; update status vocabulary; update best-tracking to persist DSL source.
- Modify: `tests/test_autotune_loop.py` — add a resume-resolution test.

- [ ] **Step 5.1: Add a test for the autotune resume resolution**

Append to `tests/test_autotune_loop.py`:

```python
def test_run_autotune_reads_dsl_code_from_outer_dir(tmp_path):
    """The autotune entry point's resume path resolves to outer_<N>/dsl_code.py."""
    outer_dir = tmp_path / "outer_0"
    outer_dir.mkdir()
    (outer_dir / "dsl_code.py").write_text("# placeholder dsl source\n")

    code, src = autotune_mod._resolve_resume_dsl_with_source(
        str(outer_dir), kernel_name="dummy",
    )
    assert code == "# placeholder dsl source\n"
    assert src == outer_dir / "dsl_code.py"


def test_run_autotune_rejects_path_with_no_dsl_code(tmp_path):
    """Pointing at a directory with no dsl_code.py fails loudly."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises((AssertionError, FileNotFoundError)):
        autotune_mod._resolve_resume_dsl_with_source(
            str(empty_dir), kernel_name="dummy",
        )
```

- [ ] **Step 5.2: Run the new tests and verify they fail**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/test_autotune_loop.py::test_run_autotune_reads_dsl_code_from_outer_dir tests/test_autotune_loop.py::test_run_autotune_rejects_path_with_no_dsl_code -v`
Expected: FAIL — `_resolve_resume_dsl_with_source` doesn't exist yet.

- [ ] **Step 5.3: Add `_resolve_resume_dsl_with_source` helper**

In `src/autotune.py`, **delete** `_find_passing_extract` and `_resolve_resume_build_graph` (currently around line 56-100) and **replace** them with the following helper. It mirrors `orchestrator._resolve_resume_dsl` but also returns the source path so the autotune `config.json` can record `resume_resolved_to`:

```python
def _resolve_resume_dsl_with_source(resume_from: str, kernel_name: str) -> tuple[str, Path]:
    """Resolve `resume_from` to (dsl_code, source_path).

    Same path semantics as ``orchestrator._resolve_resume_dsl`` but returns
    the resolved file's Path so the autotune config.json can record where
    the baseline came from.
    """
    p = Path(resume_from)
    assert p.exists(), f"resume_from path does not exist: {p}"

    if p.suffix == ".py" and p.is_file():
        return p.read_text(), p

    if p.is_dir() and (p / "dsl_code.py").is_file():
        chosen = p / "dsl_code.py"
        return chosen.read_text(), chosen

    if p.is_dir():
        candidates = sorted((p / kernel_name).glob("outer_*/dsl_code.py"))
        assert candidates, (
            f"No dsl_code.py found under {p / kernel_name}/outer_*/. "
            f"Ensure refactor_final succeeded in the checkpoint you're resuming from."
        )
        chosen = candidates[0]
        print(f"  Resolved resume path: {chosen}")
        return chosen.read_text(), chosen

    raise FileNotFoundError(
        f"Cannot resolve resume_from='{resume_from}'. "
        f"Expected a .py file, a directory with dsl_code.py, "
        f"or a checkpoint root directory."
    )
```

- [ ] **Step 5.4: Run the resume tests and verify they pass**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/test_autotune_loop.py::test_run_autotune_reads_dsl_code_from_outer_dir tests/test_autotune_loop.py::test_run_autotune_rejects_path_with_no_dsl_code -v`
Expected: PASS — both tests pass.

- [ ] **Step 5.5: Rewrite `run_autotune` body**

In `src/autotune.py`, replace the body of `run_autotune` (currently around line 544-733) with the version below. This:

- Calls the new resolver.
- Hard-fails on a baseline that doesn't pass DSL exec, translate, or IR sim.
- Writes `baseline_dsl.py` + `baseline_translated.py` + `baseline_timing.txt`.
- Builds each turn's prompt from `current_dsl_code`.
- Calls `_evaluate_dsl_turn` and dispatches on its `status`.
- Writes per-turn artifacts as named in the spec.
- Updates `current_dsl_code` only on full PASS so failures don't cascade.
- Persists `best.py` (DSL source) + `best_translated.py` (build_graph) + `best_timing.txt`.

```python
async def run_autotune(
    kernel_name: str,
    preset: str,
    llm_config: dict,
    autotune_config: dict,
    resume_from: str,
    max_turns: int = None,
    checkpoint_dir: str = None,
    agent_variant: str = "general",
) -> dict:
    """Autotune a verified DSL ``tiled_reference`` starting from a past checkpoint.

    Args:
        autotune_config: dict with keys ``hw_config``, ``constraints``,
            ``max_turns`` (see autotune_config.json).
        resume_from: path to the successful implementer checkpoint — a file,
            outer dir, or checkpoint root. See ``_resolve_resume_dsl_with_source``.
        max_turns: overrides ``autotune_config["max_turns"]`` if provided.
        agent_variant: which autotuner agent to run — ``"general"`` (default)
            or ``"parallel"`` (Parallelize/StaticReassemble specialist).
    """
    hw_config = autotune_config["hw_config"]
    constraints = autotune_config["constraints"]
    max_total_compute_bw = constraints["max_total_compute_bw"]
    if max_turns is None:
        max_turns = autotune_config.get("max_turns", 8)

    # Resolve dims from StepDB
    config = _load_stepdb_config()
    assert kernel_name in config, f"Kernel '{kernel_name}' not found"
    assert preset in config[kernel_name]["presets"], f"Preset '{preset}' not found"
    dims = config[kernel_name]["presets"][preset]

    # Load the baseline DSL from the resume checkpoint.
    baseline_dsl_code, baseline_src = _resolve_resume_dsl_with_source(
        resume_from, kernel_name)
    print(f"Loaded baseline DSL ({len(baseline_dsl_code)} chars) from {baseline_src}")

    # Precompute tensors (same call used by the implementer pipeline).
    tensors = precompute_tensors(kernel_name, dims)
    print(f"Pre-computed tensors: {sorted(tensors.keys())}")

    # Verify baseline through every gate up front — required invariant.
    eval_baseline = _evaluate_dsl_turn(
        baseline_dsl_code, kernel_name, dims, tensors, hw_config, max_total_compute_bw)
    assert eval_baseline["status"] == "PASS", (
        f"Baseline DSL from {baseline_src} did not pass all gates: "
        f"status={eval_baseline['status']}\n\n"
        f"DSL correctness:\n{eval_baseline['dsl_correctness_text']}\n\n"
        f"Translate error:\n{eval_baseline['translate_error_text']}\n\n"
        f"Graph correctness:\n{eval_baseline['graph_correctness_text']}\n\n"
        f"Timing error:\n{eval_baseline['timing_error_text']}"
    )
    baseline_translated = eval_baseline["translated_code"]
    baseline_cycles = eval_baseline["new_cycles"]
    baseline_report = eval_baseline["new_report"]
    print(f"Baseline total_cycles = {baseline_cycles}")

    # Merge hw_config + constraints for the system prompt's {hw_constraints} block.
    prompt_constraints = {**hw_config, **constraints}

    # Create the autotuner agent.
    agent = make_autotune_agent(llm_config, prompt_constraints, variant=agent_variant)

    # Checkpoint setup.
    if checkpoint_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
        checkpoint_dir = str(Path("checkpoints_autotune") / ts)
    ckpt_root = Path(checkpoint_dir) / kernel_name
    ckpt_root.mkdir(parents=True, exist_ok=True)

    _write(ckpt_root / "config.json", json.dumps({
        "kernel": kernel_name,
        "preset": preset,
        "dims": dims,
        "llm_config": {k: v for k, v in llm_config.items() if k != "api_key"},
        "autotune_config": autotune_config,
        "resume_from": str(resume_from),
        "resume_resolved_to": str(baseline_src),
        "max_turns": max_turns,
        "agent_variant": agent_variant,
    }, indent=2))
    _write(ckpt_root / "baseline_dsl.py", baseline_dsl_code)
    _write(ckpt_root / "baseline_translated.py", baseline_translated)
    _write(ckpt_root / "baseline_timing.txt", baseline_report)

    best_dsl = baseline_dsl_code
    best_translated = baseline_translated
    best_cycles = baseline_cycles
    current_dsl = baseline_dsl_code
    current_report = baseline_report
    _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                    best_cycles=best_cycles, turn=-1, last_status="BASELINE")

    user_prompt = build_autotune_user_prompt(
        kernel_name, dims, current_dsl, current_report,
        baseline_cycles=baseline_cycles, best_cycles=best_cycles)
    conversation = [{"role": "user", "content": user_prompt}]

    for turn in range(max_turns):
        turn_dir = ckpt_root / f"turn_{turn}"
        print(f"[autotune] Turn {turn + 1}/{max_turns} — current best={best_cycles}")

        _write(turn_dir / "user_prompt.txt", conversation[-1]["content"])

        run_result = await Runner.run(agent, conversation)
        assistant_text = run_result.final_output or ""
        conversation.append({"role": "assistant", "content": assistant_text})
        _write(turn_dir / "response.txt", assistant_text)
        reasoning = _reasoning_text(run_result)
        if reasoning:
            _write(turn_dir / "reasoning.txt", reasoning)

        proposal = _extract_code(assistant_text)
        if not proposal:
            print("  no code block — skipping")
            _write(turn_dir / "status.txt", "NO_CODE")
            conversation.append({"role": "user", "content":
                "Your response did not contain a ```python code block. "
                "Please emit the full updated tiled_reference(dims, tensors)."})
            _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                            best_cycles=best_cycles, turn=turn, last_status="NO_CODE")
            continue
        _write(turn_dir / "extracted_code.py", proposal)

        result = _evaluate_dsl_turn(
            proposal, kernel_name, dims, tensors, hw_config, max_total_compute_bw)

        # Persist artifacts for whichever gates ran. Each artifact is
        # written iff its corresponding text was populated.
        if result["dsl_correctness_text"] is not None:
            _write(turn_dir / "dsl_correctness_result.txt", result["dsl_correctness_text"])
        if result["translated_code"] is not None:
            _write(turn_dir / "translated_code.py", result["translated_code"])
        if result["translate_error_text"] is not None:
            _write(turn_dir / "translate_error.txt", result["translate_error_text"])
        if result["graph_correctness_text"] is not None:
            _write(turn_dir / "graph_correctness_result.txt", result["graph_correctness_text"])
        if result["timing_error_text"] is not None:
            _write(turn_dir / "timing_error.txt", result["timing_error_text"])

        status = result["status"]

        if status == "DSL_FAIL":
            print(f"  DSL_FAIL: {result['dsl_correctness_text'].splitlines()[0]}")
            _write(turn_dir / "status.txt", "DSL_FAIL")
            conversation.append({"role": "user", "content":
                "## Correctness gate 1 (DSL exec): FAIL\n\n"
                "Your proposal's DSL eager exec disagreed with gold.\n\n"
                f"```\n{result['dsl_correctness_text']}\n```\n\n"
                f"### Last correct tiled_reference (use this as the base)\n\n"
                f"```python\n{current_dsl}\n```"})
            _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                            best_cycles=best_cycles, turn=turn, last_status="DSL_FAIL")
            continue

        if status == "TRANSLATE_ERROR":
            print(f"  TRANSLATE_ERROR: {result['translate_error_text'].splitlines()[-2]}")
            _write(turn_dir / "status.txt", "TRANSLATE_ERROR")
            conversation.append({"role": "user", "content":
                "## Correctness gate 2 (translate): RAISED\n\n"
                "Your DSL exec passed but the deterministic translator could not "
                "lower it. This is usually a malformed DSL pattern (unsupported "
                "assignment shape, unknown DSL function, etc.).\n\n"
                f"```\n{result['translate_error_text']}\n```\n\n"
                f"### Last correct tiled_reference (use this as the base)\n\n"
                f"```python\n{current_dsl}\n```"})
            _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                            best_cycles=best_cycles, turn=turn, last_status="TRANSLATE_ERROR")
            continue

        if status == "IR_FAIL":
            print(f"  IR_FAIL: {result['graph_correctness_text'].splitlines()[0]}")
            _write(turn_dir / "status.txt", "IR_FAIL")
            conversation.append({"role": "user", "content":
                "## Correctness gate 3 (IR sim): FAIL\n\n"
                "DSL exec passed and translation succeeded, but the lowered "
                "graph's simulator output disagreed with gold. This generally "
                "indicates a translator/lowering issue surfaced by your edit.\n\n"
                f"```\n{result['graph_correctness_text']}\n```\n\n"
                f"### Last correct tiled_reference (use this as the base)\n\n"
                f"```python\n{current_dsl}\n```"})
            _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                            best_cycles=best_cycles, turn=turn, last_status="IR_FAIL")
            continue

        if status == "TIMING_ERROR":
            print(f"  TIMING_ERROR: {result['timing_error_text'].splitlines()[-2]}")
            _write(turn_dir / "status.txt", "TIMING_ERROR")
            conversation.append({"role": "user", "content":
                "## Timing model error\n\n"
                f"```\n{result['timing_error_text']}\n```\n\n"
                "Correctness passed but analyze_timing raised. This usually "
                "means a knob is out of range."})
            _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                            best_cycles=best_cycles, turn=turn, last_status="TIMING_ERROR")
            continue

        # PASS path.
        new_cycles = result["new_cycles"]
        new_report = result["new_report"]
        _write(turn_dir / "timing.txt", new_report)
        delta = new_cycles - best_cycles
        tag = "NEW_BEST" if new_cycles < best_cycles else ("SAME" if new_cycles == best_cycles else "REGRESSION")
        _write(turn_dir / "status.txt", f"PASS {tag} cycles={new_cycles} delta={delta:+d}")
        print(f"  correctness=PASS cycles={new_cycles} ({tag}, Δ={delta:+d})")

        current_dsl = proposal
        current_report = new_report
        if new_cycles < best_cycles:
            best_cycles = new_cycles
            best_dsl = proposal
            best_translated = result["translated_code"]
            _write(ckpt_root / "best.py", best_dsl)
            _write(ckpt_root / "best_translated.py", best_translated)
            _write(ckpt_root / "best_timing.txt", new_report)

        conversation.append({"role": "user", "content": build_autotune_user_prompt(
            kernel_name, dims, current_dsl, current_report,
            baseline_cycles=baseline_cycles, best_cycles=best_cycles)})
        _write_progress(ckpt_root, baseline_cycles=baseline_cycles,
                        best_cycles=best_cycles, turn=turn, last_status=tag)

    result = {
        "success": True,
        "kernel": kernel_name,
        "preset": preset,
        "baseline_cycles": baseline_cycles,
        "best_cycles": best_cycles,
        "speedup": baseline_cycles / best_cycles if best_cycles > 0 else None,
        "turns": max_turns,
        "resume_from": str(baseline_src),
        "checkpoint_dir": str(ckpt_root),
    }
    _write(ckpt_root / "result.json", json.dumps(result, indent=2))
    if best_dsl is not baseline_dsl_code:
        _write(ckpt_root / "best.py", best_dsl)
        _write(ckpt_root / "best_translated.py", best_translated)
    print(f"\n[autotune] done — baseline={baseline_cycles} best={best_cycles} "
          f"speedup={result['speedup']:.2f}x" if result["speedup"] else "")
    return result
```

- [ ] **Step 5.6: Update the autotune.py module docstring**

Replace the existing module docstring at the top of `src/autotune.py` with:

```python
"""Autotuner orchestration.

Takes a correctness-verified DSL ``tiled_reference(dims, tensors)`` (the
output of the implementer's refactor pass, persisted as ``dsl_code.py``)
and runs an agent loop that iteratively proposes performance-oriented
rewrites of the DSL. Each proposal is run through the triple-gate chain
(DSL exec -> translate -> IR sim) against the reference, then fed through
the analytical timing model on the translated build_graph; the report is
sent back to the agent.

The autotuner never mutates the algorithm — it only changes knobs like
tile_row / tile_col, par_dispatch, compute_bw, write_back_mu, and the
choice of buffering / broadcast / retile DSL ops.
"""
```

- [ ] **Step 5.7: Run the full test suite**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/ -x`
Expected: all tests pass — including the existing `test_orchestrator_autotune.py` tests (which monkeypatch `autotune.run_autotune` and so are insulated from the body changes) and the existing `test_autotune_progress.py` (the `progress.json` schema is unchanged).

- [ ] **Step 5.8: Commit**

```bash
cd /workspace/DEIOpt && git add StepGenFlow9/src/autotune.py StepGenFlow9/tests/test_autotune_loop.py
git commit -m "$(cat <<'EOF'
wire autotune to triple-gate DSL loop and DSL resume

Replaces _resolve_resume_build_graph with _resolve_resume_dsl_with_source.
Baseline measurement now runs DSL exec + translate + IR sim + timing model
through _evaluate_dsl_turn and hard-fails if any gate rejects.
Per-turn loop dispatches on the helper's status (NO_CODE, DSL_FAIL,
TRANSLATE_ERROR, IR_FAIL, TIMING_ERROR, PASS *). best.py becomes the
DSL source; best_translated.py snapshots the lowered graph alongside.
EOF
)"
```

---

### Task 6: Cleanup unused imports and verify per-outer integration unchanged

**Files:**
- Modify: `src/autotune.py` — drop unused imports left from the old loop body.

- [ ] **Step 6.1: Inspect imports for residue**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && python -c "import ast; t = ast.parse(open('src/autotune.py').read()); print([n.name for s in t.body if isinstance(s, ast.ImportFrom) for n in s.names])"`

Expected output should match what the new code uses. Cross-check the autotune.py imports against names referenced in the file body via `grep -nE "\b(name)\b" src/autotune.py` for any name in the import list. Typical residues to look for:

- `traceback` — still used (in `_evaluate_dsl_turn` exception capture). Keep.
- `_extract_code`, `_reasoning_text`, `_write` — still used in `run_autotune`. Keep.
- `_run_graph_correctness`, `_run_dsl_correctness` — used inside `_evaluate_dsl_turn`. Keep.
- `_resolve_resume_dsl` — *imported but not used*: the new code uses our own `_resolve_resume_dsl_with_source`. **Drop this import.**
- `translate` — used inside `_evaluate_dsl_turn`. Keep.

- [ ] **Step 6.2: Drop the unused `_resolve_resume_dsl` import**

In `src/autotune.py`, edit the orchestrator import to remove `_resolve_resume_dsl`. Final shape:

```python
from src.orchestrator import (  # noqa: E402
    _run_dsl_correctness,
    _run_graph_correctness,
    _write,
    _extract_code,
    _reasoning_text,
)
from src.dsl_to_step import translate  # noqa: E402
```

- [ ] **Step 6.3: Verify the per-outer integration is intact**

Reading-only check — no edit expected. Confirm:

1. `src/orchestrator.py:164` `_run_outer_autotune` still passes `resume_from=str(outer_dir)` to `run_autotune`.
2. `outer_dir/dsl_code.py` is written by the orchestrator's phase-1 success path (search the file for `dsl_code.py` writes to confirm it still exists; this is unchanged territory).
3. The `progress.json` schema written by `_write_progress` matches what `_load_autotune_progress` reads — both are unchanged.

Run: `grep -n 'dsl_code.py\|progress.json\|_run_outer_autotune' src/orchestrator.py | head -20`
Expected: at least one writer for `dsl_code.py` (in phase-1 success), the progress.json read in `_load_autotune_progress`, and `_run_outer_autotune` calling `run_autotune` with `resume_from=str(outer_dir)`. No edits required.

- [ ] **Step 6.4: Run the full test suite a final time**

Run: `cd /workspace/DEIOpt/StepGenFlow9 && pytest tests/ -v`
Expected: every test passes. Total new test count: 7 (Task 1) + 6 (Task 2) + 6+2 (Task 4 + Task 5) = 21 new tests.

- [ ] **Step 6.5: Commit**

```bash
cd /workspace/DEIOpt && git add StepGenFlow9/src/autotune.py
git commit -m "$(cat <<'EOF'
drop unused _resolve_resume_dsl import in autotune.py

Final cleanup after the DSL-form autotune loop refactor — the autotune
entry point uses its own _resolve_resume_dsl_with_source helper to also
return the source path for config.json provenance.
EOF
)"
```

---

## Manual smoke test (after Task 6, before declaring done)

Not part of the test suite. Run a one-turn autotune against a small kernel that today autotunes successfully, against the same model the implementer pipeline normally uses. The intent is to confirm:

1. The new prompt assembles correctly (no missing placeholder keys).
2. The LLM's response is recognizable as DSL form (a `tiled_reference` block, not a `build_graph` block).
3. `dsl_correctness_result.txt`, `translated_code.py`, `graph_correctness_result.txt`, and `timing.txt` all appear under `turn_0/`.
4. `best.py` is DSL source, not build_graph.

Pick a small kernel from `regression_subsets.yaml`, point `run_autotune.py --resume` at a known-good `outer_<N>/` containing a `dsl_code.py`, and inspect the resulting `checkpoints_autotune/<stamp>/<kernel>/` tree.

If the smoke test surfaces a wiring issue not caught by unit tests (e.g., a placeholder name typo, a missing artifact write), fix it as a follow-up commit and add a regression test where feasible.
