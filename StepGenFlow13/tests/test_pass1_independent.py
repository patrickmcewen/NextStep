"""Smoke tests for independent pass1 + smoothing pass.

These are deterministic — no LLM calls. They exercise:
  * `build_pass1_independent_user_prompt` rendering for leaf, non-leaf,
    and list-of-tensor args.
  * `_validate_load_once` AST gate (positive + negative cases).
  * `smooth_compose` on hand-crafted 2-function trees, with both a
    matching-shape and a mismatched-shape bridge.
"""
import textwrap

import pytest

from src.prompts import build_pass1_independent_user_prompt
from src.node_signature import TensorArg, ListOfTensorArg, ListOfIntArg
from src.orchestrator import _validate_load_once
from src.pass_smooth import smooth_compose


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_independent_prompt_leaf_has_raw_inputs_section():
    prompt = build_pass1_independent_user_prompt(
        node_name="kernel",
        is_root=False,
        reference_code="class Model: ...",
        dims={"B": 2},
        tensors={},
        arg_names=("Q", "K"),
        arg_specs=(TensorArg(shape=(2, 4, 8, 16)), TensorArg(shape=(2, 4, 8, 16))),
        children_signatures=[],
        function_signature="def kernel(Q, K):",
    )
    # Independent-mode markers
    assert "independent mode" in prompt
    assert "Inputs (independent mode)" in prompt
    assert "every positional input is **RAW**".lower() in prompt.lower()
    assert "exactly once" in prompt.lower()
    # Should NOT carry any contract-mode artifacts
    assert "Contract (declared by parent)" not in prompt
    assert "out_shapes" not in prompt   # no kwarg in independent signature
    assert "out_perms" not in prompt
    # Children block is absent (leaf)
    assert "Child Callables" not in prompt
    # Each tensor arg is mentioned
    assert "`Q`" in prompt and "`K`" in prompt


def test_independent_prompt_nonleaf_lists_child_placeholders():
    prompt = build_pass1_independent_user_prompt(
        node_name="root",
        is_root=True,
        reference_code="def compute_gold(dims): pass",
        dims={"B": 2},
        tensors={},
        arg_names=(),
        arg_specs=(),
        children_signatures=[
            ("attention", ("Q", "K", "V"),
             tuple(TensorArg(shape=(2, 4, 8, 16)) for _ in range(3)),
             ((2, 4, 8, 16),), False),
        ],
        function_signature="def tiled_reference(dims, tensors):",
    )
    # Children block present, but in placeholder mode (no kwargs)
    assert "Child Callables (placeholders)" in prompt
    assert "`attention(Q, K, V)`" in prompt
    # Independent mode tells the LLM to call without out_shapes
    assert "without `out_shapes`" in prompt or "WITHOUT `out_shapes`" in prompt


def test_independent_prompt_list_of_tensor_arg_describes_loop():
    prompt = build_pass1_independent_user_prompt(
        node_name="moe",
        is_root=False,
        reference_code="class Model: ...",
        dims={"B": 2},
        tensors={},
        arg_names=("x", "experts"),
        arg_specs=(
            TensorArg(shape=(2, 64)),
            ListOfTensorArg(elem_shape=(64, 64), length=8),
        ),
        children_signatures=[],
        function_signature="def moe(x, experts):",
    )
    # The list arg gets per-element load instructions.
    assert "list[Tensor(64, 64)] x 8" in prompt
    assert "offchip_load(experts[i]" in prompt
    assert "shared layout across all elements" in prompt


# ---------------------------------------------------------------------------
# AST load-once validator
# ---------------------------------------------------------------------------


def test_validate_load_once_passes_for_clean_code():
    code = textwrap.dedent("""
        def kernel(Q, K):
            q = offchip_load(Q, stride=(1,), out_shape_tiled=(8,), tile_row=8, tile_col=16)
            k = offchip_load(K, stride=(1,), out_shape_tiled=(8,), tile_row=8, tile_col=16)
            r = binary_matmul(q, k)
            return r
    """)
    err = _validate_load_once(
        code,
        arg_names=("Q", "K"),
        arg_specs=(TensorArg(shape=(2, 4, 8, 16)), TensorArg(shape=(2, 4, 8, 16))),
        function_name="kernel",
    )
    assert err is None, f"expected pass, got error: {err}"


def test_validate_load_once_rejects_double_load():
    code = textwrap.dedent("""
        def kernel(Q):
            q1 = offchip_load(Q, stride=(1,), out_shape_tiled=(8,), tile_row=8, tile_col=16)
            q2 = offchip_load(Q, stride=(1,), out_shape_tiled=(4,), tile_row=8, tile_col=32)
            r = binary_add(q1, q2)
            return r
    """)
    err = _validate_load_once(
        code,
        arg_names=("Q",),
        arg_specs=(TensorArg(shape=(2, 4, 8, 16)),),
        function_name="kernel",
    )
    assert err is not None
    assert "exactly one" in err
    assert "Q" in err


def test_validate_load_once_rejects_no_load():
    code = textwrap.dedent("""
        def kernel(Q):
            return Q
    """)
    err = _validate_load_once(
        code,
        arg_names=("Q",),
        arg_specs=(TensorArg(shape=(2, 4, 8, 16)),),
        function_name="kernel",
    )
    assert err is not None
    assert "Q" in err


def test_validate_load_once_accepts_list_loop():
    code = textwrap.dedent("""
        def kernel(experts):
            loaded = [offchip_load(experts[i], stride=(1,), out_shape_tiled=(8,),
                                    tile_row=8, tile_col=16)
                      for i in range(len(experts))]
            return loaded[0]
    """)
    err = _validate_load_once(
        code,
        arg_names=("experts",),
        arg_specs=(ListOfTensorArg(elem_shape=(64, 64), length=8),),
        function_name="kernel",
    )
    assert err is None, f"expected pass, got: {err}"


def test_validate_load_once_ignores_int_list():
    code = textwrap.dedent("""
        def kernel(seq_lens, x):
            x_loaded = offchip_load(x, stride=(1,), out_shape_tiled=(8,),
                                     tile_row=8, tile_col=16)
            seq_t = torch.tensor(seq_lens)
            return x_loaded
    """)
    err = _validate_load_once(
        code,
        arg_names=("seq_lens", "x"),
        arg_specs=(
            ListOfIntArg(length=4),
            TensorArg(shape=(2, 64)),
        ),
        function_name="kernel",
    )
    assert err is None, f"expected pass, got: {err}"


# ---------------------------------------------------------------------------
# Smoother
# ---------------------------------------------------------------------------


def test_smoother_inlines_simple_child_with_bridge():
    """Parent calls one child. Both have offchip_load + DSL ops + offchip_store.
    Smoother inlines the child, strips its boundary load, inserts an
    offchip_store + offchip_load round-trip bridge for the parent's value,
    and strips the child's terminal offchip_store.
    """
    parent = textwrap.dedent("""
        def root_kernel(x_raw):
            x = offchip_load(x_raw, stride=(1,), out_shape_tiled=(8,),
                              tile_row=8, tile_col=16)
            y = my_child(x)
            return offchip_store(y)
    """)
    child = textwrap.dedent("""
        def my_child(arg):
            arg_loaded = offchip_load(arg, stride=(2,), out_shape_tiled=(4,),
                                       tile_row=4, tile_col=32)
            inner = unary_relu(arg_loaded)
            return offchip_store(inner)
    """)

    fused = smooth_compose(
        root_dsl=parent,
        root_function_name="root_kernel",
        children_dsls={"my_child": child},
    )

    # The child call is gone.
    assert "my_child(" not in fused
    # The terminal offchip_store of the child is stripped: the inner
    # value flows into the parent.
    # The bridge is inserted (offchip_store(parent_value) + offchip_load
    # with the child's params).
    assert "offchip_store" in fused        # parent's terminal store still present
    assert "stride=(2," in fused           # child's params are used in the bridge
    assert "out_shape_tiled=(4," in fused
    assert "tile_row=4" in fused
    assert "tile_col=32" in fused
    # The child's `unary_relu` survived inlining.
    assert "unary_relu(" in fused


def test_smoother_strips_terminal_offchip_store_only_at_boundary():
    """A child whose terminal expr is a bare on-chip name (no offchip_store)
    should still be inlined cleanly with no boundary store-strip needed.
    """
    parent = textwrap.dedent("""
        def root(x_raw):
            x = offchip_load(x_raw, stride=(1,), out_shape_tiled=(8,),
                              tile_row=8, tile_col=16)
            y = pure_child(x)
            return offchip_store(y)
    """)
    child = textwrap.dedent("""
        def pure_child(arg):
            arg_loaded = offchip_load(arg, stride=(1,), out_shape_tiled=(8,),
                                       tile_row=8, tile_col=16)
            return unary_relu(arg_loaded)
    """)
    fused = smooth_compose(
        root_dsl=parent,
        root_function_name="root",
        children_dsls={"pure_child": child},
    )
    assert "pure_child(" not in fused
    assert "unary_relu(" in fused


def test_smoother_handles_multi_child_calls():
    """Parent calls two different children. Each gets inlined separately."""
    parent = textwrap.dedent("""
        def root(x_raw, y_raw):
            x = offchip_load(x_raw, stride=(1,), out_shape_tiled=(8,),
                              tile_row=8, tile_col=16)
            y = offchip_load(y_raw, stride=(1,), out_shape_tiled=(8,),
                              tile_row=8, tile_col=16)
            a = child_a(x)
            b = child_b(y)
            return binary_add(a, b)
    """)
    child_a = textwrap.dedent("""
        def child_a(arg):
            arg_loaded = offchip_load(arg, stride=(1,), out_shape_tiled=(8,),
                                       tile_row=8, tile_col=16)
            return unary_relu(arg_loaded)
    """)
    child_b = textwrap.dedent("""
        def child_b(arg):
            arg_loaded = offchip_load(arg, stride=(1,), out_shape_tiled=(8,),
                                       tile_row=8, tile_col=16)
            return unary_silu(arg_loaded)
    """)
    fused = smooth_compose(
        root_dsl=parent,
        root_function_name="root",
        children_dsls={"child_a": child_a, "child_b": child_b},
    )
    assert "child_a(" not in fused and "child_b(" not in fused
    assert "unary_relu(" in fused and "unary_silu(" in fused


def test_smoother_three_level_post_order_compose():
    """Build a 3-level tree manually:
      root -> mid -> leaf
    Smooth bottom-up: leaf is verbatim, mid composes with leaf, root
    composes with the smoothed mid. Confirm that after both passes,
    `leaf` and `mid` calls are gone and `unary_silu` (the leaf's body)
    survives.
    """
    leaf = textwrap.dedent("""
        def leaf(arg):
            arg_loaded = offchip_load(arg, stride=(1,), out_shape_tiled=(8,),
                                       tile_row=8, tile_col=16)
            return offchip_store(unary_silu(arg_loaded))
    """)
    mid = textwrap.dedent("""
        def mid(x):
            x_loaded = offchip_load(x, stride=(1,), out_shape_tiled=(8,),
                                     tile_row=8, tile_col=16)
            inner = leaf(x_loaded)
            return offchip_store(inner)
    """)
    root = textwrap.dedent("""
        def root(t):
            t_loaded = offchip_load(t, stride=(1,), out_shape_tiled=(8,),
                                     tile_row=8, tile_col=16)
            out = mid(t_loaded)
            return offchip_store(out)
    """)

    # Step 1: smooth `mid` against its child `leaf`.
    mid_smoothed = smooth_compose(
        root_dsl=mid,
        root_function_name="mid",
        children_dsls={"leaf": leaf},
    )
    assert "leaf(" not in mid_smoothed
    assert "unary_silu(" in mid_smoothed

    # Step 2: smooth `root` against the now-smoothed `mid`.
    root_smoothed = smooth_compose(
        root_dsl=root,
        root_function_name="root",
        children_dsls={"mid": mid_smoothed},
    )
    assert "mid(" not in root_smoothed
    assert "leaf(" not in root_smoothed
    assert "unary_silu(" in root_smoothed


def test_smoother_avoids_local_name_collisions():
    """If parent and child both bind a local named `tmp`, the smoother
    must rename the child's local to avoid clobbering the parent's.
    """
    parent = textwrap.dedent("""
        def root(x_raw):
            tmp = offchip_load(x_raw, stride=(1,), out_shape_tiled=(8,),
                                tile_row=8, tile_col=16)
            y = my_child(tmp)
            return offchip_store(y)
    """)
    child = textwrap.dedent("""
        def my_child(arg):
            arg_loaded = offchip_load(arg, stride=(1,), out_shape_tiled=(8,),
                                       tile_row=8, tile_col=16)
            tmp = unary_relu(arg_loaded)
            return offchip_store(tmp)
    """)
    fused = smooth_compose(
        root_dsl=parent,
        root_function_name="root",
        children_dsls={"my_child": child},
    )
    # Parent's `tmp` is preserved; child's `tmp` is renamed.
    # Both unary_relu (inlined) and tmp (parent's) live.
    # The child's renamed local should NOT clash with the parent's.
    # We can test this by re-parsing and confirming there's no double assignment
    # to `tmp` at the same scope.
    import ast
    tree = ast.parse(fused)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    tmp_assigns = [s for s in fn.body
                   if isinstance(s, ast.Assign)
                   and len(s.targets) == 1
                   and isinstance(s.targets[0], ast.Name)
                   and s.targets[0].id == "tmp"]
    # Only the parent's `tmp = offchip_load(...)` should remain.
    assert len(tmp_assigns) == 1, (
        f"expected exactly one `tmp = ...` (parent's), got {len(tmp_assigns)}")
