from src.orchestrator import _check_banned_ops, _check_dataflow_invariant


REFACTOR_PASS = "refactor_final"   # the existing standalone-mode pass key


# ---------------------------------------------------------------------------
# Existing required-op (blackbox-name) textual check.
# ---------------------------------------------------------------------------

def test_extra_required_ops_does_not_flag_uncalled_blackbox():
    """Blackboxes are tools, not a quota: a root that ignores its declared
    children entirely (and instead implements the kernel directly with DSL
    ops) must NOT be flagged for missing child names."""
    code = '''
def tiled_reference(dims, tensors):
    x = offchip_load(tensors["x"], ...)
    return offchip_store(x)
'''
    violations = _check_banned_ops(
        code, REFACTOR_PASS, is_root=True,
        extra_required_ops=("attention", "feed_forward"),
    )
    blackbox_violations = [v for v in violations
                           if "attention" in v or "feed_forward" in v]
    assert blackbox_violations == [], blackbox_violations


def test_extra_required_ops_satisfied_when_called():
    code = '''
def tiled_reference(dims, tensors):
    x = tensors["x"]
    a = attention(x, out_shapes=(tuple(x.shape),))
    b = feed_forward(a, out_shapes=(tuple(a.shape),))
    return offchip_store(b)
'''
    violations = _check_banned_ops(
        code, REFACTOR_PASS, is_root=True,
        extra_required_ops=("attention", "feed_forward"),
    )
    blackbox_violations = [v for v in violations
                           if "attention" in v or "feed_forward" in v]
    assert blackbox_violations == []


def test_extra_required_ops_default_empty_is_backward_compatible():
    code = '''
def tiled_reference(dims, tensors):
    x = offchip_load(tensors["x"], ...)
    return offchip_store(x)
'''
    v1 = _check_banned_ops(code, REFACTOR_PASS, is_root=True)
    v2 = _check_banned_ops(code, REFACTOR_PASS, is_root=True,
                           extra_required_ops=())
    assert v1 == v2
    assert v1 == []


# ---------------------------------------------------------------------------
# offchip_store is required at root regardless of decomposition;
# offchip_load is dropped (subsumed by the AST dataflow walk).
# ---------------------------------------------------------------------------

def test_orchestrator_root_with_blackbox_child_must_call_offchip_store():
    """Root that is a pure blackbox orchestrator still needs offchip_store —
    the kernel's externally-observable output goes off-chip via a single sink
    at the root, regardless of whether children handle their own intermediate
    storage internally."""
    bad = '''
def tiled_reference(dims, tensors):
    x = tensors["x"]
    return moe_path(x, out_shapes=((4, 8),), out_perms=(None,))
'''
    violations = _check_banned_ops(
        bad, REFACTOR_PASS, is_root=True,
        extra_required_ops=("moe_path",),
    )
    assert any("offchip_store" in v for v in violations), violations

    good = '''
def tiled_reference(dims, tensors):
    x = tensors["x"]
    return offchip_store(moe_path(x, out_shapes=((4, 8),), out_perms=(None,)))
'''
    violations = _check_banned_ops(
        good, REFACTOR_PASS, is_root=True,
        extra_required_ops=("moe_path",),
    )
    assert violations == [], violations


def test_two_blackbox_passthrough_root_must_call_offchip_store():
    bad = '''
def tiled_reference(dims, tensors):
    inp = tensors["input_tensor"]
    res = attention_path(inp, out_shapes=((4, 8),))
    return moe_path(res, out_shapes=((4, 8),))
'''
    violations = _check_banned_ops(
        bad, REFACTOR_PASS, is_root=True,
        extra_required_ops=("attention_path", "moe_path"),
    )
    assert any("offchip_store" in v for v in violations), violations

    good = '''
def tiled_reference(dims, tensors):
    inp = tensors["input_tensor"]
    res = attention_path(inp, out_shapes=((4, 8),))
    out = moe_path(res, out_shapes=((4, 8),))
    return offchip_store(out)
'''
    violations = _check_banned_ops(
        good, REFACTOR_PASS, is_root=True,
        extra_required_ops=("attention_path", "moe_path"),
    )
    assert violations == [], violations


def test_nonroot_with_blackbox_child_does_not_need_offchip_store():
    """Non-root nodes hand a stream up to their parent — they never need
    offchip_store. The is_root=False carve-out remains."""
    code = '''
def helper(x, *, out_shapes, out_perms=None):
    return moe_path(x, out_shapes=out_shapes)
'''
    violations = _check_banned_ops(
        code, REFACTOR_PASS, is_root=False,
        extra_required_ops=("moe_path",),
    )
    assert violations == [], violations


def test_mixed_root_with_proper_loads_passes():
    code = '''
def tiled_reference(dims, tensors):
    x = offchip_load(tensors["x"], ...)
    y = binary_mul(x, x)
    z = unary_rsqrt(y)
    return offchip_store(z)
'''
    violations = _check_banned_ops(
        code, REFACTOR_PASS, is_root=True, extra_required_ops=(),
    )
    assert violations == []


def test_dsl_consumer_on_raw_tensor_subscript_is_flagged():
    """`binary_mul(tensors["x"], ...)` directly with no load is rejected."""
    code = '''
def tiled_reference(dims, tensors):
    y = binary_mul(tensors["x"], tensors["x"])
    return offchip_store(y)
'''
    violations = _check_dataflow_invariant(code)
    assert any("binary_mul" in v and "tensors" in v for v in violations)


def test_dsl_consumer_on_named_raw_tensor_is_flagged():
    """Same but routed through a name binding."""
    code = '''
def tiled_reference(dims, tensors):
    x = tensors["x"]
    y = binary_mul(x, x)
    return offchip_store(y)
'''
    violations = _check_dataflow_invariant(code)
    assert any("binary_mul" in v for v in violations)
    assert any("`x`" in v for v in violations)


def test_offchip_store_on_raw_tensor_is_flagged():
    """offchip_store is a sink (consumer); raw tensors[...] input must load first."""
    code = '''
def tiled_reference(dims, tensors):
    return offchip_store(tensors["x"])
'''
    violations = _check_dataflow_invariant(code)
    assert any("offchip_store" in v and "tensors" in v for v in violations)


def test_method_chain_reshape_on_raw_is_flagged():
    """Host-side reshape laundering must not slip past the dataflow check."""
    code = '''
def tiled_reference(dims, tensors):
    y = binary_mul(tensors["x"].reshape(4, 8), tensors["x"].reshape(4, 8))
    return offchip_store(y)
'''
    violations = _check_dataflow_invariant(code)
    assert any("binary_mul" in v for v in violations)


def test_nonroot_node_with_intermediate_args_passes():
    """The attention_path case: positional args are on-chip per parent contract,
    so no offchip_load is needed for DSL ops that consume them directly."""
    code = '''
def attention_path(input_tensor, q_proj, *, out_shapes, out_perms=None):
    x_sq = binary_mul(input_tensor, input_tensor)
    sum_sq = unary_rowwise_sum(x_sq)
    inv = unary_rsqrt(sum_sq)
    Q, K, V = pre_attention(input_tensor, q_proj,
                             out_shapes=((4, 8), (4, 8), (4, 8)))
    attn = attention(Q, K, V,
                     out_shapes=out_shapes,
                     out_perms=out_perms or (None,))
    return offchip_store(attn)
'''
    violations = _check_banned_ops(
        code, REFACTOR_PASS, is_root=False,
        extra_required_ops=("pre_attention", "attention"),
    )
    assert violations == []


def test_tuple_unpack_from_blackbox_is_onchip():
    code = '''
def helper(x, *, out_shapes, out_perms=None):
    Q, K, V = pre_attention(x, out_shapes=((4, 8), (4, 8), (4, 8)))
    s = binary_add(Q, K)
    return offchip_store(s)
'''
    violations = _check_banned_ops(
        code, REFACTOR_PASS, is_root=False,
        extra_required_ops=("pre_attention",),
    )
    assert violations == []


def test_dsl_chain_is_transitively_onchip():
    code = '''
def tiled_reference(dims, tensors):
    x = offchip_load(tensors["x"], ...)
    a = unary_square(x)
    b = unary_rowwise_sum(a)
    c = unary_rsqrt(b)
    d = binary_mul(x, c)
    return offchip_store(d)
'''
    violations = _check_dataflow_invariant(code)
    assert violations == []


def test_producer_op_arg_not_traced():
    """A producer (offchip_load, select_gen, …) takes a raw tensor; we should
    not flag its tensor arg even though it's a raw `tensors[...]` subscript."""
    code = '''
def tiled_reference(dims, tensors):
    x = offchip_load(tensors["x"], ...)
    sel = select_gen(is_multihot=True, tensor=tensors["mask"], n=4)
    return offchip_store(x)
'''
    violations = _check_dataflow_invariant(code)
    assert violations == []


def test_blackbox_call_can_take_raw_subscript_directly():
    """Blackbox stubs accept either vanilla raw tensors or tiled streams; we
    must not flag a raw `tensors[...]` flowing into a blackbox."""
    code = '''
def tiled_reference(dims, tensors):
    return offchip_store(moe_path(tensors["x"], out_shapes=((4, 8),)))
'''
    violations = _check_banned_ops(
        code, REFACTOR_PASS, is_root=True,
        extra_required_ops=("moe_path",),
    )
    assert violations == []
