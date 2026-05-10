from src.orchestrator import (
    _check_banned_ops,
    _check_dataflow_invariant,
    _extract_call_site_rawness,
)


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


# ---------------------------------------------------------------------------
# Raw-positional-arg rule: a non-root function whose contract tags a positional
# arg as raw must `offchip_load` (or another producer) it before feeding it to
# any DSL consumer. Blackbox-child call sites remain exempt.
# ---------------------------------------------------------------------------

def test_raw_positional_arg_into_consumer_is_flagged():
    """Forwarded raw weight fed straight into binary_matmul must be flagged."""
    code = '''
def attention_o_proj(Q, weight, *, out_shapes, out_perms=None):
    out = binary_matmul(Q, weight)
    return out
'''
    violations = _check_dataflow_invariant(
        code,
        blackbox_names=(),
        raw_arg_names=frozenset({"weight"}),
    )
    assert any("weight" in v and "binary_matmul" in v for v in violations), violations


def test_raw_positional_arg_loaded_first_is_ok():
    """Same code with offchip_load on the raw weight must pass."""
    code = '''
def attention_o_proj(Q, weight, *, out_shapes, out_perms=None):
    W = offchip_load(weight, ...)
    out = binary_matmul(Q, W)
    return out
'''
    violations = _check_dataflow_invariant(
        code,
        blackbox_names=(),
        raw_arg_names=frozenset({"weight"}),
    )
    assert violations == []


def test_raw_positional_arg_into_blackbox_is_ok():
    """Raw arg may flow straight into a child blackbox without a load."""
    code = '''
def parent(x, weight, *, out_shapes, out_perms=None):
    out = inner(x, weight, out_shapes=((4, 1, 8),))
    return out
'''
    violations = _check_dataflow_invariant(
        code,
        blackbox_names=("inner",),
        raw_arg_names=frozenset({"weight"}),
    )
    assert violations == []


def test_onchip_positional_arg_no_load_required():
    """An on-chip positional arg (parent fed in a streamed value) feeds
    consumers directly — no load required."""
    code = '''
def proj(x_stream, *, out_shapes, out_perms=None):
    return unary_square(x_stream)
'''
    violations = _check_dataflow_invariant(
        code,
        blackbox_names=(),
        raw_arg_names=frozenset(),  # no raw args
    )
    assert violations == []


def test_method_chain_on_raw_arg_into_consumer_is_flagged():
    """``raw_arg.reshape(...)`` then into a consumer must still be flagged
    because the receiver classification propagates via _classify_value."""
    code = '''
def proj(weight, *, out_shapes, out_perms=None):
    return unary_square(weight.reshape(4, 8))
'''
    violations = _check_dataflow_invariant(
        code,
        blackbox_names=(),
        raw_arg_names=frozenset({"weight"}),
    )
    assert any("weight" in v and "unary_square" in v for v in violations), violations


# ---------------------------------------------------------------------------
# Static rawness extractor over a parent's verified code.
# ---------------------------------------------------------------------------

def test_extractor_classifies_root_call_site():
    """At the root, every `tensors[...]` arg is raw; blackbox returns are on-chip."""
    code = '''
def tiled_reference(dims, tensors):
    Q, K, V = pre_attention(tensors["x"], tensors["q_proj"],
                             out_shapes=((4,1,8),(4,1,8),(4,1,8)))
    out = attention_o_proj(Q, K, V, tensors["o_proj_weight"], tensors["x"],
                            out_shapes=((4,1,8),))
    return offchip_store(out)
'''
    rawness = _extract_call_site_rawness(
        code,
        child_names=("pre_attention", "attention_o_proj"),
        blackbox_names=("pre_attention", "attention_o_proj"),
    )
    assert rawness["pre_attention"] == (True, True)
    # Q, K, V are blackbox returns (on-chip); o_proj_weight + x are raw.
    assert rawness["attention_o_proj"] == (False, False, False, True, True)


def test_extractor_propagates_parent_raw_args():
    """A non-root parent forwarding its own raw arg keeps it raw at the
    grandchild's call site — only when ``parent_raw_arg_names`` is fed in."""
    code = '''
def pre_attention(input_tensor, q_proj, k_proj, v_proj, cos, sin,
                   *, out_shapes, out_perms=None):
    Q, K, V = pre_attn_norm_and_proj(
        input_tensor, q_proj, k_proj, v_proj, cos,
        out_shapes=out_shapes,
    )
    Q, K, V = per_head_norm_and_rope(
        Q, K, V, cos, sin, out_shapes=out_shapes,
    )
    return Q, K, V
'''
    parent_raw = frozenset({
        "input_tensor", "q_proj", "k_proj", "v_proj", "cos", "sin",
    })
    rawness = _extract_call_site_rawness(
        code,
        child_names=("pre_attn_norm_and_proj", "per_head_norm_and_rope"),
        blackbox_names=("pre_attn_norm_and_proj", "per_head_norm_and_rope"),
        parent_raw_arg_names=parent_raw,
    )
    # First call: every arg is a forwarded raw arg.
    assert rawness["pre_attn_norm_and_proj"] == (True, True, True, True, True)
    # Second call: Q/K/V come from the prior blackbox (on-chip); cos/sin are
    # forwarded raw args from the parent.
    assert rawness["per_head_norm_and_rope"] == (False, False, False, True, True)


def test_extractor_records_only_first_call():
    """Contracts are recorded on first stub invocation; the extractor must
    mirror that by classifying the first textual call site only."""
    code = '''
def parent(x, weight, *, out_shapes, out_perms=None):
    a = inner(x, out_shapes=((4,1,8),))
    b = inner(weight, out_shapes=((4,1,8),))
    return b
'''
    rawness = _extract_call_site_rawness(
        code,
        child_names=("inner",),
        blackbox_names=("inner",),
        parent_raw_arg_names=frozenset({"weight"}),
    )
    # First call passes ``x`` (on-chip intermediate arg).
    assert rawness["inner"] == (False,)


def test_extractor_handles_dsl_producer_arg():
    """An arg whose name was assigned from a DSL producer is on-chip."""
    code = '''
def tiled_reference(dims, tensors):
    x = offchip_load(tensors["x"], ...)
    out = child(x, out_shapes=((4,1,8),))
    return offchip_store(out)
'''
    rawness = _extract_call_site_rawness(
        code,
        child_names=("child",),
        blackbox_names=("child",),
    )
    assert rawness["child"] == (False,)


def test_compliance_allows_torch_tensor_for_list_int_conversion():
    """``torch.tensor(<list_of_ints>)`` is the sole permitted ``torch.*``
    constructor — required to feed list[int] inputs into ``metadata_gen``
    for ragged dataflow."""
    code = '''
def tiled_reference(dims, tensors):
    x = offchip_load(tensors["x"], ...)
    seq_len_t = torch.tensor(tensors["num_token_list"])
    seq_len = metadata_gen(seq_len_t)
    return offchip_store(x)
'''
    violations = _check_banned_ops(code, REFACTOR_PASS, is_root=True)
    torch_violations = [v for v in violations if "torch.tensor" in v]
    assert torch_violations == [], torch_violations


def test_compliance_still_blocks_other_torch_constructors():
    """``torch.zeros``/``torch.ones`` etc. remain banned — the allowance is
    narrow and specifically for list[int] → tensor conversion."""
    code = '''
def tiled_reference(dims, tensors):
    z = torch.zeros(4, 8)
    return offchip_store(z)
'''
    violations = _check_banned_ops(code, REFACTOR_PASS, is_root=True)
    torch_violations = [v for v in violations if "torch.zeros" in v]
    assert len(torch_violations) >= 1, violations
