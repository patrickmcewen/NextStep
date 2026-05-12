from src.orchestrator import (
    _check_banned_ops,
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
# offchip_load is not required at refactor_final (a node consuming only
# intermediate args / blackbox returns never needs it). Consumer-source
# and producer-raw-slot rules are enforced at runtime by StepTensor type
# assertions in step_dsl.py (_step_meta / _assert_raw), not here.
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
# Static rawness extractor over a parent's verified code — used by
# _pass1_walk to (a) detect children that the parent inlined entirely and
# (b) stamp ``arg_is_raw`` onto each surviving child contract.
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
