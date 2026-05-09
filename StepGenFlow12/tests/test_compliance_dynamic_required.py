from src.orchestrator import _check_banned_ops


REFACTOR_PASS = "refactor_final"   # the existing standalone-mode pass key


def test_extra_required_ops_flags_missing_blackbox():
    code = '''
def tiled_reference(dims, tensors):
    out = tensors["x"]
    return offchip_store(out)
'''
    violations = _check_banned_ops(
        code, REFACTOR_PASS, is_root=True,
        extra_required_ops=("attention", "feed_forward"),
    )
    assert any("attention" in v for v in violations)
    assert any("feed_forward" in v for v in violations)


def test_extra_required_ops_satisfied_when_called():
    code = '''
def tiled_reference(dims, tensors):
    x = tensors["x"]
    a = attention(x, out_shape=tuple(x.shape))
    b = feed_forward(a, out_shape=tuple(a.shape))
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
    return offchip_store(tensors["x"])
'''
    v1 = _check_banned_ops(code, REFACTOR_PASS, is_root=True)
    v2 = _check_banned_ops(code, REFACTOR_PASS, is_root=True,
                           extra_required_ops=())
    assert v1 == v2
