"""Tests for step_dsl_memory.py — the metric-recording wrapper around step_dsl."""

import torch

from src import step_dsl_memory as sdm


def test_tracker_context_manager_lifecycle():
    # No tracker active outside the with-block.
    assert sdm._ACTIVE is None
    with sdm.tracker() as t:
        assert sdm._ACTIVE is t
        assert isinstance(t, sdm.Tracker)
    assert sdm._ACTIVE is None


def test_tracker_default_mock_bf16_is_true():
    with sdm.tracker() as t:
        assert t.mock_bf16 is True


def test_tracker_mock_bf16_can_be_overridden():
    with sdm.tracker(mock_bf16=False) as t:
        assert t.mock_bf16 is False


def test_nested_trackers_save_and_restore():
    with sdm.tracker() as outer:
        with sdm.tracker(mock_bf16=False) as inner:
            assert sdm._ACTIVE is inner
        assert sdm._ACTIVE is outer
    assert sdm._ACTIVE is None


def test_unwrapped_call_outside_tracker_does_not_record():
    # Calling a wrapped DSL fn outside a tracker is fine and records nothing.
    a = torch.randn(2, 4, 4, dtype=torch.float32)
    b = torch.randn(2, 4, 4, dtype=torch.float32)
    out = sdm.binary_add(a, b)
    assert torch.equal(out, a + b)


def test_wrapped_function_signatures_preserved():
    # Wrapping must keep the function name and basic call signature usable.
    assert callable(sdm.offchip_load)
    assert callable(sdm.binary_matmul)
    assert callable(sdm.unary_silu)
    assert callable(sdm.offchip_store)


def test_dsl_functions_reexported():
    # Every name in step_dsl.DSL_FUNCTIONS is exported from step_dsl_memory.
    from src import step_dsl
    for name in step_dsl.DSL_FUNCTIONS:
        assert hasattr(sdm, name), f"step_dsl_memory missing {name}"


def test_records_empty_when_no_metric_fn_yet():
    # Until ops are registered in METRIC_FNS, records list stays empty even
    # if a wrapped DSL function is called inside a tracker.
    a = torch.randn(2, 4, 4, dtype=torch.float32)
    b = torch.randn(2, 4, 4, dtype=torch.float32)
    with sdm.tracker() as t:
        sdm.binary_add(a, b)
    assert t.records == []
