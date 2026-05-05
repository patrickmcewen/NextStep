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


def test_n_byte_float16():
    assert sdm._n_byte(torch.float16, mock_bf16=False) == 2
    assert sdm._n_byte(torch.float16, mock_bf16=True) == 2


def test_n_byte_float32_real():
    assert sdm._n_byte(torch.float32, mock_bf16=False) == 4


def test_n_byte_float32_mock_bf16():
    assert sdm._n_byte(torch.float32, mock_bf16=True) == 2


def test_n_byte_uint64():
    # Uint64 is keyed by IR datatype class name in ops.py; we never see it
    # via a torch dtype in the eager DSL, but the helper handles the name.
    assert sdm._n_byte_for_name("Uint64", mock_bf16=False) == 8
    assert sdm._n_byte_for_name("Uint64", mock_bf16=True) == 8


def test_tile_bytes():
    a = torch.randn(2, 3, 5, 7, dtype=torch.float32)
    # tile = (5, 7); n_byte = 4 (real) or 2 (mock_bf16)
    assert sdm._tile_bytes(a, mock_bf16=False) == 5 * 7 * 4
    assert sdm._tile_bytes(a, mock_bf16=True) == 5 * 7 * 2


def test_stream_total_elements_2d_is_one():
    # Pure tile, no stream dims — stream total elements is 1.
    a = torch.randn(4, 4, dtype=torch.float32)
    assert sdm._stream_total_elements(a) == 1


def test_stream_total_elements_higher_rank():
    a = torch.randn(2, 3, 5, 7, dtype=torch.float32)
    # stream shape = (2, 3); total = 6
    assert sdm._stream_total_elements(a) == 6


def test_stream_dtype_size_bytes_uses_output_tile():
    # Mirrors ops.py's stream.stream_dtype.size_in_bytes() — tile_r * tile_c * n_byte.
    a = torch.randn(1, 8, 16, dtype=torch.float16)
    # tile = (8, 16); n_byte = 2
    assert sdm._stream_dtype_size_bytes(a, mock_bf16=True) == 8 * 16 * 2
