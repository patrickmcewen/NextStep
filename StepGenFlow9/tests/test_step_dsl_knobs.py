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
