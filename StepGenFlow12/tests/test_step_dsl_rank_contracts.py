import pytest
import torch

from src.step_dsl import (
    Float32,
    Uint64,
    StepTensor,
    Tile,
    accum_retile_col,
    accum_retile_row,
    accum_signal_req_all_read,
    binary_map_accum,
    flatmap_counter,
    promote,
)


def _rank0_tiles(n=4, shape=(1, 1)):
    return StepTensor(
        torch.ones(n, *shape),
        stream_dtype=Tile(Float32(), shape),
    )


def test_promote_nonzero_rank_rejects_rank0_stream():
    x = _rank0_tiles()

    with pytest.raises(AssertionError, match="promote.*input stream rank"):
        promote(x, rank=1)


@pytest.mark.parametrize(
    "op",
    [accum_retile_row, accum_retile_col, accum_signal_req_all_read],
)
def test_accum_variants_reject_rank0_stream_for_rank1_reduction(op):
    x = _rank0_tiles()

    with pytest.raises(AssertionError, match="input stream rank"):
        op(x, rank=1)


def test_binary_map_accum_rejects_rank0_stream_for_rank1_reduction():
    a = _rank0_tiles(shape=(1, 2))
    b = _rank0_tiles(shape=(2, 1))

    with pytest.raises(AssertionError, match="input stream rank"):
        binary_map_accum(a, b, rank=1)


def _counter_input(n=4):
    return StepTensor(
        torch.tensor([[[n]]], dtype=torch.int64),
        stream_dtype=Tile(Uint64(), (1, 1)),
    )


def test_binary_map_accum_rejects_distinct_dynamic_producers_with_same_size():
    count = _counter_input(4)
    a_counter = flatmap_counter(count)
    b_counter = flatmap_counter(count)
    a = StepTensor(
        torch.ones(1, 4, 1, 2),
        stream_dtype=Tile(Float32(), (1, 2)),
        dyn_mask=a_counter.dyn_mask,
        dyn_origins=a_counter.dyn_origins,
    )
    b = StepTensor(
        torch.ones(1, 4, 2, 1),
        stream_dtype=Tile(Float32(), (2, 1)),
        dyn_mask=b_counter.dyn_mask,
        dyn_origins=b_counter.dyn_origins,
    )

    with pytest.raises(AssertionError, match="dynamic stream origin mismatch"):
        binary_map_accum(a, b, rank=1)


def test_binary_map_accum_accepts_reused_dynamic_producer():
    count = _counter_input(4)
    counter = flatmap_counter(count)
    a = StepTensor(
        torch.ones(1, 4, 1, 2),
        stream_dtype=Tile(Float32(), (1, 2)),
        dyn_mask=counter.dyn_mask,
        dyn_origins=counter.dyn_origins,
    )
    b = StepTensor(
        torch.ones(1, 4, 2, 1),
        stream_dtype=Tile(Float32(), (2, 1)),
        dyn_mask=counter.dyn_mask,
        dyn_origins=counter.dyn_origins,
    )

    out = binary_map_accum(a, b, rank=1)

    assert tuple(out.underlying_tensor.shape) == (1, 1, 1)
