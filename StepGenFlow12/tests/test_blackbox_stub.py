import torch
import torch.nn as nn
from src.blackbox_stub import make_stub, ContractRecorder
from src.node_signature import (
    ListOfIntArg,
    ListOfTensorArg,
    TensorArg,
)


def _tensor_specs(*shapes):
    return tuple(TensorArg(shape=s) for s in shapes)


class _Add(nn.Module):
    def forward(self, a, b):
        return a + b


class _SplitQKV(nn.Module):
    """Returns a 3-tuple — used to verify multi-output stub semantics."""
    def forward(self, x):
        # Treat last dim as 3*head_dim and split.
        h = x.shape[-1] // 3
        return x[..., :h], x[..., h:2 * h], x[..., 2 * h:]


def test_stub_preserves_reference_semantics():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        arg_specs=_tensor_specs((1, 4, 8), (1, 4, 8)),
        recorder=rec,
    )
    a = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    b = torch.ones(1, 4, 8) * 10
    out = stub(a, b, out_shapes=((1, 4, 8),))
    assert torch.equal(out, a + b)


def test_stub_handles_tile_input_via_flatten_reshape():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        arg_specs=_tensor_specs((4, 8), (4, 8)),
        recorder=rec,
    )
    a_vanilla = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b_vanilla = torch.ones(4, 8) * 10
    a_tiled = a_vanilla.reshape(2, 2, 4, 2)
    b_tiled = b_vanilla.reshape(2, 2, 4, 2)
    out = stub(a_tiled, b_tiled, out_shapes=((2, 2, 4, 2),))
    expected = (a_vanilla + b_vanilla).reshape(2, 2, 4, 2)
    assert torch.equal(out, expected)


def test_recorder_captures_contract_on_first_call():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        arg_specs=_tensor_specs((4, 8), (4, 8)),
        recorder=rec,
    )
    a = torch.randn(2, 2, 4, 2)
    b = torch.randn(2, 2, 4, 2)
    stub(a, b, out_shapes=((2, 2, 4, 2),))
    contract = rec.contract
    assert contract is not None
    assert contract.arg_names == ("a", "b")
    assert contract.tiled_shapes == ((2, 2, 4, 2), (2, 2, 4, 2))
    assert torch.equal(contract.tiled_values[0], a)
    assert contract.out_shapes == ((2, 2, 4, 2),)


def test_recorder_only_captures_first_call():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        arg_specs=_tensor_specs((4, 8), (4, 8)),
        recorder=rec,
    )
    a = torch.randn(2, 2, 4, 2)
    b = torch.randn(2, 2, 4, 2)
    stub(a, b, out_shapes=((2, 2, 4, 2),))
    a2 = torch.randn(2, 2, 4, 2)
    stub(a2, b, out_shapes=((2, 2, 4, 2),))
    assert torch.equal(rec.contract.tiled_values[0], a)   # first call's value


def test_stub_returns_tuple_for_multi_output_ref():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_SplitQKV(),
        arg_names=("x",),
        arg_specs=_tensor_specs((1, 4, 24)),
        recorder=rec,
    )
    x = torch.arange(96, dtype=torch.float32).reshape(1, 4, 24)
    q, k, v = stub(x, out_shapes=((1, 4, 8), (1, 4, 8), (1, 4, 8)))
    assert torch.equal(q, x[..., :8])
    assert torch.equal(k, x[..., 8:16])
    assert torch.equal(v, x[..., 16:24])
    assert rec.contract.out_shapes == ((1, 4, 8), (1, 4, 8), (1, 4, 8))


def test_stub_rejects_output_count_mismatch():
    """Asking for 2 outputs from a 3-output ref must fail loudly."""
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_SplitQKV(),
        arg_names=("x",),
        arg_specs=_tensor_specs((1, 4, 24)),
        recorder=rec,
    )
    x = torch.randn(1, 4, 24)
    try:
        stub(x, out_shapes=((1, 4, 8), (1, 4, 8)))
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_recorder_captures_tiled_outputs_single():
    """Single-output ref: ``out_is_tuple=False``, ``tiled_outputs`` is a
    1-tuple parallel to ``out_shapes`` and matches the stub return value."""
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        arg_specs=_tensor_specs((4, 8), (4, 8)),
        recorder=rec,
    )
    a_vanilla = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b_vanilla = torch.ones(4, 8) * 10
    a_tiled = a_vanilla.reshape(2, 2, 4, 2)
    b_tiled = b_vanilla.reshape(2, 2, 4, 2)
    out = stub(a_tiled, b_tiled, out_shapes=((2, 2, 4, 2),))
    contract = rec.contract
    assert contract.out_is_tuple is False
    assert len(contract.tiled_outputs) == 1
    assert torch.equal(contract.tiled_outputs[0], out)


def test_recorder_captures_tiled_outputs_tuple():
    """Tuple-output ref: ``out_is_tuple=True``, ``tiled_outputs`` parallel
    to ``out_shapes`` and matches each stub output element."""
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_SplitQKV(),
        arg_names=("x",),
        arg_specs=_tensor_specs((1, 4, 24)),
        recorder=rec,
    )
    x = torch.arange(96, dtype=torch.float32).reshape(1, 4, 24)
    q, k, v = stub(x, out_shapes=((1, 4, 8), (1, 4, 8), (1, 4, 8)))
    contract = rec.contract
    assert contract.out_is_tuple is True
    assert len(contract.tiled_outputs) == 3
    assert torch.equal(contract.tiled_outputs[0], q)
    assert torch.equal(contract.tiled_outputs[1], k)
    assert torch.equal(contract.tiled_outputs[2], v)


# ---------------------------------------------------------------------------
# List-typed child args (per-expert weight stacks, per-batch seq lengths)
# ---------------------------------------------------------------------------

class _MoEExpertSum(nn.Module):
    """Reference for a child that takes ``list[Tensor]`` per-expert weights."""
    def forward(self, x, w_list):
        out = torch.zeros_like(x)
        for w in w_list:
            out = out + x @ w @ w.T
        return out


def test_stub_passes_list_of_tensor_through_unchanged():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_MoEExpertSum(),
        arg_names=("x", "w_list"),
        arg_specs=(
            TensorArg(shape=(1, 4, 8)),
            ListOfTensorArg(length=3, elem_shape=(8, 8)),
        ),
        recorder=rec,
    )
    x = torch.randn(1, 4, 8)
    w_list = [torch.randn(8, 8) for _ in range(3)]
    out = stub(x, w_list, out_shapes=((1, 4, 8),))
    expected = torch.zeros_like(x)
    for w in w_list:
        expected = expected + x @ w @ w.T
    assert torch.equal(out, expected)


def test_recorder_captures_list_of_tensor_contract():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_MoEExpertSum(),
        arg_names=("x", "w_list"),
        arg_specs=(
            TensorArg(shape=(1, 4, 8)),
            ListOfTensorArg(length=3, elem_shape=(8, 8)),
        ),
        recorder=rec,
    )
    x = torch.randn(1, 4, 8)
    w_list = [torch.randn(8, 8) for _ in range(3)]
    stub(x, w_list, out_shapes=((1, 4, 8),))

    contract = rec.contract
    assert contract.arg_names == ("x", "w_list")
    # vanilla/tiled shape entries are () for list args; spec carries the info.
    assert contract.vanilla_shapes == ((1, 4, 8), ())
    assert contract.tiled_shapes == ((1, 4, 8), ())
    assert contract.arg_specs == (
        TensorArg(shape=(1, 4, 8)),
        ListOfTensorArg(length=3, elem_shape=(8, 8)),
    )
    # tiled_values mirrors the heterogeneous arg kinds.
    assert isinstance(contract.tiled_values[0], torch.Tensor)
    assert isinstance(contract.tiled_values[1], list)
    assert len(contract.tiled_values[1]) == 3
    for recorded, original in zip(contract.tiled_values[1], w_list):
        assert torch.equal(recorded, original)


class _RaggedScatter(nn.Module):
    """Reference for a child that takes ``list[int]`` per-row sequence lengths."""
    def forward(self, x, num_token_list):
        out = torch.zeros_like(x)
        for i in range(x.shape[0]):
            n = num_token_list[i]
            out[i, :n] = x[i, :n] * 2.0
        return out


def test_stub_passes_list_of_int_through_unchanged():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_RaggedScatter(),
        arg_names=("x", "num_token_list"),
        arg_specs=(
            TensorArg(shape=(4, 8)),
            ListOfIntArg(length=4),
        ),
        recorder=rec,
    )
    x = torch.randn(4, 8)
    num_token_list = [2, 5, 3, 7]
    out = stub(x.reshape(2, 2, 4, 2), num_token_list,
               out_shapes=((2, 2, 4, 2),))

    contract = rec.contract
    assert contract.arg_specs[1] == ListOfIntArg(length=4)
    assert contract.tiled_values[1] == [2, 5, 3, 7]
    # tiled_values for the int list is a fresh copy, not the same reference.
    assert contract.tiled_values[1] is not num_token_list


def test_stub_rejects_non_tensor_for_tensor_arg():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        arg_specs=_tensor_specs((4, 8), (4, 8)),
        recorder=rec,
    )
    a = torch.randn(4, 8)
    bad_b = [torch.randn(4, 8)]   # parent passed a list where a tensor was declared
    try:
        stub(a, bad_b, out_shapes=((4, 8),))
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_stub_rejects_wrong_length_list():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_MoEExpertSum(),
        arg_names=("x", "w_list"),
        arg_specs=(
            TensorArg(shape=(1, 4, 8)),
            ListOfTensorArg(length=3, elem_shape=(8, 8)),
        ),
        recorder=rec,
    )
    x = torch.randn(1, 4, 8)
    w_list_short = [torch.randn(8, 8) for _ in range(2)]   # declared 3, got 2
    try:
        stub(x, w_list_short, out_shapes=((1, 4, 8),))
        raised = False
    except AssertionError:
        raised = True
    assert raised
