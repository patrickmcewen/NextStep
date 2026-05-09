import torch
import torch.nn as nn
from src.blackbox_stub import make_stub, ContractRecorder


class _Add(nn.Module):
    def forward(self, a, b):
        return a + b


def test_stub_preserves_reference_semantics():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        vanilla_shapes=((4, 8),) * 2,
        recorder=rec,
    )
    a = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b = torch.ones(4, 8) * 10
    out = stub(a, b, out_shape=(4, 8))
    assert torch.equal(out, a + b)


def test_stub_handles_tile_input_via_flatten_reshape():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        vanilla_shapes=((4, 8),) * 2,
        recorder=rec,
    )
    a_vanilla = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b_vanilla = torch.ones(4, 8) * 10
    a_tiled = a_vanilla.reshape(2, 2, 4, 2)
    b_tiled = b_vanilla.reshape(2, 2, 4, 2)
    out = stub(a_tiled, b_tiled, out_shape=(2, 2, 4, 2))
    expected = (a_vanilla + b_vanilla).reshape(2, 2, 4, 2)
    assert torch.equal(out, expected)


def test_stub_applies_out_perm_then_reshape():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        vanilla_shapes=((4, 8),) * 2,
        recorder=rec,
    )
    a = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b = torch.zeros(4, 8)
    out = stub(a, b, out_shape=(8, 4), out_perm=(1, 0))
    assert torch.equal(out, a.permute(1, 0))


def test_recorder_captures_contract_on_first_call():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        vanilla_shapes=((4, 8),) * 2,
        recorder=rec,
    )
    a = torch.randn(2, 2, 4, 2)
    b = torch.randn(2, 2, 4, 2)
    stub(a, b, out_shape=(2, 2, 4, 2))
    contract = rec.contract
    assert contract is not None
    assert contract.arg_names == ("a", "b")
    assert contract.tiled_shapes == ((2, 2, 4, 2), (2, 2, 4, 2))
    assert torch.equal(contract.tiled_values[0], a)
    assert contract.out_shape == (2, 2, 4, 2)
    assert contract.out_perm is None


def test_recorder_only_captures_first_call():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_Add(),
        arg_names=("a", "b"),
        vanilla_shapes=((4, 8),) * 2,
        recorder=rec,
    )
    a = torch.randn(2, 2, 4, 2)
    b = torch.randn(2, 2, 4, 2)
    stub(a, b, out_shape=(2, 2, 4, 2))
    a2 = torch.randn(2, 2, 4, 2)
    stub(a2, b, out_shape=(2, 2, 4, 2))
    assert torch.equal(rec.contract.tiled_values[0], a)   # first call's value
