import torch
import torch.nn as nn
from src.blackbox_stub import make_stub, ContractRecorder


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
        vanilla_shapes=((4, 8),) * 2,
        recorder=rec,
    )
    a = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    b = torch.ones(4, 8) * 10
    out = stub(a, b, out_shapes=((4, 8),))
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
    out = stub(a_tiled, b_tiled, out_shapes=((2, 2, 4, 2),))
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
    out = stub(a, b, out_shapes=((8, 4),), out_perms=((1, 0),))
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
    stub(a, b, out_shapes=((2, 2, 4, 2),))
    contract = rec.contract
    assert contract is not None
    assert contract.arg_names == ("a", "b")
    assert contract.tiled_shapes == ((2, 2, 4, 2), (2, 2, 4, 2))
    assert torch.equal(contract.tiled_values[0], a)
    assert contract.out_shapes == ((2, 2, 4, 2),)
    assert contract.out_perms == (None,)


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
    stub(a, b, out_shapes=((2, 2, 4, 2),))
    a2 = torch.randn(2, 2, 4, 2)
    stub(a2, b, out_shapes=((2, 2, 4, 2),))
    assert torch.equal(rec.contract.tiled_values[0], a)   # first call's value


def test_stub_returns_tuple_for_multi_output_ref():
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_SplitQKV(),
        arg_names=("x",),
        vanilla_shapes=((4, 24),),
        recorder=rec,
    )
    x = torch.arange(96, dtype=torch.float32).reshape(4, 24)
    q, k, v = stub(x, out_shapes=((4, 8), (4, 8), (4, 8)))
    assert torch.equal(q, x[:, :8])
    assert torch.equal(k, x[:, 8:16])
    assert torch.equal(v, x[:, 16:24])
    assert rec.contract.out_shapes == ((4, 8), (4, 8), (4, 8))
    assert rec.contract.out_perms == (None, None, None)


def test_stub_rejects_output_count_mismatch():
    """Asking for 2 outputs from a 3-output ref must fail loudly."""
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_SplitQKV(),
        arg_names=("x",),
        vanilla_shapes=((4, 24),),
        recorder=rec,
    )
    x = torch.randn(4, 24)
    try:
        stub(x, out_shapes=((4, 8), (4, 8)))
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_stub_per_output_perm_for_multi_output():
    """``out_perms`` is parallel to ``out_shapes``; ``None`` entries skip permute."""
    rec = ContractRecorder()
    stub = make_stub(
        ref_module=_SplitQKV(),
        arg_names=("x",),
        vanilla_shapes=((4, 24),),
        recorder=rec,
    )
    x = torch.arange(96, dtype=torch.float32).reshape(4, 24)
    # Permute only the second output (K): swap the two dims.
    q, k, v = stub(x, out_shapes=((4, 8), (8, 4), (4, 8)),
                   out_perms=(None, (1, 0), None))
    assert torch.equal(q, x[:, :8])
    assert torch.equal(k, x[:, 8:16].permute(1, 0))
    assert torch.equal(v, x[:, 16:24])
