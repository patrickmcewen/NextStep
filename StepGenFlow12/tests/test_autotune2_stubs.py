"""Unit tests for autotune2 variant stubs + registry I/O."""

import ast

import pytest
import torch
import torch.nn as nn

from src.autotune2.contracts import TensorContract, vanilla_contract_for
from src.autotune2.stubs import (
    _inverse_permutation,
    apply_output_contract,
    emit_variants_module,
    invert_input_contract,
    load_variants_module,
    make_variant_stub,
)
from src.node_signature import ListOfIntArg, ListOfTensorArg, TensorArg
from src.step_dsl import StepTensor


# --- adapter primitives --------------------------------------------------------


def test_inverse_permutation_identity():
    assert _inverse_permutation((0, 1, 2)) == (0, 1, 2)


def test_inverse_permutation_arbitrary():
    perm = (2, 0, 3, 1)
    inv = _inverse_permutation(perm)
    # composing perm . inv (apply inv then perm) should be identity
    composed = tuple(perm[inv[i]] for i in range(len(perm)))
    assert composed == tuple(range(len(perm)))


def test_invert_input_identity():
    vanilla_shape = (4, 6, 8)
    contract = vanilla_contract_for(vanilla_shape)
    x = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)
    out = invert_input_contract(x, vanilla_shape, contract)
    assert out.shape == vanilla_shape
    assert torch.equal(out, x)


def test_invert_input_with_permute():
    vanilla_shape = (4, 6, 8)
    contract = TensorContract(reshape=(4, 6, 8), permutation=(2, 0, 1))
    vanilla = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)
    contracted = vanilla.permute(2, 0, 1).contiguous()  # what the parent's DSL emits
    recovered = invert_input_contract(contracted, vanilla_shape, contract)
    assert recovered.shape == vanilla_shape
    assert torch.equal(recovered, vanilla)


def test_invert_input_with_reshape_and_permute():
    vanilla_shape = (12, 8)
    # split 12 -> (3, 4), permute to (4, 3, 8)
    contract = TensorContract(reshape=(3, 4, 8), permutation=(1, 0, 2))
    vanilla = torch.arange(96, dtype=torch.float32).reshape(vanilla_shape)
    contracted = vanilla.reshape(3, 4, 8).permute(1, 0, 2).contiguous()
    recovered = invert_input_contract(contracted, vanilla_shape, contract)
    assert torch.equal(recovered, vanilla)


def test_invert_input_size_mismatch_asserts():
    contract = vanilla_contract_for((4, 6))
    bad = torch.zeros(5, 6)
    with pytest.raises(AssertionError, match="element count"):
        invert_input_contract(bad, (4, 6), contract)


def test_apply_output_identity():
    vanilla_shape = (4, 6, 8)
    contract = vanilla_contract_for(vanilla_shape)
    x = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)
    out = apply_output_contract(x, contract, vanilla_shape)
    assert torch.equal(out, x)


def test_apply_output_with_permute_round_trip():
    """apply_output ∘ invert_input == identity for any contract."""
    vanilla_shape = (4, 6, 8)
    contract = TensorContract(reshape=(2, 2, 6, 8), permutation=(2, 0, 3, 1))
    vanilla = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)
    contracted = vanilla.reshape(2, 2, 6, 8).permute(2, 0, 3, 1).contiguous()
    recovered = invert_input_contract(contracted, vanilla_shape, contract)
    round_tripped = apply_output_contract(recovered, contract, contracted.shape)
    assert torch.equal(round_tripped, contracted)


# --- make_variant_stub end-to-end ----------------------------------------------


class _Double(nn.Module):
    def forward(self, x):
        return x * 2.0


class _TwoOut(nn.Module):
    def forward(self, x):
        return x + 1.0, x * 3.0


def test_variant_stub_identity_matches_ref():
    """Identity contracts ⇒ stub output equals ref_module(vanilla)."""
    vanilla_shape = (4, 6, 8)
    out_shape = (8, 4, 6)  # caller picks a tile-stream rank-3 view
    x = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)

    stub = make_variant_stub(
        ref_module=_Double(),
        arg_names=("x",),
        arg_specs=(TensorArg(shape=vanilla_shape),),
        output_names=("out_0",),
        input_contracts={},
        output_contracts={},
        variant_name="double_v0",
    )
    # Tile-stream input has same flat layout as vanilla under identity contract.
    tiled_in = x.reshape(out_shape)
    out = stub(tiled_in, out_shapes=(out_shape,))
    assert isinstance(out, StepTensor)
    expected = (x * 2.0).reshape(out_shape)
    assert torch.equal(out.underlying_tensor, expected)


def test_variant_stub_permute_round_trip_identity_ref():
    """Pass contracted tensor through nn.Identity-wrapped variant: returns input."""
    vanilla_shape = (4, 6, 8)
    contract = TensorContract(reshape=(4, 6, 8), permutation=(2, 0, 1))
    contracted_shape = contract.post_permute_shape()  # (8, 4, 6)
    x = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)
    contracted = x.permute(2, 0, 1).contiguous()

    stub = make_variant_stub(
        ref_module=nn.Identity(),
        arg_names=("x",),
        arg_specs=(TensorArg(shape=vanilla_shape),),
        output_names=("out_0",),
        input_contracts={"x": contract},
        output_contracts={"out_0": contract},
        variant_name="identity_perm_v1",
    )
    out = stub(contracted, out_shapes=(contracted_shape,))
    assert torch.equal(out.underlying_tensor, contracted)


def test_variant_stub_permute_compute_module():
    """Variant on a real compute module: result equals contracted(ref(vanilla))."""
    vanilla_shape = (4, 6, 8)
    contract = TensorContract(reshape=(4, 6, 8), permutation=(2, 0, 1))
    contracted_shape = contract.post_permute_shape()
    x = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)
    contracted = x.permute(2, 0, 1).contiguous()

    stub = make_variant_stub(
        ref_module=_Double(),
        arg_names=("x",),
        arg_specs=(TensorArg(shape=vanilla_shape),),
        output_names=("out_0",),
        input_contracts={"x": contract},
        output_contracts={"out_0": contract},
        variant_name="double_perm_v1",
    )
    out = stub(contracted, out_shapes=(contracted_shape,))
    expected = (x * 2.0).permute(2, 0, 1).contiguous()
    assert torch.equal(out.underlying_tensor, expected)


def test_variant_stub_step_tensor_input_unwraps():
    """Wrapping the input as a StepTensor is transparently handled."""
    vanilla_shape = (4, 6, 8)
    out_shape = (8, 4, 6)
    x = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)
    tiled_in = x.reshape(out_shape)
    from src.step_dsl import Tile, _elem_from_torch
    st_in = StepTensor(tiled_in, stream_dtype=Tile(_elem_from_torch(tiled_in.dtype), (4, 6)))

    stub = make_variant_stub(
        ref_module=_Double(),
        arg_names=("x",),
        arg_specs=(TensorArg(shape=vanilla_shape),),
        output_names=("out_0",),
        input_contracts={},
        output_contracts={},
        variant_name="double_v0_st",
    )
    out = stub(st_in, out_shapes=(out_shape,))
    expected = (x * 2.0).reshape(out_shape)
    assert torch.equal(out.underlying_tensor, expected)


def test_variant_stub_multi_output_tuple():
    """Tuple-returning ref propagates through as a tuple of StepTensors."""
    vanilla_shape = (4, 6, 8)
    out_shape = (8, 4, 6)
    x = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)

    stub = make_variant_stub(
        ref_module=_TwoOut(),
        arg_names=("x",),
        arg_specs=(TensorArg(shape=vanilla_shape),),
        output_names=("out_0", "out_1"),
        input_contracts={},
        output_contracts={},
        variant_name="twoout_v0",
    )
    tiled_in = x.reshape(out_shape)
    out = stub(tiled_in, out_shapes=(out_shape, out_shape))
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.equal(out[0].underlying_tensor, (x + 1.0).reshape(out_shape))
    assert torch.equal(out[1].underlying_tensor, (x * 3.0).reshape(out_shape))


def test_variant_stub_list_arg_passthrough():
    """List args pass through untouched (no contract axis)."""

    class _SumList(nn.Module):
        def forward(self, x, ks):
            return x + sum(ks)

    vanilla_shape = (4, 6, 8)
    out_shape = (8, 4, 6)
    x = torch.arange(192, dtype=torch.float32).reshape(vanilla_shape)
    stub = make_variant_stub(
        ref_module=_SumList(),
        arg_names=("x", "ks"),
        arg_specs=(TensorArg(shape=vanilla_shape), ListOfIntArg(length=3)),
        output_names=("out_0",),
        input_contracts={},
        output_contracts={},
        variant_name="sumlist_v0",
    )
    out = stub(x.reshape(out_shape), [1, 2, 3], out_shapes=(out_shape,))
    expected = (x + 6.0).reshape(out_shape)
    assert torch.equal(out.underlying_tensor, expected)


# --- validation --------------------------------------------------------------


def test_make_variant_stub_rejects_contract_on_list_arg():
    with pytest.raises(AssertionError, match="list arg"):
        make_variant_stub(
            ref_module=nn.Identity(),
            arg_names=("ks",),
            arg_specs=(ListOfIntArg(length=2),),
            output_names=("out_0",),
            input_contracts={"ks": vanilla_contract_for((2,))},
            output_contracts={},
            variant_name="bad",
        )


def test_make_variant_stub_rejects_unknown_input_key():
    with pytest.raises(AssertionError, match="input_contracts key"):
        make_variant_stub(
            ref_module=nn.Identity(),
            arg_names=("x",),
            arg_specs=(TensorArg(shape=(4,)),),
            output_names=("out_0",),
            input_contracts={"y": vanilla_contract_for((4,))},
            output_contracts={},
            variant_name="bad",
        )


def test_make_variant_stub_rejects_unknown_output_key():
    with pytest.raises(AssertionError, match="output_contracts key"):
        make_variant_stub(
            ref_module=nn.Identity(),
            arg_names=("x",),
            arg_specs=(TensorArg(shape=(4,)),),
            output_names=("out_0",),
            input_contracts={},
            output_contracts={"out_99": vanilla_contract_for((4,))},
            variant_name="bad",
        )


def test_variant_stub_rejects_wrong_out_shapes_arity():
    stub = make_variant_stub(
        ref_module=_Double(),
        arg_names=("x",),
        arg_specs=(TensorArg(shape=(4, 6)),),
        output_names=("out_0",),
        input_contracts={},
        output_contracts={},
        variant_name="bad_call",
    )
    with pytest.raises(AssertionError, match="out_shapes"):
        stub(torch.zeros(4, 6), out_shapes=((4, 6), (4, 6)))


# --- registry I/O ------------------------------------------------------------


def test_emit_variants_module_syntactic_validity(tmp_path):
    out_path = tmp_path / "child" / "variants.py"
    variants = {
        0: {
            "input_contracts": {"x": vanilla_contract_for((4, 6, 8))},
            "output_contracts": {"out_0": vanilla_contract_for((4, 6, 8))},
        },
        3: {
            "input_contracts": {
                "x": TensorContract(reshape=(4, 6, 8), permutation=(2, 0, 1)),
            },
            "output_contracts": {
                "out_0": TensorContract(reshape=(2, 3, 8, 8), permutation=(0, 2, 1, 3)),
            },
        },
    }
    emit_variants_module(out_path=out_path, child_name="attention_block", variants=variants)
    src = out_path.read_text()
    ast.parse(src)  # syntactic compile-check; raises SyntaxError on failure


def test_emit_then_load_round_trip(tmp_path):
    out_path = tmp_path / "child" / "variants.py"
    variants = {
        0: {
            "input_contracts": {},
            "output_contracts": {"out_0": vanilla_contract_for((4, 6, 8))},
        },
        5: {
            "input_contracts": {
                "q": TensorContract(reshape=(8, 6, 4), permutation=(1, 0, 2)),
            },
            "output_contracts": {
                "out_0": TensorContract(reshape=(4, 48), permutation=(1, 0)),
            },
        },
    }
    emit_variants_module(out_path=out_path, child_name="gqa_attention", variants=variants)
    loaded = load_variants_module(out_path)
    assert set(loaded.keys()) == {0, 5}
    assert loaded[0]["input_contracts"] == {}
    assert loaded[0]["output_contracts"]["out_0"] == vanilla_contract_for((4, 6, 8))
    assert loaded[5]["input_contracts"]["q"] == TensorContract(
        reshape=(8, 6, 4), permutation=(1, 0, 2)
    )
    assert loaded[5]["output_contracts"]["out_0"] == TensorContract(
        reshape=(4, 48), permutation=(1, 0)
    )


def test_emit_variants_module_rejects_non_py(tmp_path):
    with pytest.raises(AssertionError, match=".py file"):
        emit_variants_module(
            out_path=tmp_path / "variants.txt",
            child_name="x",
            variants={},
        )


def test_emit_variants_module_rejects_bad_entry_shape(tmp_path):
    with pytest.raises(AssertionError, match="input_contracts.*output_contracts"):
        emit_variants_module(
            out_path=tmp_path / "variants.py",
            child_name="x",
            variants={0: {"input_contracts": {}}},  # missing output_contracts
        )
