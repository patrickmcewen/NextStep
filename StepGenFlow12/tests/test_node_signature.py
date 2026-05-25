import pytest
import torch
import torch.nn as nn

from src.node_signature import (
    IntArg,
    ListOfIntArg,
    ListOfTensorArg,
    TensorArg,
    classify_arg,
    extract_signature,
)

REF_CODE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(8, 8))

    def forward(self, x, y):
        z = x @ self.W
        return z + y
"""


def test_extract_signature_arg_names_and_shapes():
    canonical_inputs = {
        "x": torch.randn(2, 8),
        "y": torch.randn(2, 8),
    }
    sig = extract_signature(REF_CODE, canonical_inputs)
    assert sig.arg_names == ("x", "y")
    assert sig.arg_shapes == ((2, 8), (2, 8))
    assert sig.out_shapes == ((2, 8),)
    assert sig.out_is_tuple is False
    assert "W" in sig.weight_names


def test_extract_signature_rejects_missing_input():
    canonical_inputs = {"x": torch.randn(2, 8)}   # missing y
    try:
        extract_signature(REF_CODE, canonical_inputs)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "extract_signature should fail-loud when a forward arg has no canonical input"


TUPLE_REF_CODE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        # Mirrors the QKV-split style: same input feeds three different reshapes.
        S, HKV, QPKV, D = 4, 4, 4, 8
        Qh = x.view(S, HKV, QPKV, D).permute(1, 2, 0, 3)
        Kh = x.view(S, HKV * QPKV, D).permute(1, 0, 2).unsqueeze(1)
        Vh = x.view(S, HKV * QPKV, D).permute(1, 0, 2).unsqueeze(1)
        return Qh, Kh, Vh
"""


def test_extract_signature_accepts_tuple_return():
    canonical_inputs = {"x": torch.randn(4, 4 * 4 * 8)}
    sig = extract_signature(TUPLE_REF_CODE, canonical_inputs)
    assert sig.arg_names == ("x",)
    assert sig.out_is_tuple is True
    assert len(sig.out_shapes) == 3
    assert sig.out_shapes[0] == (4, 4, 4, 8)   # Qh after permute
    assert sig.out_shapes[1] == (4 * 4, 1, 4, 8)
    assert sig.out_shapes[2] == (4 * 4, 1, 4, 8)


def test_extract_signature_rejects_non_tensor_non_tuple():
    """A forward returning, say, a dict or list of non-tensors must fail loud."""
    bad_code = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return {'data': x}
"""
    try:
        extract_signature(bad_code, {"x": torch.randn(4, 4)})
        raised = False
    except AssertionError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# List-typed forward args (per-expert weight stacks, per-batch seq lengths)
# ---------------------------------------------------------------------------

def test_classify_arg_tensor():
    spec = classify_arg("x", torch.randn(4, 8))
    assert spec == TensorArg(shape=(4, 8))


def test_classify_arg_list_of_tensor_homogeneous():
    weights = [torch.randn(8, 16) for _ in range(3)]
    spec = classify_arg("w_gate_list", weights)
    assert spec == ListOfTensorArg(length=3, elem_shape=(8, 16))


def test_classify_arg_list_of_tensor_rejects_mismatched_shapes():
    weights = [torch.randn(8, 16), torch.randn(8, 32)]
    with pytest.raises(AssertionError, match="mismatched element shapes"):
        classify_arg("w_gate_list", weights)


def test_classify_arg_list_of_int():
    spec = classify_arg("num_token_list", [3, 7, 2, 5])
    assert spec == ListOfIntArg(length=4)


def test_classify_arg_rejects_mixed_list():
    with pytest.raises(AssertionError, match="mixed/unsupported element types"):
        classify_arg("bad", [torch.randn(4), 3])


def test_classify_arg_rejects_empty_list():
    with pytest.raises(AssertionError):
        classify_arg("empty", [])


def test_classify_arg_scalar_int():
    """Python ints are now first-class — used for head counts, tile bounds, etc."""
    assert classify_arg("n_head", 12) == IntArg()


def test_classify_arg_rejects_bool():
    """``bool`` is an ``int`` subclass in Python; we reject it because there's
    no ``BoolArg`` spec and silently classifying ``True`` as ``IntArg(1)`` would
    hide a planner bug behind a true-as-1 coercion."""
    with pytest.raises(AssertionError, match="unsupported type"):
        classify_arg("flag", True)


def test_classify_arg_rejects_scalar_tensor():
    """0-D ``torch.Tensor``s have no tile-stream representation — the planner
    must pass scalars as Python ints (IntArg) instead, or the deep-stack
    ``tiled shape () must be rank >= 2`` crash returns."""
    with pytest.raises(AssertionError, match="0-D torch.Tensor"):
        classify_arg("n_head_tensor", torch.tensor(12, dtype=torch.int64))


def test_classify_arg_rejects_1d_singleton_tensor():
    """1-D singleton tensors are scalar-wraps in disguise — the planner LLM
    reaches for ``torch.tensor([n_head])`` when the 0-D guard fires, and
    pass-1 then crashes in ``_wrap_on_chip_call_args`` with ``tiled shape (1,)
    must be rank >= 2`` (kernelbench_opt_1p3b outer_0). ``classify_arg`` must
    reject the same evasion the planner guard rejects, keeping the two
    layers in sync."""
    with pytest.raises(AssertionError, match="Python int"):
        classify_arg("n_head_tensor", torch.tensor([12], dtype=torch.int64))


def test_classify_arg_rejects_dict():
    with pytest.raises(AssertionError, match="unsupported type"):
        classify_arg("d", {"a": 1})


# Reference module that takes a Python int alongside a tensor — exercises the
# IntArg path end-to-end through ``extract_signature``.
INT_INPUT_REF = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, n_head):
        # n_head is a Python int — head count knob the parent passes down.
        B, T, D = x.shape
        D_head = D // n_head
        return x.view(B, T, n_head, D_head).sum(dim=2)
"""


def test_extract_signature_with_int_input():
    canonical_inputs = {"x": torch.randn(2, 4, 12), "n_head": 4}
    sig = extract_signature(INT_INPUT_REF, canonical_inputs)
    assert sig.arg_names == ("x", "n_head")
    assert sig.arg_specs == (TensorArg(shape=(2, 4, 12)), IntArg())
    assert sig.out_shapes == ((2, 4, 12 // 4),)


# Reference module that takes the three supported arg kinds in one forward.
LIST_INPUT_REF = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x, w_gate_list, num_token_list):
        # Mirrors the end_to_end pattern: per-expert weight loop + ragged
        # per-row indexing via a Python int list.
        out = torch.zeros_like(x)
        for e_idx, w in enumerate(w_gate_list):
            out = out + x @ w @ w.T
        for i in range(x.shape[0]):
            out[i, :num_token_list[i]] = out[i, :num_token_list[i]] * 2.0
        return out
"""


def test_extract_signature_with_list_inputs():
    canonical_inputs = {
        "x": torch.randn(4, 8),
        "w_gate_list": [torch.randn(8, 8) for _ in range(3)],
        "num_token_list": [2, 5, 3, 7],
    }
    sig = extract_signature(LIST_INPUT_REF, canonical_inputs)
    assert sig.arg_names == ("x", "w_gate_list", "num_token_list")
    assert sig.arg_specs == (
        TensorArg(shape=(4, 8)),
        ListOfTensorArg(length=3, elem_shape=(8, 8)),
        ListOfIntArg(length=4),
    )
    assert sig.out_shapes == ((4, 8),)


def test_arg_shapes_property_fails_loud_on_list_args():
    """The legacy ``arg_shapes`` accessor is tensor-only — list specs raise.

    Downstream consumers (make_stub, Pass-1 prompt rendering of child arg
    shapes) don't yet know how to format list specs; a clean assertion here
    is the right failure mode until they do.
    """
    canonical_inputs = {
        "x": torch.randn(4, 8),
        "w_gate_list": [torch.randn(8, 8) for _ in range(3)],
        "num_token_list": [2, 5, 3, 7],
    }
    sig = extract_signature(LIST_INPUT_REF, canonical_inputs)
    with pytest.raises(AssertionError, match="arg_shapes is only valid"):
        _ = sig.arg_shapes
