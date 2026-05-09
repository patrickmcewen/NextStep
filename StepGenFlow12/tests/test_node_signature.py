import torch
import torch.nn as nn
from src.node_signature import extract_signature

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
