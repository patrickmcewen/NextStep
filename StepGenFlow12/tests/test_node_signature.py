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
    assert sig.out_shape == (2, 8)
    assert "W" in sig.weight_names


def test_extract_signature_rejects_missing_input():
    canonical_inputs = {"x": torch.randn(2, 8)}   # missing y
    try:
        extract_signature(REF_CODE, canonical_inputs)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "extract_signature should fail-loud when a forward arg has no canonical input"
