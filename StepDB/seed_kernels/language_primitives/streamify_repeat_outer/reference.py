"""PyTorch reference: rank-2 Bufferize + Streamify outer-repeat.

Loads a 2D tensor, bufferizes it as a rank-2 tile-grid, then streamifies
it back out R times via stride=0 in the leading dim. Tests the
broadcast/repeat pattern that's the most common Streamify usage.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, R):
        super().__init__()
        self.R = R

    def forward(self, x):
        # OffChipStore produces a 2D (R*M, K) layout — R copies of x stacked
        # vertically — matching the row-major tile-emission order of the new
        # Streamify with leading-dim stride=0.
        return x.repeat(self.R, 1)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return [dims["R"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
