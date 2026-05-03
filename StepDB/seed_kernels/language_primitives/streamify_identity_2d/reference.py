"""PyTorch reference: rank-2 Bufferize + Streamify identity roundtrip.

Loads a 2D tensor, bufferizes it as a rank-2 tile-grid (entire tensor goes
into one Buffer), then streamifies it back out in row-major tile order.
The new Streamify (stride, out_shape_tiled) interface is exercised with a
multi-dim out_shape_tiled and non-trivial row-major stride.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
