"""PyTorch reference: 2D matrix transpose via tile-grid permutation.

Transposes (M, K) -> (K, M) with rectangular tiles (tile_m != tile_k),
exercising both the per-tile transpose at load and the tile-grid axis
swap at streamify.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.t().contiguous()


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
