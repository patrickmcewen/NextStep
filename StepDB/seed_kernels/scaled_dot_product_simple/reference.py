"""PyTorch reference: Simplified scaled dot-product.

Computes (Q @ K^T) / sqrt(D).
"""
import torch
import torch.nn as nn
import math

SEED = 42


class Model(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.scale = 1.0 / math.sqrt(D)

    def forward(self, Q, K):
        return self.scale * (Q @ K.T)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, N, D = dims["M"], dims["N"], dims["D"]
    Q = torch.randn(M, D)
    K = torch.randn(N, D)
    return Q, K


def get_init_inputs(dims):
    return [dims["D"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
