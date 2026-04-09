"""PyTorch reference: Chained unary operations.

Computes rsqrt(silu(exp(x^2))) element-wise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rsqrt(F.silu(torch.exp(x.pow(2))))


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.rand(M, K) * 0.5 + 0.1]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
