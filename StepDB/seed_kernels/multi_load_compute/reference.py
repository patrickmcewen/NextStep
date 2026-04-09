"""PyTorch reference: Multiple loads feeding compute.

Computes A*B + C*D element-wise.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B, C, D):
        return A * B + C * D


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    A = torch.randn(M, K)
    B = torch.randn(M, K)
    C = torch.randn(M, K)
    D = torch.randn(M, K)
    return A, B, C, D


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
