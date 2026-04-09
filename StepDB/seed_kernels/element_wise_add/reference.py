"""PyTorch reference: Element-wise addition (residual connection).

Computes A + B — used for residual add in transformer decoder layers
after attention and MoE blocks.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return A + B


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    A = torch.randn(M, K)
    B = torch.randn(M, K)
    return A, B


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    A, B = get_inputs(dims)
    return A + B
