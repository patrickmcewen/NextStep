"""PyTorch reference: Outer product with batch accumulation.

Computes sum over batch dimension of outer products: sum_b(a[b,:] outer b[b,:]).
Equivalent to A^T @ B where A is [B, M] and B is [B, N].
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        # A: [B, M], B: [B, N]
        # A^T @ B = [M, B] @ [B, N] = [M, N]
        return A.T @ B


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    M, N = dims["M"], dims["N"]
    A = torch.randn(B, M)
    B_data = torch.randn(B, N)
    return A, B_data


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
