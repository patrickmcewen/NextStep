"""PyTorch reference: Two-pass numerically-stable attention.

Computes softmax(Q @ K^T - mean(Q @ K^T)) @ V.
The mean subtraction improves numerical stability vs raw exp.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        scores = Q @ K.T                             # [M, N]
        shift = scores.mean(dim=-1, keepdim=True)     # [M, 1]
        exp_scores = torch.exp(scores - shift)        # [M, N]
        context = exp_scores @ V                      # [M, D]
        norm = exp_scores.sum(dim=-1, keepdim=True)   # [M, 1]
        return context / norm                         # [M, D]


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, N, D = dims["M"], dims["N"], dims["D"]
    Q = torch.randn(M, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)
    return Q, K, V


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
