"""PyTorch reference: Scaled Dot-Product Attention with 1/sqrt(D).

Computes softmax(Q @ K^T / sqrt(D)) @ V using exp-normalize (no max subtraction).
"""
import torch
import torch.nn as nn
import math

SEED = 42


class Model(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.scale = 1.0 / math.sqrt(D)

    def forward(self, Q, K, V):
        scores = Q @ K.T * self.scale
        exp_scores = torch.exp(scores)
        context = exp_scores @ V
        norm = exp_scores.sum(dim=-1, keepdim=True)
        return context / norm


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, N, D = dims["M"], dims["N"], dims["D"]
    Q = torch.randn(M, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)
    return Q, K, V


def get_init_inputs(dims):
    return [dims["D"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
