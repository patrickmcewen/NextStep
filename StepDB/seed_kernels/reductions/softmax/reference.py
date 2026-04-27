"""PyTorch reference: Row-wise softmax (exp-normalize, no max subtraction).

Computes exp(x) / sum(exp(x), dim=-1, keepdim=True).
No max-subtraction for numerical stability — matches the STeP implementation.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        e = torch.exp(x)
        return e / e.sum(dim=-1, keepdim=True)


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
