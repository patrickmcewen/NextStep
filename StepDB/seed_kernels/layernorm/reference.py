"""PyTorch reference: Layer Normalization (row-wise, no learnable params).

Computes (x - mean(x)) * rsqrt(var(x) + eps) per row.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return [dims.get("eps", 1e-6)]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
