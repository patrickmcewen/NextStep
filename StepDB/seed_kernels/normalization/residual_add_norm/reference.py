"""PyTorch reference: Residual add + RMS norm.

Computes rms_norm(x + residual) where
  rms_norm(y) = y * rsqrt(mean(y^2, dim=-1) + eps).
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, residual):
        y = x + residual
        norm = y.pow(2).mean(dim=-1, keepdim=True)
        return y * torch.rsqrt(norm + self.eps)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    X = torch.randn(M, K)
    R = torch.randn(M, K)
    return X, R


def get_init_inputs(dims):
    return [dims.get("eps", 1e-6)]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
