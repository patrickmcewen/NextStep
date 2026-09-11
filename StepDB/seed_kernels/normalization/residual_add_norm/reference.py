"""PyTorch reference: Residual add + RMS norm.

Computes rms_norm(x + residual) where
  rms_norm(y) = y * rsqrt(mean(y^2, dim=-1) + eps).
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X, R):
        y = X + R
        norm = y.pow(2).mean(dim=-1, keepdim=True)
        return y * torch.rsqrt(norm + self.eps)


def get_init_inputs(dims):
    return [dims.get("eps", 1e-6)]


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K), torch.randn(M, K)]


def compute_gold(dims):
    return Model(*get_init_inputs(dims))(*get_inputs(dims))
