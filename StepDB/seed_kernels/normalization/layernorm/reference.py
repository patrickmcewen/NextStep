"""PyTorch reference: Layer Normalization (row-wise, no learnable params).

Computes (x - mean(x)) * rsqrt(var(x) + eps) per row.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    x = tensors["input"]
    eps = tensors["eps"]
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps)
