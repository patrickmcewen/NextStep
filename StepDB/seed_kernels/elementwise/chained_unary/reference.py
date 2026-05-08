"""PyTorch reference: Chained unary operations.

Computes rsqrt(silu(exp(x^2))) element-wise.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch
import torch.nn.functional as F


def compute_gold(dims, tensors):
    x = tensors["input"]
    return torch.rsqrt(F.silu(torch.exp(x.pow(2))))
