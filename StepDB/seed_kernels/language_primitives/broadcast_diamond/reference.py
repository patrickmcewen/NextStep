"""PyTorch reference: Broadcast diamond pattern.

Computes x^2 + silu(x) + exp(x).
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch
import torch.nn.functional as F


def compute_gold(dims, tensors):
    x = tensors["input"]
    return x.pow(2) + F.silu(x) + torch.exp(x)
