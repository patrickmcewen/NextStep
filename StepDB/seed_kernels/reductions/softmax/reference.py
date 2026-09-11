"""PyTorch reference: Row-wise softmax (exp-normalize, no max subtraction).

Computes exp(x) / sum(exp(x), dim=-1, keepdim=True).
No max-subtraction for numerical stability - matches the STeP implementation.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    e = torch.exp(tensors["input"])
    return e / e.sum(dim=-1, keepdim=True)
