"""PyTorch reference: RMS Normalization (row-wise).

Computes x * rsqrt(mean(x^2, dim=-1) + eps) - the normalization used
in Qwen/LLama-style transformer layers before QKV projection.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    x = tensors["input"]
    norm = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(norm + tensors["eps"])
