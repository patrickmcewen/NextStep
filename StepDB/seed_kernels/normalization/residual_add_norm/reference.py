"""PyTorch reference: Residual add + RMS norm.

Computes rms_norm(x + residual) where
  rms_norm(y) = y * rsqrt(mean(y^2, dim=-1) + eps).
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    y = tensors["X"] + tensors["R"]
    norm = y.pow(2).mean(dim=-1, keepdim=True)
    return y * torch.rsqrt(norm + tensors["eps"])
