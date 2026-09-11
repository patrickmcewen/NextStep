"""PyTorch reference: GEMM of two row-wise RMS-normalized matrices.

Computes matmul(rms_norm(A), rms_norm(B)) where
  rms_norm(x) = x * rsqrt(mean(x^2, dim=-1) + eps).
A is [M, K] normalized along K; B is [K, N] normalized along N.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def _rms_norm(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def compute_gold(dims, tensors):
    eps = tensors["eps"]
    A_n = _rms_norm(tensors["A"], eps)
    B_n = _rms_norm(tensors["B"], eps)
    return torch.matmul(A_n, B_n)
