"""PyTorch reference: General matrix multiplication.

Computes A @ B. Origin: consolidation of KernelBench level1/{1,2,6,7}.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    return torch.matmul(tensors["A"], tensors["B"])
