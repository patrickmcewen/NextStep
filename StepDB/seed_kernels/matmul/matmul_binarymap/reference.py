"""PyTorch reference: MatMul via raw BinaryMap (no Linear kernel abstraction).

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    return torch.matmul(tensors["A"], tensors["W"])
