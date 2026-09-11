"""PyTorch reference: Double bufferize chain.

Computes silu(x^2) - the bufferize/streamify round-trips are identity.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch.nn.functional as F


def compute_gold(dims, tensors):
    return F.silu(tensors["input"].pow(2))
