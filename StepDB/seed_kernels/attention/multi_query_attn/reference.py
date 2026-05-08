"""PyTorch reference: Multi-Query Attention.

H query heads share a single K and V.
  output[h] = softmax(Q[h] @ K^T) @ V  for each head h.
Uses exp-normalize (no max subtraction).

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    H = dims["H"]
    M = dims["M"]
    Q = tensors["Q"]
    K = tensors["K"]
    V = tensors["V"]

    Q_heads = Q.view(H, M, -1)
    scores = Q_heads @ K.T
    exp_scores = torch.exp(scores)
    context = exp_scores @ V
    norm = exp_scores.sum(dim=-1, keepdim=True)
    return (context / norm).reshape(H * M, -1)
