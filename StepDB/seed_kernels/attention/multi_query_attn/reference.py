"""PyTorch reference: Multi-Query Attention.

H query heads share a single K and V.
  output[h] = softmax(Q[h] @ K^T) @ V  for each head h.
Uses exp-normalize (no max subtraction).
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, H, M):
        # Q: [H*M, D], K: [N, D], V: [N, D]
        Q_heads = Q.view(H, M, -1)  # [H, M, D]
        # Compute attention per head
        scores = Q_heads @ K.T  # [H, M, N]
        exp_scores = torch.exp(scores)
        context = exp_scores @ V  # [H, M, D]
        norm = exp_scores.sum(dim=-1, keepdim=True)  # [H, M, 1]
        return (context / norm).reshape(H * M, -1)  # [H*M, D]


def get_inputs(dims):
    torch.manual_seed(SEED)
    H = dims["H"]
    M, N, D = dims["M"], dims["N"], dims["D"]
    Q = torch.randn(H * M, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)
    return Q, K, V, H, M


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
