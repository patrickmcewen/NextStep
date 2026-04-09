"""PyTorch reference: Scaled Dot-Product Attention (core compute).

Computes: output = softmax(Q @ K^T) @ V
  where softmax is exp-normalize (no max-subtraction), matching the STeP
  flash-attention implementation from step_tl/end_to_end/attention/flashattn.py
  and step_tl/dynamic_par/flashattn.py (stages 8-11).

Shapes:
  Q: [M, D]   — M query vectors of dimension D
  K: [N, D]   — N key vectors
  V: [N, D]   — N value vectors
  output: [M, D]
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        # Q: [M, D], K: [N, D], V: [N, D]
        scores = Q @ K.T                          # [M, N]
        exp_scores = torch.exp(scores)             # [M, N]
        context = exp_scores @ V                   # [M, D]
        norm = exp_scores.sum(dim=-1, keepdim=True)  # [M, 1]
        return context / norm                      # [M, D]


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, N, D = dims["M"], dims["N"], dims["D"]
    Q = torch.randn(M, D)
    K = torch.randn(N, D)
    V = torch.randn(N, D)
    return Q, K, V


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    Q, K, V = get_inputs(dims)
    return model(Q, K, V)
