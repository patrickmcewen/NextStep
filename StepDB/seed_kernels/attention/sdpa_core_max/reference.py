"""PyTorch reference: Scaled Dot-Product Attention (core compute, safe softmax).

Same I/O as ``sdpa_core`` but uses the numerically-stable softmax form with
per-row max subtraction:
    output = softmax(Q @ K^T) @ V
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

Shapes:
  Q: [M, D]   - M query vectors of dimension D
  K: [N, D]   - N key vectors
  V: [N, D]   - N value vectors
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
        scores = Q @ K.T                                     # [M, N]
        row_max = scores.max(dim=-1, keepdim=True).values    # [M, 1]
        exp_scores = torch.exp(scores - row_max)             # [M, N]
        context = exp_scores @ V                             # [M, D]
        norm = exp_scores.sum(dim=-1, keepdim=True)          # [M, 1]
        return context / norm                                # [M, D]


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
