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

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    Q = tensors["Q"]
    K = tensors["K"]
    V = tensors["V"]
    scores = Q @ K.T
    row_max = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - row_max)
    context = exp_scores @ V
    norm = exp_scores.sum(dim=-1, keepdim=True)
    return context / norm
