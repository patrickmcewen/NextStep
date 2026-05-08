"""PyTorch reference: Scaled Dot-Product Attention (core compute).

Computes: output = softmax(Q @ K^T) @ V
  where softmax is exp-normalize (no max-subtraction), matching the STeP
  flash-attention implementation from step_tl/end_to_end/attention/flashattn.py
  and step_tl/dynamic_par/flashattn.py (stages 8-11).

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
    exp_scores = torch.exp(scores)
    context = exp_scores @ V
    norm = exp_scores.sum(dim=-1, keepdim=True)
    return context / norm
