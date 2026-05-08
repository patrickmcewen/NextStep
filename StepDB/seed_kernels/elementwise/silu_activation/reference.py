"""PyTorch reference: SiLU (Swish) activation function.

Computes x * sigmoid(x) — the activation used in gated MLPs
(Qwen, LLama, Mixtral gate projections).

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch.nn.functional as F


def compute_gold(dims, tensors):
    return F.silu(tensors["input"])
