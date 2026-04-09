"""PyTorch reference: SiLU (Swish) activation function.

Computes x * sigmoid(x) — the activation used in gated MLPs
(Qwen, LLama, Mixtral gate projections).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
