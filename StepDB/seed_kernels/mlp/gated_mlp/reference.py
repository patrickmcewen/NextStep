"""PyTorch reference: Gated MLP (SwiGLU variant).

Computes: down(silu(gate(x)) * up(x))

This is the core computation of each MoE expert in Qwen/Mixtral models.
Three linear projections (gate, up, down) with SiLU gating.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42


class Model(nn.Module):
    def __init__(self, D, F_dim):
        super().__init__()
        self.gate = nn.Linear(D, F_dim, bias=False)
        self.up = nn.Linear(D, F_dim, bias=False)
        self.down = nn.Linear(F_dim, D, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, D = dims["B"], dims["D"]
    return [torch.randn(B, D)]


def get_init_inputs(dims):
    return [dims["D"], dims["F"]]


def compute_gold(dims):
    # Match the exact RNG sequence used in step_impl.py:
    # seed -> gate_model -> up_model -> down_model -> x
    torch.manual_seed(SEED)
    D, F_dim = dims["D"], dims["F"]
    model = Model(D, F_dim)
    x = torch.randn(dims["B"], D)
    with torch.no_grad():
        return model(x)
