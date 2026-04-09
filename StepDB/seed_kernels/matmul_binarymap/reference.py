"""PyTorch reference: MatMul via raw BinaryMap (no Linear kernel abstraction)."""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, W):
        return torch.matmul(A, W)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, H = dims["B"], dims["H"]
    A = torch.randn(B, H)
    model = torch.nn.Linear(H, H, bias=False)
    W = model.weight.T.detach().clone().contiguous()  # [H, H]
    return A, W


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    A, W = get_inputs(dims)
    return torch.matmul(A, W)
