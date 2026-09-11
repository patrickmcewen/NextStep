"""PyTorch reference: 4D axis permute, (B, H, S, D) -> (B*S, H*D).

The canonical "move heads past sequence" rearrangement that appears at
the boundary between QKV-projection layouts (B, H, S, D) and the
attention-output layout (B, S, H, D). Output is flattened to 2D to match
offchip_store's collapse.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, H, S, D = x.shape
        return x.permute(0, 2, 1, 3).reshape(B * S, H * D)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, H, S, D = dims["B"], dims["H"], dims["S"], dims["D"]
    return [torch.randn(B, H, S, D)]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    inputs = get_inputs(dims)
    return model(*inputs)
