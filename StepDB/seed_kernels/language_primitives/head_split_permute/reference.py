"""PyTorch reference: head split + permute, (S, H*D) -> (H, S*D).

Splits the trailing axis of a (S, H*D) tensor into (S, H, D), permutes to
(H, S, D), then flattens trailing dims to match offchip_store's 2D layout.
This is the canonical "expose head axis as outer stream dim" pattern.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H

    def forward(self, x):
        S, HD = x.shape
        D = HD // self.H
        return x.view(S, self.H, D).permute(1, 0, 2).reshape(self.H, S * D)


def get_inputs(dims):
    torch.manual_seed(SEED)
    S, H, D = dims["S"], dims["H"], dims["D"]
    return [torch.randn(S, H * D)]


def get_init_inputs(dims):
    return [dims["H"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
