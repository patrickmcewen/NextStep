"""PyTorch reference: Vector reduction (sum along K dimension).

Computes sum(A, dim=-1) but keeps the tile_k shape for comparison.
Actually computes A reshaped to (M//tile_m, K//tile_k, tile_m, tile_k)
then summed over dim=1 and reshaped back.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A):
        return A


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    return [torch.randn(M, K)]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    torch.manual_seed(SEED)
    M, K = dims["M"], dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    A = torch.randn(M, K)
    # Reshape to tiled form and sum over K-tile dimension
    A_tiled = A.reshape(M // tile_m, tile_m, K // tile_k, tile_k)
    A_tiled = A_tiled.permute(0, 2, 1, 3)  # (M//tile_m, K//tile_k, tile_m, tile_k)
    reduced = A_tiled.sum(dim=1)  # (M//tile_m, tile_m, tile_k)
    # Reconstruct to (M, tile_k) — each row-group's K tiles are summed
    return reduced.reshape(M, tile_k)
