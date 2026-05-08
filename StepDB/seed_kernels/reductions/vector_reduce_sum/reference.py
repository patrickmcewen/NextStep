"""PyTorch reference: Vector reduction (sum along K dimension, tiled).

Reshapes A to (M//tile_m, K//tile_k, tile_m, tile_k), sums over the K-tile
axis, and reconstructs to (M, tile_k) - each row-group's K tiles are summed.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""


def compute_gold(dims, tensors):
    M = dims["M"]
    K = dims["K"]
    tile_m = dims.get("tile_m", 16)
    tile_k = dims.get("tile_k", 16)

    A = tensors["A"]
    A_tiled = A.reshape(M // tile_m, tile_m, K // tile_k, tile_k)
    A_tiled = A_tiled.permute(0, 2, 1, 3)
    reduced = A_tiled.sum(dim=1)
    return reduced.reshape(M, tile_k)
