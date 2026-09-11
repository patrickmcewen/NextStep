"""PyTorch reference: rotate_half helper (HuggingFace-style half-swap).

    rotate_half(x) = concat([-x[..., D/2:], x[..., :D/2]], dim=-1)

Operates on x [batch, num_heads, head_dim]. compute_gold flattens the
batch and head axes so the expected output is 2-D, matching the tile
layout produced by the STeP implementation.

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    x = tensors["x"]
    half = x.shape[-1] // 2
    out = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return out.reshape(dims["batch"] * dims["num_heads"], dims["head_dim"])
