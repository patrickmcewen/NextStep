"""PyTorch reference: rotate_half helper (HuggingFace-style half-swap).

The rotation used in RoPE:
    rotate_half(x) = concat([-x[..., D/2:], x[..., :D/2]], dim=-1)

so that the full RoPE can be written as
    y_rope = x * cos + rotate_half(x) * sin.

Operates on x [batch, num_heads, head_dim]. compute_gold flattens the
batch and head axes so the expected output is 2-D, matching the tile
layout produced by the STeP implementation (tile rows = num_heads,
tile cols = head_dim).
"""
import torch
import torch.nn as nn

SEED = 42


def _rotate_half(x):
    """Rotary embedding helper: concat([-x2, x1], dim=-1)."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _rotate_half(x)


def get_inputs(dims):
    torch.manual_seed(SEED)
    batch = dims["batch"]
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    assert head_dim % 2 == 0, "head_dim must be even for rotate_half"

    x = torch.randn(batch, num_heads, head_dim)
    return [x]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    out = model(*inputs)
    batch = dims["batch"]
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    return out.reshape(batch * num_heads, head_dim)
