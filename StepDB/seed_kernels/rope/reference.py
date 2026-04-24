"""PyTorch reference: Rotary Position Embedding (RoPE) on Q and K.

Extracted from the end_to_end transformer layer (step [4] of
seed_kernels/end_to_end/reference.py):

    Q = Q * cos + rotate_half(Q) * sin
    K = K * cos + rotate_half(K) * sin

where rotate_half(x) = concat([-x[..., D/2:], x[..., :D/2]], dim=-1).

Operates on Q [batch, num_q_heads, head_dim] and K [batch, num_kv_heads,
head_dim], with cos/sin of shape [batch, 1, head_dim] broadcasting over the
head axis.

compute_gold concatenates Q_out and K_out along the heads axis (dim=1),
then flattens to (batch*(num_q_heads + num_kv_heads), head_dim). Rows are
ordered per-batch: each batch contributes its num_q_heads Q rows followed
by its num_kv_heads K rows. This is the natural layout produced by the
STeP impl, which stacks Q and K vertically per batch and runs a single
rotate_half + cos/sin pipeline over the combined tensor.
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

    def forward(self, Q, K, cos, sin):
        Q_out = Q * cos + _rotate_half(Q) * sin
        K_out = K * cos + _rotate_half(K) * sin
        return Q_out, K_out


def get_inputs(dims):
    torch.manual_seed(SEED)
    batch = dims["batch"]
    num_q_heads = dims["num_q_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    assert head_dim % 2 == 0, "head_dim must be even for rotate_half"

    Q = torch.randn(batch, num_q_heads, head_dim)
    K = torch.randn(batch, num_kv_heads, head_dim)
    cos = torch.randn(batch, 1, head_dim)
    sin = torch.randn(batch, 1, head_dim)
    return [Q, K, cos, sin]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    Q_out, K_out = model(*inputs)
    head_dim = dims["head_dim"]
    combined = torch.cat([Q_out, K_out], dim=1)
    return combined.reshape(-1, head_dim)
