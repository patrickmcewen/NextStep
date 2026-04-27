"""PyTorch reference: QKV linear projection.

Computes Q, K, V projections from input hidden state:
  Q = input @ W_q    [B, D] @ [D, num_heads * head_dim] -> [B, num_heads * head_dim]
  K = input @ W_k    [B, D] @ [D, num_kv_heads * head_dim] -> [B, num_kv_heads * head_dim]
  V = input @ W_v    [B, D] @ [D, num_kv_heads * head_dim] -> [B, num_kv_heads * head_dim]

For simplicity this produces the concatenated QKV as a single matmul:
  output = input @ W_qkv    [B, D] @ [D, proj_dim] -> [B, proj_dim]

Extracted from step_tl/end_to_end/attention/qkv_gen.py::projection.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, W):
        return torch.matmul(x, W)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B, D, proj_dim = dims["B"], dims["D"], dims["proj_dim"]
    x = torch.randn(B, D)
    W = torch.randn(D, proj_dim)
    return x, W


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    x, W = get_inputs(dims)
    return torch.matmul(x, W)
