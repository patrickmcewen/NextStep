import torch
import torch.nn as nn

def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, cos, sin):
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin
        return Q, K

def get_inputs(dims):
    import torch
    from precompute import SEED
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B,
        Qwen30B, SmallerQwen30B,
    )
    torch.manual_seed(SEED + 3)

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    Q = torch.randn(seq_len, mc.num_heads, mc.head_dim)
    K = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)
    cos = torch.randn(seq_len, 1, mc.head_dim)
    sin = torch.randn(seq_len, 1, mc.head_dim)
    return [Q, K, cos, sin]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
