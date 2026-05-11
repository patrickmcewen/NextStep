import torch
import torch.nn as nn
from precompute import SEED
from end_to_end.model_configs import (
    Mixtral8x7B, SmallerMixtral8x7B,
    Qwen30B, SmallerQwen30B,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, Vh):
        # Weighted sum over values
        num = attn_weights @ Vh                         # [Hkv, qpkv, S, D]

        # Reshape to [seq_len, num_heads, head_dim]
        seq_len = num.shape[2]
        num_heads = num.shape[0] * num.shape[1]
        head_dim = num.shape[3]
        attn = num.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)
        return attn

def get_inputs(dims):
    import torch
    from precompute import SEED
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B,
        Qwen30B, SmallerQwen30B,
    )
    torch.manual_seed(SEED + 30)

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    query_per_kvhead = mc.num_heads // mc.num_kv_heads

    attn_weights = torch.randn(
        mc.num_kv_heads, query_per_kvhead, seq_len, seq_len
    )
    Vh = torch.randn(
        mc.num_kv_heads, 1, seq_len, mc.head_dim
    )
    return [attn_weights, Vh]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
