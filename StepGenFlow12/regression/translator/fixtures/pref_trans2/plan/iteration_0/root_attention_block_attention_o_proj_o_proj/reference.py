import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn, o_proj_weight):
        # attn: [S, num_heads, head_dim]
        seq_len, num_heads, head_dim = attn.shape
        attn_flat = attn.reshape(seq_len, num_heads * head_dim)
        o_proj_out = attn_flat @ o_proj_weight  # [S, hidden_dim]
        return o_proj_out

def get_inputs(dims):
    torch.manual_seed(54321)   # unique per‑child seed

    model_name = dims["model_name"]
    is_small = dims.get("is_small", False)
    from end_to_end.model_configs import (
        Mixtral8x7B,
        SmallerMixtral8x7B,
        Qwen30B,
        SmallerQwen30B,
    )

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    seq_len = dims["seq_len"]
    num_heads = mc.num_heads
    head_dim = mc.head_dim
    hidden_dim = mc.hidden_dim

    attn = torch.randn(seq_len, num_heads, head_dim)
    o_proj_weight = torch.randn(num_heads * head_dim, hidden_dim)

    return [attn, o_proj_weight]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
