import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, attn, o_proj_weight, input_tensor):
        seq_len, num_heads, head_dim = attn.shape
        attn_flat = attn.reshape(seq_len, num_heads * head_dim)
        o_proj_out = attn_flat @ o_proj_weight
        res_add_0 = o_proj_out + input_tensor
        return res_add_0

def get_inputs(dims):
    import torch
    from precompute import SEED
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B,
        Qwen30B, SmallerQwen30B,
    )
    torch.manual_seed(SEED + 5)

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    attn = torch.randn(seq_len, mc.num_heads, mc.head_dim)
    o_proj_weight = torch.randn(mc.num_heads * mc.head_dim, mc.hidden_dim)
    input_tensor = torch.randn(seq_len, mc.hidden_dim)
    return [attn, o_proj_weight, input_tensor]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
