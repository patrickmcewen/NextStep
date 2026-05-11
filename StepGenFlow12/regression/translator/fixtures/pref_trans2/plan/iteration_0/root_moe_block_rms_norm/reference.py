import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, res_add_0):
        eps = 1e-6
        # RMSNorm: x * rsqrt(mean(x^2) + eps)
        return res_add_0 * torch.rsqrt(res_add_0.pow(2).mean(dim=-1, keepdim=True) + eps)

def get_inputs(dims):
    # unique seed for this child
    torch.manual_seed(11111)

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
    hidden_dim = mc.hidden_dim

    # Residual after the first add (same shape as model hidden state)
    res_add_0 = torch.randn(seq_len, hidden_dim)

    return [res_add_0]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
