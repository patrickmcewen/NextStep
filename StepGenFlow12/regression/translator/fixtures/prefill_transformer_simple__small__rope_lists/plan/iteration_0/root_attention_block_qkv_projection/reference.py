import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, normed, q_proj, k_proj, v_proj, cos):
        head_dim = cos.shape[-1]
        seq_len = normed.shape[0]
        Q = (normed @ q_proj).view(seq_len, q_proj.shape[1] // head_dim, head_dim)
        K = (normed @ k_proj).view(seq_len, k_proj.shape[1] // head_dim, head_dim)
        V = (normed @ v_proj).view(seq_len, v_proj.shape[1] // head_dim, head_dim)
        return Q, K, V

def get_inputs(dims):
    import torch
    from precompute import SEED
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B,
        Qwen30B, SmallerQwen30B,
    )
    torch.manual_seed(SEED + 1)

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    normed = torch.randn(seq_len, mc.hidden_dim)
    q_proj = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(seq_len, 1, mc.head_dim)
    return [normed, q_proj, k_proj, v_proj, cos]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
