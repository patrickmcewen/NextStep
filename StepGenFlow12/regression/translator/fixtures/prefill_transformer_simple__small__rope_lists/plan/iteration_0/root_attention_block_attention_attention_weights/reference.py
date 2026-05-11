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

    def forward(self, Qh, Kh):
        # Compute attention scores and stable softmax weights
        scores = Qh @ Kh.transpose(-1, -2)                # [Hkv, qpkv, S, S]
        row_max = scores.amax(dim=-1, keepdim=True)      # [Hkv, qpkv, S, 1]
        e = torch.exp(scores - row_max)                  # stability
        denom = e.sum(dim=-1, keepdim=True)              # [Hkv, qpkv, S, 1]
        attn_weights = e / denom                         # [Hkv, qpkv, S, S]
        return attn_weights

def get_inputs(dims):
    import torch
    from precompute import SEED
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B,
        Qwen30B, SmallerQwen30B,
    )
    torch.manual_seed(SEED + 20)

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

    # Directly synthesize Qh and Kh in the shapes required for this child
    Qh = torch.randn(mc.num_kv_heads, query_per_kvhead, seq_len, mc.head_dim)
    Kh = torch.randn(mc.num_kv_heads, 1, seq_len, mc.head_dim)
    return [Qh, Kh]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
