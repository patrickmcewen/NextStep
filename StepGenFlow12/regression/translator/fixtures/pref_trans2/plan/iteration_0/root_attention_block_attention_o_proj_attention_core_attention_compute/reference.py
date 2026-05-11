import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Qh, Kh, Vh):
        # attention (max‑sub softmax)
        scores = Qh @ Kh.transpose(-1, -2)               # [Hkv, qpkv, S, S]
        row_max = scores.amax(dim=-1, keepdim=True)      # [Hkv, qpkv, S, 1]
        e = torch.exp(scores - row_max)                  # [Hkv, qpkv, S, S]
        num = e @ Vh                                      # [Hkv, qpkv, S, D]
        denom = e.sum(dim=-1, keepdim=True)              # [Hkv, qpkv, S, 1]
        attn = num / denom                                # [Hkv, qpkv, S, D]
        return attn

def get_inputs(dims):
    # unique seed for this child
    torch.manual_seed(54321)

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
    num_kv_heads = mc.num_kv_heads
    head_dim = mc.head_dim
    query_per_kvhead = num_heads // num_kv_heads

    # generate Q, K, V and reshape to Qh, Kh, Vh (same pattern as parent)
    Q = torch.randn(seq_len, num_heads, head_dim)
    K = torch.randn(seq_len, num_kv_heads, head_dim)
    V = torch.randn(seq_len, num_kv_heads, head_dim)

    Qh = (
        Q.view(seq_len, num_kv_heads, query_per_kvhead, head_dim)
        .permute(1, 2, 0, 3)
    )  # [Hkv, qpkv, S, D]
    Kh = K.permute(1, 0, 2).unsqueeze(1)  # [Hkv, 1, S, D]
    Vh = V.permute(1, 0, 2).unsqueeze(1)  # [Hkv, 1, S, D]

    return [Qh, Kh, Vh]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
