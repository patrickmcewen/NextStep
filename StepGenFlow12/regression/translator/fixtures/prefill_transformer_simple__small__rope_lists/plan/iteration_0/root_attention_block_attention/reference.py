import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V):
        seq_len = Q.shape[0]
        head_dim = Q.shape[-1]
        num_heads = Q.shape[1]
        num_kv_heads = K.shape[1]
        query_per_kvhead = num_heads // num_kv_heads

        Qh = (
            Q.view(seq_len, num_kv_heads, query_per_kvhead, head_dim)
              .permute(1, 2, 0, 3)
        )                                 # [Hkv, qpkv, S, D]
        Kh = K.permute(1, 0, 2).unsqueeze(1)   # [Hkv, 1, S, D]
        Vh = V.permute(1, 0, 2).unsqueeze(1)   # [Hkv, 1, S, D]

        scores = Qh @ Kh.transpose(-1, -2)   # [Hkv, qpkv, S, S]
        row_max = scores.amax(dim=-1, keepdim=True)
        e = torch.exp(scores - row_max)      # stability
        num = e @ Vh
        denom = e.sum(dim=-1, keepdim=True)
        attn = num / denom
        attn = attn.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)
        return attn

def get_inputs(dims):
    import torch
    from precompute import SEED
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B,
        Qwen30B, SmallerQwen30B,
    )
    torch.manual_seed(SEED + 4)

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
    V = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)
    return [Q, K, V]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
