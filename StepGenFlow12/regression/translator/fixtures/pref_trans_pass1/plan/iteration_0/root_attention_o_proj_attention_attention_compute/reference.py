import torch
import torch.nn as nn
from end_to_end.model_configs import (
    SmallerMixtral8x7B, Mixtral8x7B,
    SmallerQwen30B, Qwen30B,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Qh, Kh, Vh):
        # GQA full‑sequence attention (stable softmax)
        scores = Qh @ Kh.transpose(-1, -2)
        row_max = scores.amax(dim=-1, keepdim=True)
        e = torch.exp(scores - row_max)
        num = e @ Vh
        denom = e.sum(dim=-1, keepdim=True)
        attn = num / denom

        # reshape back to (seq_len, num_heads, head_dim)
        seq_len = attn.shape[2]
        num_heads = attn.shape[0] * attn.shape[1]
        head_dim = attn.shape[3]
        attn = attn.permute(2, 0, 1, 3).reshape(seq_len, num_heads, head_dim)

        return attn

def get_inputs(dims):
    torch.manual_seed(3333)   # distinct seed for this child

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise AssertionError(f"Unknown model_name: {model_name!r}")

    # Generate raw Q, K, V and then apply the same reshaping as the first child
    Q = torch.randn(seq_len, mc.num_heads, mc.head_dim)
    K = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)
    V = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)

    num_heads = Q.shape[1]
    num_kv_heads = K.shape[1]
    query_per_kvhead = num_heads // num_kv_heads

    Qh = (
        Q.view(seq_len, num_kv_heads, query_per_kvhead, mc.head_dim)
           .permute(1, 2, 0, 3)
    )
    Kh = K.permute(1, 0, 2).unsqueeze(1)
    Vh = V.permute(1, 0, 2).unsqueeze(1)

    return [Qh, Kh, Vh]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
