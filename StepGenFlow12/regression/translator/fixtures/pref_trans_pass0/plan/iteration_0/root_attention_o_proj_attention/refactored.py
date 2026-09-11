import torch
import torch.nn as nn
from end_to_end.model_configs import (
    SmallerMixtral8x7B, Mixtral8x7B,
    SmallerQwen30B, Qwen30B,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.compute_qkv = ComputeQkvModel()
        self.attention_compute = AttentionComputeModel()

    def forward(self, Q, K, V):
        Qh, Kh, Vh = self.compute_qkv(Q, K, V)
        attn = self.attention_compute(Qh, Kh, Vh)
        return attn

def get_inputs(dims):
    torch.manual_seed(1111)   # unique seed for this child

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise AssertionError(f"Unknown model_name: {model_name!r}")

    Q = torch.randn(seq_len, mc.num_heads, mc.head_dim)
    K = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)
    V = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)

    return [Q, K, V]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)