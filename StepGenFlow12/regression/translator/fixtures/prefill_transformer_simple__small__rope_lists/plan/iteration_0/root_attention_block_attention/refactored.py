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
        self.prepare_qkv = PrepareQkvModel()
        self.attention_weights = AttentionWeightsModel()
        self.apply_weights_and_reshape = ApplyWeightsAndReshapeModel()

    def forward(self, Q, K, V):
        Qh, Kh, Vh = self.prepare_qkv(Q, K, V)
        attn_weights = self.attention_weights(Qh, Kh)
        attn = self.apply_weights_and_reshape(attn_weights, Vh)
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

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)


def get_init_inputs(dims):
    return []
