import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_core = AttentionCoreModel()
        self.o_proj = OProjModel()

    def forward(self, Q, K, V, o_proj_weight):
        attn = self.attention_core(Q, K, V)
        out = self.o_proj(attn, o_proj_weight)
        return out

def get_inputs(dims):
    torch.manual_seed(22222)   # unique per‑child seed

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
    num_heads = mc.num_heads
    num_kv_heads = mc.num_kv_heads
    head_dim = mc.head_dim

    Q = torch.randn(seq_len, num_heads, head_dim)
    K = torch.randn(seq_len, num_kv_heads, head_dim)
    V = torch.randn(seq_len, num_kv_heads, head_dim)
    o_proj_weight = torch.randn(num_heads * head_dim, hidden_dim)

    return [Q, K, V, o_proj_weight]

# expose the class name for the parent
AttentionOProjModel = Model

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)