import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Qh, Kh):
        # scores = Qh @ Kh^T
        scores = Qh @ Kh.transpose(-1, -2)          # [Hkv, qpkv, S, S]
        # row-wise max for numerical stability
        row_max = scores.amax(dim=-1, keepdim=True) # [Hkv, qpkv, S, 1]
        # exponentiate shifted scores
        e = torch.exp(scores - row_max)             # [Hkv, qpkv, S, S]
        return e

def get_inputs(dims):
    torch.manual_seed(302)

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        from end_to_end.model_configs import SmallerMixtral8x7B, Mixtral8x7B
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        from end_to_end.model_configs import SmallerQwen30B, Qwen30B
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise AssertionError(f"Unknown model_name: {model_name!r}")

    num_heads = mc.num_heads
    num_kv_heads = mc.num_kv_heads
    head_dim = mc.head_dim
    query_per_kvhead = num_heads // num_kv_heads

    Qh = torch.randn(num_kv_heads, query_per_kvhead, seq_len, head_dim)
    Kh = torch.randn(num_kv_heads, 1, seq_len, head_dim)

    return [Qh, Kh]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
