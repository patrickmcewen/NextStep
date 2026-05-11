import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.compute_e = ComputeEModel()
        self.compute_attn = ComputeAttnModel()

    def forward(self, Qh, Kh, Vh):
        e = self.compute_e(Qh, Kh)
        attn = self.compute_attn(e, Vh)
        return attn

def get_inputs(dims):
    # distinct seed for this child
    torch.manual_seed(301)

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
    Vh = torch.randn(num_kv_heads, 1, seq_len, head_dim)

    return [Qh, Kh, Vh]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)