import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_compute = AttentionComputeModel()

    def forward(self, Q, K, V):
        # infer dimensions
        seq_len = Q.shape[0]
        num_heads = Q.shape[1]
        head_dim = Q.shape[2]
        num_kv_heads = K.shape[1]
        query_per_kvhead = num_heads // num_kv_heads

        # reshape for GQA attention
        Qh = (
            Q.view(seq_len, num_kv_heads, query_per_kvhead, head_dim)
            .permute(1, 2, 0, 3)
        )  # [Hkv, qpkv, S, D]
        Kh = K.permute(1, 0, 2).unsqueeze(1)  # [Hkv, 1, S, D]
        Vh = V.permute(1, 0, 2).unsqueeze(1)  # [Hkv, 1, S, D]

        # delegate core attention computation
        attn = self.attention_compute(Qh, Kh, Vh)

        # restore original layout
        attn = attn.permute(2, 0, 1, 3).reshape(
            seq_len, num_heads, head_dim
        )  # [S, num_heads, D]

        return attn

def get_inputs(dims):
    torch.manual_seed(12345)   # unique per‑child seed

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

    Q = torch.randn(seq_len, num_heads, head_dim)
    K = torch.randn(seq_len, num_kv_heads, head_dim)
    V = torch.randn(seq_len, num_kv_heads, head_dim)

    return [Q, K, V]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)