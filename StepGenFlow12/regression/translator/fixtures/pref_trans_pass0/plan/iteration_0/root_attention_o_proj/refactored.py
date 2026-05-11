import torch
import torch.nn as nn
from end_to_end.model_configs import (
    SmallerMixtral8x7B, Mixtral8x7B,
    SmallerQwen30B, Qwen30B,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AttentionModel()

    def forward(
        self,
        Q,
        K,
        V,
        o_proj_weight,
        input_tensor,
    ):
        # Compute attention output via child
        attn = self.attention(Q, K, V)   # [seq_len, num_heads, head_dim]

        # O‑projection + first residual add
        seq_len = Q.shape[0]
        num_heads = Q.shape[1]
        head_dim = Q.shape[2]

        attn_flat = attn.reshape(seq_len, num_heads * head_dim)
        o_proj_out = attn_flat @ o_proj_weight
        res_add_0 = o_proj_out + input_tensor

        return res_add_0


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
