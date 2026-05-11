import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_preprocess = QkvPreprocessModel()
        self.attention_o_proj = AttentionOProjModel()

    def forward(self, normed, q_proj, k_proj, v_proj, cos, sin, o_proj_weight):
        Q, K, V = self.qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin)
        o_proj_out = self.attention_o_proj(Q, K, V, o_proj_weight)
        return o_proj_out


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
