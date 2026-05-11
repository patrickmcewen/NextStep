import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention_norm = PreAttentionNormModel()
        self.qkv_projection = QkvProjectionModel()
        self.per_head_norm = PerHeadNormModel()
        self.apply_rope = ApplyRopeModel()
        self.attention = AttentionModel()
        self.o_proj_residual = OProjResidualModel()

    def forward(self, input_tensor, q_proj, k_proj, v_proj, cos, sin, o_proj_weight):
        # 1️⃣ Pre‑attention RMSNorm
        normed = self.pre_attention_norm(input_tensor)

        # 2️⃣ QKV projections (head_dim inferred via cos)
        Q, K, V = self.qkv_projection(normed, q_proj, k_proj, v_proj, cos)

        # 3️⃣ Per‑head RMSNorm
        Q, K = self.per_head_norm(Q, K)

        # 4️⃣ RoPE
        Q, K = self.apply_rope(Q, K, cos, sin)

        # 5️⃣ GQA full‑sequence attention (max‑sub softmax)
        attn = self.attention(Q, K, V)

        # 6️⃣ O‑projection + residual addition
        res_add_0 = self.o_proj_residual(attn, o_proj_weight, input_tensor)

        return res_add_0


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
