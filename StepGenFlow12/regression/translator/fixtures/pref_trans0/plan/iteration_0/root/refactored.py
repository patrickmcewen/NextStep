import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention = PreAttentionModel()
        self.attention = AttentionModel()
        self.moe = MoeModel()

    def forward(self,
                input_tensor, q_proj, k_proj, v_proj,
                cos, sin, o_proj_weight,
                w_gate, w_up, w_down,
                expert_weights, expert_onehot):
        # Stage 1: RMSNorm → QKV → per‑head RMSNorm → RoPE
        Q, K, V = self.pre_attention(
            input_tensor, q_proj, k_proj, v_proj, cos, sin
        )

        # Stage 2: GQA attention, O‑projection and first residual add
        res_add_0 = self.attention(Q, K, V, o_proj_weight, input_tensor)

        # Stage 3: Post‑attention RMSNorm, MoE, final residual add
        out = self.moe(
            res_add_0,
            w_gate, w_up, w_down,
            expert_weights, expert_onehot,
        )
        return out


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
