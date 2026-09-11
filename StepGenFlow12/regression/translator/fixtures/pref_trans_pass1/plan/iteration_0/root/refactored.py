import torch
import torch.nn as nn

# Original get_inputs is reused unchanged.
def get_inputs(dims):
    from precompute import precompute_tensors  # StepDB/precompute.py
    t = precompute_tensors("prefill_transformer_simple", dims)
    return [
        t["input_tensor"], t["q_proj"], t["k_proj"], t["v_proj"],
        t["cos"], t["sin"], t["o_proj_weight"],
        t["w_gate"], t["w_up"], t["w_down"],
        t["expert_weights"], t["expert_onehot"],
    ]

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attention = PreAttentionModel()
        self.attention_o_proj = AttentionOProjModel()
        self.moe = MoeModel()

    def forward(
        self,
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        o_proj_weight,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
    ):
        # 1️⃣ Pre‑attention pipeline
        Q, K, V = self.pre_attention(
            input_tensor,
            q_proj,
            k_proj,
            v_proj,
            cos,
            sin,
        )

        # 2️⃣ Attention + O‑projection + first residual add
        res_add_0 = self.attention_o_proj(
            Q,
            K,
            V,
            o_proj_weight,
            input_tensor,
        )

        # 3️⃣ Post‑attention RMSNorm + MoE + final residual add
        out = self.moe(
            res_add_0,
            w_gate,
            w_up,
            w_down,
            expert_weights,
            expert_onehot,
        )
        return out


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
