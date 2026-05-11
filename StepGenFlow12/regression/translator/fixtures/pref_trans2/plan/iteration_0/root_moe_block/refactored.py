import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms_norm = RmsNormModel()
        self.moe_aggregation = MoeAggregationModel()

    def forward(
        self,
        res_add_0,          # [S, HID] – result after O‑proj + first residual add
        w_gate,             # [n_routed_experts, HID, moe_inter_dim]
        w_up,               # [n_routed_experts, HID, moe_inter_dim]
        w_down,             # [n_routed_experts, moe_inter_dim, HID]
        expert_weights,     # [S, n_activated_experts] (float)
        expert_onehot,      # [S, n_activated_experts, n_routed_experts] (int64)
    ):
        # 1️⃣ RMSNorm of the residual
        normed_2 = self.rms_norm(res_add_0)

        # 2️⃣ MoE aggregation using the normalized representation
        moe_output = self.moe_aggregation(
            normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot
        )
        return moe_output


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
