import torch
import torch.nn as nn

def _rms_norm(x, eps: float = 1e-6):
    """RMSNorm used by the parent (pre‑attention)."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_block = AttentionBlockModel()
        self.moe_block = MoeBlockModel()

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
        # ---------- [1] Pre‑attention RMSNorm ----------
        normed = _rms_norm(input_tensor)        # [S, HID]

        # ---------- [2] Attention sub‑module ----------
        o_proj_out = self.attention_block(
            normed,
            q_proj,
            k_proj,
            v_proj,
            cos,
            sin,
            o_proj_weight,
        )

        # ---------- [3] First residual add ----------
        res_add_0 = o_proj_out + input_tensor

        # ---------- [4] MoE sub‑module ----------
        moe_output = self.moe_block(
            res_add_0,
            w_gate,
            w_up,
            w_down,
            expert_weights,
            expert_onehot,
        )

        # ---------- [5] Final residual add ----------
        return moe_output + res_add_0


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
