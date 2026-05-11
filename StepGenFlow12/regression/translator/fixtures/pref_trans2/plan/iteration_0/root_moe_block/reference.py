import torch
import torch.nn as nn
import torch.nn.functional as F

def _rms_norm(x, eps: float = 1e-6):
    """RMSNorm used inside the MoE block."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        res_add_0,          # [S, HID] – result after O‑proj + first residual add
        w_gate,             # [n_routed_experts, HID, moe_inter_dim]
        w_up,               # [n_routed_experts, HID, moe_inter_dim]
        w_down,             # [n_routed_experts, moe_inter_dim, HID]
        expert_weights,     # [S, n_activated_experts] (float)
        expert_onehot,      # [S, n_activated_experts, n_routed_experts] (int64)
    ):
        seq_len, dim = res_add_0.shape
        n_routed_experts, _, _ = w_gate.shape

        # Post‑attention RMSNorm
        normed_2 = _rms_norm(res_add_0)

        # MoE aggregation
        moe_output = torch.zeros(seq_len, dim, dtype=torch.float32)
        for e_idx in range(n_routed_experts):
            tok, top_pos = torch.where(expert_onehot[:, :, e_idx] == 1)
            if len(tok) == 0:
                continue
            gate_out = normed_2[tok] @ w_gate[e_idx]          # [#tok, moe_inter_dim]
            up_out = normed_2[tok] @ w_up[e_idx]              # [#tok, moe_inter_dim]
            hidden = F.silu(gate_out) * up_out                # [#tok, moe_inter_dim]
            down_out = hidden @ w_down[e_idx]                 # [#tok, HID]
            moe_output[tok] += down_out * expert_weights[tok, top_pos, None]
        return moe_output


def get_inputs(dims):
    """Synthetic inputs for the MoE sub‑module."""
    torch.manual_seed(67890)   # unique per‑child seed

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
    hidden_dim = mc.hidden_dim          # same as `dim` used throughout the model
    moe_inter_dim = mc.moe_inter_dim
    n_routed_experts = mc.n_routed_experts
    n_activated_experts = mc.n_activated_experts

    # Residual after the first add (same shape as model hidden state)
    res_add_0 = torch.randn(seq_len, hidden_dim)

    # Stacked expert weights
    w_gate = torch.randn(n_routed_experts, hidden_dim, moe_inter_dim)
    w_up   = torch.randn(n_routed_experts, hidden_dim, moe_inter_dim)
    w_down = torch.randn(n_routed_experts, moe_inter_dim, hidden_dim)

    # Routing tensors – random but shape‑compatible
    expert_weights = torch.randn(seq_len, n_activated_experts)
    expert_onehot = torch.randint(
        0, 2,
        (seq_len, n_activated_experts, n_routed_experts),
        dtype=torch.int64,
    )

    return [
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
    ]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
