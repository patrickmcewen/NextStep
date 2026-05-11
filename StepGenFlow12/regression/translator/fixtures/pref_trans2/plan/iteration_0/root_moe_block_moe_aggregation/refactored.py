import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_contribute = ExpertContributeModel()

    def forward(
        self,
        normed_2,          # [S, HID] – RMS‑normed token representations
        w_gate,            # [n_routed_experts, HID, moe_inter_dim]
        w_up,              # [n_routed_experts, HID, moe_inter_dim]
        w_down,            # [n_routed_experts, moe_inter_dim, HID]
        expert_weights,    # [S, n_activated_experts] (float)
        expert_onehot,     # [S, n_activated_experts, n_routed_experts] (int64)
    ):
        seq_len, dim = normed_2.shape
        n_routed_experts, _, _ = w_gate.shape

        # Output buffer
        moe_output = torch.zeros(seq_len, dim, dtype=torch.float32)

        # MoE aggregation per routed expert
        for e_idx in range(n_routed_experts):
            # Tokens assigned to this expert
            tok, top_pos = torch.where(expert_onehot[:, :, e_idx] == 1)
            if len(tok) == 0:
                continue

            # Routing scores for the selected tokens
            routing_weights_cur = expert_weights[tok, top_pos]   # [#tok]

            # Compute this expert's contribution
            contrib = self.expert_contribute(
                normed_2[tok],
                w_gate[e_idx],
                w_up[e_idx],
                w_down[e_idx],
                routing_weights_cur,
            )

            # Accumulate into the final output
            moe_output[tok] += contrib

        return moe_output

# original helper functions (unchanged)

def get_inputs(dims):
    # unique seed for this child
    torch.manual_seed(22222)

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
    hidden_dim = mc.hidden_dim
    moe_inter_dim = mc.moe_inter_dim
    n_routed_experts = mc.n_routed_experts
    n_activated_experts = mc.n_activated_experts

    # ----- generate the raw token tensor and RMS‑norm it -----
    res_add_0 = torch.randn(seq_len, hidden_dim)
    eps = 1e-6
    normed_2 = res_add_0 * torch.rsqrt(res_add_0.pow(2).mean(dim=-1, keepdim=True) + eps)

    # ----- stacked expert weights -----
    w_gate = torch.randn(n_routed_experts, hidden_dim, moe_inter_dim)
    w_up   = torch.randn(n_routed_experts, hidden_dim, moe_inter_dim)
    w_down = torch.randn(n_routed_experts, moe_inter_dim, hidden_dim)

    # ----- routing tensors (shape‑compatible) -----
    expert_weights = torch.randn(seq_len, n_activated_experts)
    expert_onehot = torch.randint(
        0, 2,
        (seq_len, n_activated_experts, n_routed_experts),
        dtype=torch.int64,
    )

    return [
        normed_2,
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