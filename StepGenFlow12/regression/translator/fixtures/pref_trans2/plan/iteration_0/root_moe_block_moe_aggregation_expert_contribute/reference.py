import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        normed_selected,   # [#tok, HID]
        w_gate_e,          # [HID, moe_inter_dim]
        w_up_e,            # [HID, moe_inter_dim]
        w_down_e,          # [moe_inter_dim, HID]
        routing_weights,   # [#tok] (float32)
    ):
        # Linear projections for the selected tokens
        gate_out = normed_selected @ w_gate_e          # [#tok, moe_inter_dim]
        up_out   = normed_selected @ w_up_e            # [#tok, moe_inter_dim]

        # SILU activation on gate output, element‑wise multiply with up output
        hidden = F.silu(gate_out) * up_out             # [#tok, moe_inter_dim]

        # Down projection
        down_out = hidden @ w_down_e                    # [#tok, HID]

        # Weight by routing scores
        weighted = down_out * routing_weights.unsqueeze(-1)  # [#tok, HID]

        return weighted

def get_inputs(dims):
    # unique seed for this child
    torch.manual_seed(33333)

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

    # Simulated subset of token representations (use full seq_len for simplicity)
    normed_selected = torch.randn(seq_len, hidden_dim)

    # Expert‑specific weight matrices
    w_gate_e = torch.randn(hidden_dim, moe_inter_dim)
    w_up_e   = torch.randn(hidden_dim, moe_inter_dim)
    w_down_e = torch.randn(moe_inter_dim, hidden_dim)

    # Routing scores for the selected tokens
    routing_weights = torch.randn(seq_len)

    return [
        normed_selected,
        w_gate_e,
        w_up_e,
        w_down_e,
        routing_weights,
    ]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
