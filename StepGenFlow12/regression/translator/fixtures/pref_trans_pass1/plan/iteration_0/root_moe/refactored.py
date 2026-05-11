import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import step_py as _sp
_STEP_TL_ROOT = str(Path(_sp.__file__).resolve().parent.parent.parent)
if _STEP_TL_ROOT not in sys.path:
    sys.path.insert(0, _STEP_TL_ROOT)

from end_to_end.model_configs import (
    SmallerMixtral8x7B, Mixtral8x7B,
    SmallerQwen30B, Qwen30B,
)

def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_dispatch = MoeDispatchModel()

    def forward(
        self,
        res_add_0,
        w_gate,
        w_up,
        w_down,
        expert_weights,
        expert_onehot,
    ):
        # MoE dispatch & aggregation (child) + final residual add
        moe_output = self.moe_dispatch(
            res_add_0,
            w_gate,
            w_up,
            w_down,
            expert_weights,
            expert_onehot,
        )
        # [9] Final residual add
        return moe_output + res_add_0

def get_inputs(dims):
    torch.manual_seed(3333)   # unique seed for this child

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise AssertionError(f"Unknown model_name: {model_name!r}")

    # residual from attention + first add
    res_add_0 = torch.randn(seq_len, mc.hidden_dim)

    # Expert weight tensors
    w_gate = torch.randn(mc.n_routed_experts, mc.dim, mc.moe_inter_dim)
    w_up   = torch.randn(mc.n_routed_experts, mc.dim, mc.moe_inter_dim)
    w_down = torch.randn(mc.n_routed_experts, mc.moe_inter_dim, mc.dim)

    # Routing tensors (shapes match those emitted by precompute)
    expert_weights = torch.randn(seq_len, mc.n_activated_experts)
    expert_onehot = torch.randint(
        0, 2,
        (seq_len, mc.n_activated_experts, mc.n_routed_experts),
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