"""PyTorch reference: Routed MoE with top-k expert selection.

Full MoE layer with routing: each token selects top-k experts,
computes expert(x) = down(silu(gate(x)) * up(x)) for each selected expert,
and sums the weighted expert outputs.

  output[i] = sum_j( weight[i,j] * expert_j(x[i]) )  for j in top-k experts

Tensors (raw RNG'd weights/inputs and the float32 routing decisions) come
from StepDB/precompute.py via the `tensors` arg.

Based on step_tl/src/utils/moe.py::moe_gold_calc and
step_tl/end_to_end/moe/static_no_timemultiplex.py.
"""
import torch
import torch.nn.functional as F


def compute_gold(dims, tensors):
    B = dims["B"]
    D = dims["D"]
    n_experts = dims["n_experts"]

    gate_weights = tensors["gate_weights"]
    up_weights = tensors["up_weights"]
    down_weights = tensors["down_weights"]
    x = tensors["x"]
    expert_indices = tensors["expert_indices"]
    expert_weights = tensors["expert_weights"]

    y = torch.zeros(B, D)
    with torch.no_grad():
        for i in range(n_experts):
            idx, top_pos = torch.where(expert_indices == i)
            if len(idx) == 0:
                continue
            gate_out = x[idx] @ gate_weights[i]
            up_out = x[idx] @ up_weights[i]
            projected = F.silu(gate_out) * up_out
            down_out = projected @ down_weights[i]
            y[idx] += down_out * expert_weights[idx, top_pos, None]

    return y
