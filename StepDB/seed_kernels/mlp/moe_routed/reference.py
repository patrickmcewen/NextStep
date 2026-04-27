"""PyTorch reference: Routed MoE with top-k expert selection.

Full MoE layer with routing: each token selects top-k experts,
computes expert(x) = down(silu(gate(x)) * up(x)) for each selected expert,
and sums the weighted expert outputs.

  output[i] = sum_j( weight[i,j] * expert_j(x[i]) )  for j in top-k experts

Based on step_tl/src/utils/moe.py::moe_gold_calc and
step_tl/end_to_end/moe/static_no_timemultiplex.py.
"""
import torch
import torch.nn.functional as F

SEED = 42


def compute_gold(dims):
    B = dims["B"]
    D = dims["D"]
    F_dim = dims["F"]
    n_experts = dims["n_experts"]
    n_active = dims["n_active"]

    torch.manual_seed(SEED)

    # Must match step_impl RNG order exactly:
    # gate_weights[0..n], up_weights[0..n], down_weights[0..n], x, router_w
    gate_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    up_weights = [torch.randn(D, F_dim) for _ in range(n_experts)]
    down_weights = [torch.randn(F_dim, D) for _ in range(n_experts)]
    x = torch.randn(B, D)
    router_w = torch.randn(D, n_experts)

    # Router: top-k selection
    router_logits = x @ router_w  # [B, n_experts]
    _, expert_indices = torch.topk(router_logits, n_active, dim=-1)
    expert_weights_raw, _ = torch.topk(router_logits, n_active, dim=-1)
    expert_weights = torch.softmax(expert_weights_raw, dim=-1)  # [B, n_active]

    # Compute MoE output (matches moe_gold_calc pattern)
    y = torch.zeros(B, D)
    with torch.no_grad():
        for i in range(n_experts):
            idx, top_pos = torch.where(expert_indices == i)
            if len(idx) == 0:
                continue
            # gate(x) @ gate_weights[i] -> [tokens, F]
            gate_out = x[idx] @ gate_weights[i]
            up_out = x[idx] @ up_weights[i]
            projected = F.silu(gate_out) * up_out
            down_out = projected @ down_weights[i]
            y[idx] += down_out * expert_weights[idx, top_pos, None]

    return y
