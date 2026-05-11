import torch
import torch.nn as nn

def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.moe_compute = MoeComputeModel()

    def forward(self,
                normed_2,
                w_gate, w_up, w_down,
                expert_weights, expert_onehot):
        # Delegate the heavy MoE computation to the child model.
        return self.moe_compute(
            normed_2,
            w_gate, w_up, w_down,
            expert_weights, expert_onehot)

def get_inputs(dims):
    torch.manual_seed(102)   # distinct seed for this child

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        from end_to_end.model_configs import SmallerMixtral8x7B, Mixtral8x7B
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        from end_to_end.model_configs import SmallerQwen30B, Qwen30B
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise AssertionError(f"Unknown model_name: {model_name!r}")

    # generate a dummy residual tensor and compute its RMSNorm so that
    # this child can be run in isolation
    res_add_0 = torch.randn(seq_len, mc.dim)
    normed_2 = _rms_norm(res_add_0)

    # stacked expert weight tensors
    w_gate = torch.randn(mc.n_routed_experts, mc.dim, mc.moe_inter_dim)
    w_up   = torch.randn(mc.n_routed_experts, mc.dim, mc.moe_inter_dim)
    w_down = torch.randn(mc.n_routed_experts, mc.moe_inter_dim, mc.dim)

    # routing tensors (contents can be arbitrary)
    expert_weights = torch.randn(seq_len, mc.n_activated_experts).softmax(dim=-1)
    expert_onehot = torch.zeros(
        seq_len, mc.n_activated_experts, mc.n_routed_experts, dtype=torch.int64
    )

    return [normed_2, w_gate, w_up, w_down, expert_weights, expert_onehot]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)