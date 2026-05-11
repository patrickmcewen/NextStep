import torch
import torch.nn as nn
import torch.nn.functional as F

def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert_compute = ExpertComputeModel()   # child defined above

    def forward(self,
                normed_2,
                w_gate, w_up, w_down,
                expert_weights, expert_onehot):
        """
        Mixture‑of‑Experts aggregation.
        """
        seq_len, dim = normed_2.shape
        n_routed_experts = w_gate.shape[0]

        moe_output = torch.zeros(seq_len, dim, dtype=torch.float32)

        for e_idx in range(n_routed_experts):
            # tokens routed to this expert and their slot positions
            tok, top_pos = torch.where(expert_onehot[:, :, e_idx] == 1)
            if len(tok) == 0:
                continue

            # Compute the expert's raw output
            down_out = self.expert_compute(
                normed_2, tok,
                w_gate[e_idx], w_up[e_idx], w_down[e_idx]
            )
            # Apply per‑token routing weight and accumulate
            moe_output[tok] += down_out * expert_weights[tok, top_pos, None]

        return moe_output

def get_inputs(dims):
    torch.manual_seed(200)   # distinct seed for this child

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

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)


def get_init_inputs(dims):
    return []
