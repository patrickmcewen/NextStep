import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attn_rms_norm = PostAttnRmsNormModel()
        self.moe__root_moe = MoeRootMoeModel()

    def forward(self,
                res_add_0, w_gate, w_up, w_down,
                expert_weights, expert_onehot):
        # 1) post‑attention RMSNorm
        normed_2 = self.post_attn_rms_norm(res_add_0)          # [S, DIM]

        # 2) mixture‑of‑experts processing
        moe_output = self.moe__root_moe(normed_2,
                              w_gate, w_up, w_down,
                              expert_weights, expert_onehot)   # [S, DIM]

        # 3) final residual add (same as original)
        return moe_output + res_add_0

def get_inputs(dims):
    torch.manual_seed(102)   # same seed as original reference

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

    # stacked expert weight tensors
    w_gate = torch.randn(mc.n_routed_experts, mc.dim, mc.moe_inter_dim)
    w_up   = torch.randn(mc.n_routed_experts, mc.dim, mc.moe_inter_dim)
    w_down = torch.randn(mc.n_routed_experts, mc.moe_inter_dim, mc.dim)

    # routing tensors (shapes only; contents can be arbitrary)
    expert_weights = torch.randn(seq_len, mc.n_activated_experts).softmax(dim=-1)
    expert_onehot = torch.zeros(
        seq_len, mc.n_activated_experts, mc.n_routed_experts, dtype=torch.int64
    )

    # residual tensor coming from the attention block
    res_add_0 = torch.randn(seq_len, mc.dim)

    return [res_add_0, w_gate, w_up, w_down, expert_weights, expert_onehot]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)