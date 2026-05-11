import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_and_norm = ProjAndNormModel()
        self.rope = RopeModel()

    def forward(self, input_tensor, q_proj, k_proj, v_proj, cos, sin):
        # Stage 1: RMSNorm + QKV projection + per‑head RMSNorm
        Q, K, V = self.proj_and_norm(input_tensor, q_proj, k_proj, v_proj)

        # Stage 2: RoPE on Q and K
        Q, K = self.rope(Q, K, cos, sin)

        return Q, K, V

def get_inputs(dims):
    torch.manual_seed(100)   # distinct seed for this child

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

    input_tensor = torch.randn(seq_len, mc.hidden_dim)
    q_proj = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(seq_len, 1, mc.head_dim)
    sin = torch.randn(seq_len, 1, mc.head_dim)

    return [input_tensor, q_proj, k_proj, v_proj, cos, sin]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)