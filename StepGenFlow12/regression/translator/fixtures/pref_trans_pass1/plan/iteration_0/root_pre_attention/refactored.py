import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attn_norm_and_proj = PreAttnNormAndProjModel()
        self.per_head_norm_and_rope = PerHeadNormAndRopeModel()

    def forward(
        self,
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
    ):
        # Child 1: pre‑attention RMSNorm + QKV projection
        Q, K, V = self.pre_attn_norm_and_proj(
            input_tensor, q_proj, k_proj, v_proj, cos
        )
        # Child 2: per‑head RMSNorm and RoPE
        Q, K, V = self.per_head_norm_and_rope(Q, K, V, cos, sin)
        return Q, K, V

def get_inputs(dims):
    torch.manual_seed(1111)   # unique seed for this child

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise AssertionError(f"Unknown model_name: {model_name!r}")

    input_tensor = torch.randn(seq_len, mc.hidden_dim)
    q_proj = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(seq_len, 1, mc.head_dim)
    sin = torch.randn(seq_len, 1, mc.head_dim)

    return [
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
    ]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)