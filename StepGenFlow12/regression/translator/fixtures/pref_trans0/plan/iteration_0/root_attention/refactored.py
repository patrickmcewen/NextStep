import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # child model that computes the attention output
        self.attention_compute = AttentionComputeModel()

    def forward(self, Q, K, V, o_proj_weight, input_tensor):
        # compute attention (shape: [S, H, D])
        attn = self.attention_compute(Q, K, V)

        # O‑projection + first residual add
        attn_flat = attn.reshape(attn.shape[0], -1)            # (seq_len, num_heads * head_dim)
        o_proj_out = attn_flat @ o_proj_weight                # (seq_len, hidden_dim)
        res_add_0 = o_proj_out + input_tensor                 # residual connection
        return res_add_0

def get_inputs(dims):
    torch.manual_seed(101)   # distinct seed for this child

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

    Q = torch.randn(seq_len, mc.num_heads, mc.head_dim)
    K = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)
    V = torch.randn(seq_len, mc.num_kv_heads, mc.head_dim)

    o_proj_weight = torch.randn(mc.num_heads * mc.head_dim, mc.hidden_dim)
    input_tensor = torch.randn(seq_len, mc.hidden_dim)

    return [Q, K, V, o_proj_weight, input_tensor]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
