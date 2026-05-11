import torch
import torch.nn as nn

def _rms_norm(x, eps=1e-6):
    """Root‑mean‑square layer‑norm used before the QKV projections."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, q_proj, k_proj, v_proj):
        # ---------------------------------------------------------
        # Infer the model configuration from the hidden dimension.
        # ---------------------------------------------------------
        hidden_dim = input_tensor.shape[1]

        from end_to_end.model_configs import (
            SmallerMixtral8x7B, Mixtral8x7B,
            SmallerQwen30B, Qwen30B,
        )
        _candidates = [
            SmallerMixtral8x7B(), Mixtral8x7B(),
            SmallerQwen30B(), Qwen30B(),
        ]
        mc = None
        for cand in _candidates:
            if cand.hidden_dim == hidden_dim:
                mc = cand
                break
        if mc is None:
            raise AssertionError(
                f"Unable to find matching model config for hidden_dim={hidden_dim}"
            )

        seq_len = input_tensor.shape[0]

        # ---------------------------------------------------------
        # 1️⃣ Pre‑attention RMSNorm
        # ---------------------------------------------------------
        normed = _rms_norm(input_tensor)   # [S, HID]

        # ---------------------------------------------------------
        # 2️⃣ Q, K, V projections + reshape
        # ---------------------------------------------------------
        Q = (normed @ q_proj).view(seq_len, mc.num_heads, mc.head_dim)
        K = (normed @ k_proj).view(seq_len, mc.num_kv_heads, mc.head_dim)
        V = (normed @ v_proj).view(seq_len, mc.num_kv_heads, mc.head_dim)

        # Return the three projection tensors.
        return Q, K, V

def get_inputs(dims):
    """
    Generates the tensors required by this child:
        input_tensor, q_proj, k_proj, v_proj
    """
    torch.manual_seed(202)   # distinct seed for this child

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

    return [input_tensor, q_proj, k_proj, v_proj]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
