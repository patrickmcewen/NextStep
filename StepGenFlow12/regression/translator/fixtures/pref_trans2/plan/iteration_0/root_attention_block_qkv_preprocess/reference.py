import torch
import torch.nn as nn

def _rms_norm(x, eps: float = 1e-6):
    """RMSNorm used inside the attention block."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def _rotate_half(x):
    """Helper for rotary‑position‑embedding."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, normed, q_proj, k_proj, v_proj, cos, sin):
        # infer shapes
        seq_len = normed.shape[0]
        head_dim = cos.shape[-1]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim

        # QKV projections + reshape
        Q = (normed @ q_proj).view(seq_len, num_heads, head_dim)
        K = (normed @ k_proj).view(seq_len, num_kv_heads, head_dim)
        V = (normed @ v_proj).view(seq_len, num_kv_heads, head_dim)

        # per‑head RMSNorm for Q and K
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # RoPE
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        return Q, K, V

def get_inputs(dims):
    torch.manual_seed(11111)   # unique per‑child seed

    model_name = dims["model_name"]
    is_small = dims.get("is_small", False)
    from end_to_end.model_configs import (
        Mixtral8x7B,
        SmallerMixtral8x7B,
        Qwen30B,
        SmallerQwen30B,
    )

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    seq_len = dims["seq_len"]
    hidden_dim = mc.hidden_dim
    head_dim = mc.head_dim
    num_heads = mc.num_heads
    num_kv_heads = mc.num_kv_heads

    normed = torch.randn(seq_len, hidden_dim)
    q_proj = torch.randn(hidden_dim, num_heads * head_dim)
    k_proj = torch.randn(hidden_dim, num_kv_heads * head_dim)
    v_proj = torch.randn(hidden_dim, num_kv_heads * head_dim)
    cos = torch.randn(seq_len, 1, head_dim)
    sin = torch.randn(seq_len, 1, head_dim)

    return [normed, q_proj, k_proj, v_proj, cos, sin]

# expose the class name for the parent
QkvPreprocessModel = Model


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
