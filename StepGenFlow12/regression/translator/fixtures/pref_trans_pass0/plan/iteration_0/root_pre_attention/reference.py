import sys
from pathlib import Path
import torch
import torch.nn as nn

# Re‑import model configs in this child
import step_py as _sp
_STEP_TL_ROOT = str(Path(_sp.__file__).resolve().parent.parent.parent)
if _STEP_TL_ROOT not in sys.path:
    sys.path.insert(0, _STEP_TL_ROOT)

from end_to_end.model_configs import (
    SmallerMixtral8x7B, Mixtral8x7B,
    SmallerQwen30B, Qwen30B,
)

def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_tensor,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
    ):
        # shape constants
        seq_len = input_tensor.shape[0]
        head_dim = cos.shape[-1]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim

        # [1] Pre‑attention RMSNorm
        normed = _rms_norm(input_tensor)

        # [2] QKV projections
        Q = (normed @ q_proj).view(seq_len, num_heads, head_dim)
        K = (normed @ k_proj).view(seq_len, num_kv_heads, head_dim)
        V = (normed @ v_proj).view(seq_len, num_kv_heads, head_dim)

        # [3] Per‑head Q/K RMSNorm
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # [4] RoPE
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

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
