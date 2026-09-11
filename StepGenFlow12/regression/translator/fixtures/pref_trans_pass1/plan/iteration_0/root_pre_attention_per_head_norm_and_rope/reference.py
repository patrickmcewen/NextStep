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

    def forward(self, Q, K, V, cos, sin):
        # [3] Per‑head Q/K RMSNorm
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # [4] RoPE
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        # V is unchanged
        return Q, K, V

def get_inputs(dims):
    torch.manual_seed(54321)   # unique seed for this child

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise AssertionError(f"Unknown model_name: {model_name!r}")

    # Shapes derived from the model config
    head_dim = mc.head_dim
    Q = torch.randn(seq_len, mc.num_heads, head_dim)
    K = torch.randn(seq_len, mc.num_kv_heads, head_dim)
    V = torch.randn(seq_len, mc.num_kv_heads, head_dim)
    cos = torch.randn(seq_len, 1, head_dim)
    sin = torch.randn(seq_len, 1, head_dim)

    return [Q, K, V, cos, sin]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
