"""PyTorch reference: attention-only slice of the simple prefill transformer.

Extracted from seed_kernels/transformer_layer/prefill_transformer_simple by
removing the post-attention RMSNorm + routed MoE + final residual add.
The kernel runs:

    RMSNorm -> QKV + QK-RMSNorm + RoPE -> GQA full-sequence attention
        -> O-proj -> ResAdd

GQA attention uses exp-normalize softmax (no max subtraction, no 1/sqrt(d))
in float64, mirroring end_to_end / prefill_transformer_simple and the STeP
streaming-softmax kernel.

Single-sequence, no-batching, no-KV-cache: leading dim is seq_len.
"""
import sys
from pathlib import Path

import torch

import step_py as _sp
_STEP_TL_ROOT = str(Path(_sp.__file__).resolve().parent.parent.parent)
if _STEP_TL_ROOT not in sys.path:
    sys.path.insert(0, _STEP_TL_ROOT)

from end_to_end.model_configs import (
    Mixtral8x7B, SmallerMixtral8x7B, Qwen30B, SmallerQwen30B,
)

SEED = 42


def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _model_config(model_name, is_small):
    if model_name == "mixtral":
        return SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    if model_name == "qwen":
        return SmallerQwen30B() if is_small else Qwen30B()
    assert False, f"Unknown model_name: {model_name!r}"


def get_inputs(dims):
    """All randomly-generated tensors, in canonical RNG order.

    The order locked here is also the order used by precompute.py, so the
    bit-exactness check in stepdb-add-benchmark step 4 compares element-
    for-element against precompute_tensors.
    """
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)
    mc = _model_config(dims["model_name"], is_small)

    torch.manual_seed(SEED)

    input_tensor = torch.randn(seq_len, mc.hidden_dim)
    q_proj = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(seq_len, 1, mc.head_dim)
    sin = torch.randn(seq_len, 1, mc.head_dim)
    o_proj_weight = torch.randn(mc.num_heads * mc.head_dim, mc.hidden_dim)
    return [input_tensor, q_proj, k_proj, v_proj, cos, sin, o_proj_weight]


def compute_gold(dims):
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)
    mc = _model_config(dims["model_name"], is_small)

    (input_tensor, q_proj, k_proj, v_proj, cos, sin, o_proj_weight) = get_inputs(dims)

    with torch.no_grad():
        # [1] Pre-attention RMSNorm
        normed = _rms_norm(input_tensor)                                # [S, HID]

        # [2] QKV projections
        Q = (normed @ q_proj).view(seq_len, mc.num_heads, mc.head_dim)
        K = (normed @ k_proj).view(seq_len, mc.num_kv_heads, mc.head_dim)
        V = (normed @ v_proj).view(seq_len, mc.num_kv_heads, mc.head_dim)

        # [3] Per-head Q/K RMSNorm
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # [4] RoPE: cos/sin are [S, 1, D] and broadcast over the heads dim
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        # [5] GQA full-sequence attention.  Group queries by kv head:
        #   Q [S, num_heads, D] -> [Hkv, qpkv, S, D]
        #   K, V [S, num_kv_heads, D] -> [Hkv, 1, S, D]
        # exp-normalize softmax in float64 mirrors end_to_end (no max sub,
        # no 1/sqrt(d); float64 keeps exp from overflowing for any seq_len).
        Qh = (
            Q.view(seq_len, mc.num_kv_heads, mc.query_per_kvhead, mc.head_dim)
             .permute(1, 2, 0, 3)
             .double()
        )                                                               # [Hkv, qpkv, S, D]
        Kh = K.permute(1, 0, 2).unsqueeze(1).double()                   # [Hkv, 1, S, D]
        Vh = V.permute(1, 0, 2).unsqueeze(1).double()                   # [Hkv, 1, S, D]
        scores = Qh @ Kh.transpose(-1, -2)                              # [Hkv, qpkv, S, S]
        e = torch.exp(scores)
        num = e @ Vh                                                    # [Hkv, qpkv, S, D]
        denom = e.sum(dim=-1, keepdim=True)
        attn = (num / denom).float()
        attn = attn.permute(2, 0, 1, 3).reshape(
            seq_len, mc.num_heads, mc.head_dim
        )                                                               # [S, num_heads, D]

        # [6] O-projection + residual add
        attn_flat = attn.reshape(seq_len, mc.num_heads * mc.head_dim)
        o_proj_out = attn_flat @ o_proj_weight                          # [S, HID]
        output = o_proj_out + input_tensor                              # [S, HID]

    return output
