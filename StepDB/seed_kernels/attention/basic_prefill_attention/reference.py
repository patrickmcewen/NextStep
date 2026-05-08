"""PyTorch reference: attention-only slice of the simple prefill transformer.

Extracted from seed_kernels/transformer_layer/prefill_transformer_simple by
removing the post-attention RMSNorm + routed MoE + final residual add.
The kernel runs:

    RMSNorm -> QKV + QK-RMSNorm + RoPE -> GQA full-sequence attention
        -> O-proj -> ResAdd

GQA attention uses numerically-stable softmax (max-subtraction, no 1/sqrt(d))
in float32, matching the STeP streaming-softmax kernel which subtracts the
per-row max before exp. Max-subtraction is mathematically equivalent to the
no-max formulation since the exp(-m) factor cancels in num/denom.

Single-sequence, no-batching, no-KV-cache: leading dim is seq_len.

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
from types import SimpleNamespace

import torch


# Inlined model configs. Mirrors end_to_end.model_configs in step_tl, but
# kept here as concrete literals so the LLM reading this reference sees
# every value `mc.<attr>` resolves to without external imports.
_MODEL_CONFIGS = {
    ("mixtral", False): SimpleNamespace(
        hidden_dim=4096, head_dim=128, num_heads=32, num_kv_heads=8,
        query_per_kvhead=4,
        n_routed_experts=8, n_activated_experts=2,
        dim=4096, moe_inter_dim=14336),
    ("mixtral", True): SimpleNamespace(
        hidden_dim=512, head_dim=32, num_heads=16, num_kv_heads=4,
        query_per_kvhead=4,
        n_routed_experts=8, n_activated_experts=2,
        dim=512, moe_inter_dim=1792),
    ("qwen", False): SimpleNamespace(
        hidden_dim=2048, head_dim=64, num_heads=32, num_kv_heads=4,
        query_per_kvhead=8,
        n_routed_experts=128, n_activated_experts=8,
        dim=2048, moe_inter_dim=768),
    ("qwen", True): SimpleNamespace(
        hidden_dim=128, head_dim=16, num_heads=8, num_kv_heads=2,
        query_per_kvhead=4,
        n_routed_experts=128, n_activated_experts=8,
        dim=128, moe_inter_dim=48),
}


def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _model_config(model_name, is_small):
    key = (model_name, is_small)
    assert key in _MODEL_CONFIGS, (
        f"Unknown model_name/is_small combination: {key!r}. "
        f"Known: {list(_MODEL_CONFIGS.keys())}"
    )
    return _MODEL_CONFIGS[key]


def compute_gold(dims, tensors):
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)
    mc = _model_config(dims["model_name"], is_small)

    input_tensor = tensors["input_tensor"]
    q_proj = tensors["q_proj"]
    k_proj = tensors["k_proj"]
    v_proj = tensors["v_proj"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    o_proj_weight = tensors["o_proj_weight"]

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
        # Numerically-stable softmax (no 1/sqrt(d) scaling) in fp32, matching
        # the STeP streaming-softmax kernel which subtracts the per-row max
        # before exp. exp(-row_max) cancels in num/denom, so this is exactly
        # equivalent to the no-max formulation but stable in fp32.
        Qh = (
            Q.view(seq_len, mc.num_kv_heads, mc.query_per_kvhead, mc.head_dim)
             .permute(1, 2, 0, 3)
        )                                                               # [Hkv, qpkv, S, D]
        Kh = K.permute(1, 0, 2).unsqueeze(1)                            # [Hkv, 1, S, D]
        Vh = V.permute(1, 0, 2).unsqueeze(1)                            # [Hkv, 1, S, D]
        scores = Qh @ Kh.transpose(-1, -2)                              # [Hkv, qpkv, S, S]
        row_max = scores.amax(dim=-1, keepdim=True)
        e = torch.exp(scores - row_max)
        num = e @ Vh
        denom = e.sum(dim=-1, keepdim=True)
        attn = num / denom
        attn = attn.permute(2, 0, 1, 3).reshape(
            seq_len, mc.num_heads, mc.head_dim
        )

        # [6] O-projection + residual add
        attn_flat = attn.reshape(seq_len, mc.num_heads * mc.head_dim)
        o_proj_out = attn_flat @ o_proj_weight
        output = o_proj_out + input_tensor

    return output
