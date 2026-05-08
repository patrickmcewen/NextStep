"""PyTorch reference: End-to-end transformer layer (attention + MoE).

Computes a full Mixtral/Qwen transformer decode step:
  RMSNorm -> QKV + QK-RMSNorm + RoPE -> GQA Attention (KV cache)
  -> O-proj -> ResAdd -> RMSNorm -> MoE (gate/up/down + SiLU) -> ResAdd

Attention uses numerically-stable softmax (max-subtraction, no 1/sqrt(d)
scaling) in float32, matching the STeP streaming-softmax kernel. Max-sub is
mathematically equivalent to the no-max formulation since exp(-row_max)
cancels in num/denom, but it keeps exp() from overflowing in fp32.

Inputs, expert routing (loaded from .npz by precompute), trace-derived
`num_token_list`, and prefilled k_cache/v_cache come from
StepDB/precompute.py via the `tensors` arg, so reference and step_impl see
byte-identical inputs.
"""
from types import SimpleNamespace

import torch
import torch.nn.functional as F


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
    """RMS normalization along last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x):
    """Rotary embedding helper: [-x2, x1] along last dim."""
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
    model_name = dims["model_name"]
    batch = dims.get("batch", 64)
    is_small = dims.get("is_small", False)
    mc = _model_config(model_name, is_small)

    input_tensor = tensors["input_tensor"]
    q_proj = tensors["q_proj"]
    k_proj = tensors["k_proj"]
    v_proj = tensors["v_proj"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    expert_indices = tensors["expert_indices"]
    expert_weights = tensors["expert_weights"]
    w_gate_list = tensors["w_gate_list"]
    w_up_list = tensors["w_up_list"]
    w_down_list = tensors["w_down_list"]
    num_token_list = tensors["num_token_list"]
    o_proj_weight = tensors["o_proj_weight"]

    with torch.no_grad():
        # [1] RMS Norm on input
        normed = _rms_norm(input_tensor)

        # [2] QKV projections
        Q = (normed @ q_proj).view(batch, mc.num_heads, mc.head_dim)
        K = (normed @ k_proj).view(batch, mc.num_kv_heads, mc.head_dim)
        V = (normed @ v_proj).view(batch, mc.num_kv_heads, mc.head_dim)

        # [3] RMS Norm on Q and K (per-head normalization)
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # [4] Rotary position embeddings; cos/sin are [B, 1, head_dim]
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        # [5] Append new K, V to KV cache at the per-batch sequence end
        for i in range(batch):
            k_cache[i, num_token_list[i]] = K[i]
            v_cache[i, num_token_list[i]] = V[i]

        # [6] GQA attention (numerically-stable softmax, no 1/sqrt(d) scaling)
        attn_output = torch.zeros(batch, mc.num_heads, mc.head_dim)
        for i in range(batch):
            seq_len = num_token_list[i] + 1
            for h_kv in range(mc.num_kv_heads):
                q_lo = h_kv * mc.query_per_kvhead
                q_hi = q_lo + mc.query_per_kvhead
                q_group = Q[i, q_lo:q_hi, :]
                k_seq = k_cache[i, :seq_len, h_kv, :]
                v_seq = v_cache[i, :seq_len, h_kv, :]

                scores = q_group @ k_seq.T
                row_max = scores.amax(dim=-1, keepdim=True)
                exp_scores = torch.exp(scores - row_max)
                context = exp_scores @ v_seq
                attn_output[i, q_lo:q_hi, :] = context / exp_scores.sum(dim=-1, keepdim=True)

        # [7] O-projection
        attn_flat = attn_output.view(batch, mc.num_heads * mc.head_dim)
        o_proj_out = attn_flat @ o_proj_weight

        # [8] Residual add
        res_add_0 = o_proj_out + input_tensor

        # [9] Post-attention RMS Norm
        normed_2 = _rms_norm(res_add_0)

        # [10] MoE: y[i] = sum_j w[i,j] * down_j(silu(gate_j(x)) * up_j(x))
        moe_output = torch.zeros(batch, mc.dim)
        for e in range(mc.n_routed_experts):
            idx, top_pos = torch.where(expert_indices == e)
            if len(idx) == 0:
                continue
            gate_out = normed_2[idx] @ w_gate_list[e]
            up_out = normed_2[idx] @ w_up_list[e]
            hidden = F.silu(gate_out) * up_out
            down_out = hidden @ w_down_list[e]
            moe_output[idx] += down_out * expert_weights[idx, top_pos, None]

        # [11] Final residual add
        output = moe_output + res_add_0

    return output
