"""PyTorch reference: simple prefill transformer layer (attention + MoE).

Single-sequence, no-KV-cache version of seed_kernels/end_to_end.  Computes
one decoder-style layer over a sequence of length S, reusing the same
Mixtral / Qwen3-30B-A3B model configs:

    RMSNorm -> QKV + QK-RMSNorm + RoPE -> GQA full-sequence attention
        -> O-proj -> ResAdd -> RMSNorm -> Routed top-k MoE -> ResAdd

Differences vs end_to_end:
  * No batching: leading dim is seq_len, not batch
  * No KV cache: K and V come entirely from the current sequence
  * No external routing/trace files: top-k routing is computed in
    precompute.py from float64 attention (avoids fp32 boundary flips)

Attention uses numerically-stable softmax (max-subtraction, no 1/sqrt(d)
scaling) in float32, matching the STeP streaming-softmax kernel. Max-sub
is mathematically equivalent to the no-max formulation since exp(-row_max)
cancels in num/denom, but it keeps exp() from overflowing in fp32.

Inputs and routing tensors come from StepDB/precompute.py via the `tensors`
arg. Routing uses `expert_onehot` to recover (token, top-k slot) pairs per
expert without re-running the float-precision-sensitive top-k.
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
    w_gate_list = tensors["w_gate_list"]
    w_up_list = tensors["w_up_list"]
    w_down_list = tensors["w_down_list"]
    expert_weights = tensors["expert_weights"]
    expert_onehot = tensors["expert_onehot"]

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

        # [5] GQA full-sequence attention with max-sub softmax in fp32.
        Qh = (
            Q.view(seq_len, mc.num_kv_heads, mc.query_per_kvhead, mc.head_dim)
             .permute(1, 2, 0, 3)
        )                                                               # [Hkv, qpkv, S, D]
        Kh = K.permute(1, 0, 2).unsqueeze(1)                            # [Hkv, 1, S, D]
        Vh = V.permute(1, 0, 2).unsqueeze(1)                            # [Hkv, 1, S, D]
        scores = Qh @ Kh.transpose(-1, -2)
        row_max = scores.amax(dim=-1, keepdim=True)
        e = torch.exp(scores - row_max)
        num = e @ Vh
        denom = e.sum(dim=-1, keepdim=True)
        attn = num / denom
        attn = attn.permute(2, 0, 1, 3).reshape(
            seq_len, mc.num_heads, mc.head_dim
        )

        # [6] O-projection + first residual add
        attn_flat = attn.reshape(seq_len, mc.num_heads * mc.head_dim)
        o_proj_out = attn_flat @ o_proj_weight
        res_add_0 = o_proj_out + input_tensor

        # [7] Post-attention RMSNorm
        normed_2 = _rms_norm(res_add_0)

        # [8] MoE: y[t] = sum_j w[t,j] * down(silu(gate(x))*up(x)) for j in top-k.
        # expert_onehot[s, k, e] == 1 iff token s assigns its slot k to expert e,
        # so torch.where on the per-expert slice gives the (token, slot) pairs
        # that step_impl routes to expert `e_idx`.
        moe_output = torch.zeros(seq_len, mc.dim, dtype=torch.float32)
        for e_idx in range(mc.n_routed_experts):
            tok, top_pos = torch.where(expert_onehot[:, :, e_idx] == 1)
            if len(tok) == 0:
                continue
            gate_out = normed_2[tok] @ w_gate_list[e_idx]
            up_out = normed_2[tok] @ w_up_list[e_idx]
            hidden = F.silu(gate_out) * up_out
            down_out = hidden @ w_down_list[e_idx]
            moe_output[tok] += down_out * expert_weights[tok, top_pos, None]

        # [9] Final residual add
        output = moe_output + res_add_0

    return output
