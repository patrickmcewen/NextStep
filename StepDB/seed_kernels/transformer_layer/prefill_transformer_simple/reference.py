"""PyTorch reference: simple prefill transformer layer (attention + MoE).

Single-sequence, no-KV-cache version of seed_kernels/end_to_end.  Computes
one decoder-style layer over a sequence of length S, reusing the same
Mixtral / Qwen3-30B-A3B model configs:

    RMSNorm -> QKV + QK-RMSNorm + RoPE -> GQA full-sequence attention
        -> O-proj -> ResAdd -> RMSNorm -> Routed top-k MoE -> ResAdd

Differences vs end_to_end:
  * No batching: leading dim is seq_len, not batch
  * No KV cache: K and V come entirely from the current sequence
  * No external routing/trace files: top-k routing is computed in-place
    from the post-attention activations (router_w @ normed)

Attention uses exp-normalize softmax (no max subtraction, no 1/sqrt(d)
scaling) in float64, mirroring end_to_end and the STeP streaming-softmax
kernel.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
    w_gate_list = [
        torch.nn.Linear(mc.dim, mc.moe_inter_dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]
    w_up_list = [
        torch.nn.Linear(mc.dim, mc.moe_inter_dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]
    w_down_list = [
        torch.nn.Linear(mc.moe_inter_dim, mc.dim, bias=False)
        .weight.T.detach().clone().contiguous()
        for _ in range(mc.n_routed_experts)
    ]
    router_w = torch.randn(mc.dim, mc.n_routed_experts)
    return [
        input_tensor, q_proj, k_proj, v_proj, cos, sin, o_proj_weight,
        w_gate_list, w_up_list, w_down_list, router_w,
    ]


def compute_gold(dims):
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)
    mc = _model_config(dims["model_name"], is_small)

    (input_tensor, q_proj, k_proj, v_proj, cos, sin, o_proj_weight,
     w_gate_list, w_up_list, w_down_list, router_w) = get_inputs(dims)

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
        #   K, V [S, num_kv_heads, D] -> [Hkv, S, D]
        # exp-normalize softmax in float64 mirrors end_to_end (no max sub,
        # no 1/sqrt(d); float64 keeps exp from overflowing for any seq_len).
        Qh = (
            Q.view(seq_len, mc.num_kv_heads, mc.query_per_kvhead, mc.head_dim)
             .permute(1, 2, 0, 3)
             .double()
        )                                                               # [Hkv, qpkv, S, D]
        # Insert qpkv broadcast axis on K/V so matmul aligns batch dims
        # [Hkv, qpkv] vs [Hkv, 1] regardless of how Hkv compares to qpkv.
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

        # [6] O-projection + first residual add
        attn_flat = attn.reshape(seq_len, mc.num_heads * mc.head_dim)
        o_proj_out = attn_flat @ o_proj_weight                          # [S, HID]
        res_add_0 = o_proj_out + input_tensor                           # [S, HID]

        # [7] Post-attention RMSNorm
        normed_2 = _rms_norm(res_add_0)                                 # [S, HID]

        # [8] Top-k expert routing (computed in-place; no external file)
        router_logits = normed_2 @ router_w                             # [S, n_experts]
        _, expert_indices = torch.topk(
            router_logits, mc.n_activated_experts, dim=-1
        )
        expert_weights_raw, _ = torch.topk(
            router_logits, mc.n_activated_experts, dim=-1
        )
        expert_weights = torch.softmax(expert_weights_raw, dim=-1)      # [S, n_active]

        # [9] MoE: y[t] = sum_j w[t,j] * down(silu(gate(x))*up(x)) for j in top-k
        moe_output = torch.zeros(seq_len, mc.dim)
        for e_idx in range(mc.n_routed_experts):
            tok, top_pos = torch.where(expert_indices == e_idx)
            if len(tok) == 0:
                continue
            gate_out = normed_2[tok] @ w_gate_list[e_idx]
            up_out = normed_2[tok] @ w_up_list[e_idx]
            hidden = F.silu(gate_out) * up_out
            down_out = hidden @ w_down_list[e_idx]
            moe_output[tok] += down_out * expert_weights[tok, top_pos, None]

        # [10] Final residual add
        output = moe_output + res_add_0

    return output
