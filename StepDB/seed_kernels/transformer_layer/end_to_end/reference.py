"""PyTorch reference: End-to-end transformer layer (attention + MoE).

Computes a full Mixtral/Qwen transformer decode step:
  RMSNorm -> QKV + QK-RMSNorm + RoPE -> GQA Attention (KV cache)
  -> O-proj -> ResAdd -> RMSNorm -> MoE (gate/up/down + SiLU) -> ResAdd

Attention uses exp-normalize softmax (no max subtraction, no 1/sqrt(d) scaling),
matching the STeP streaming softmax implementation.

Uses the same RNG seeds and external files (expert routing, trace data) as
step_impl.py to produce identical input tensors.
"""
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

import step_py as _sp
_STEP_TL_ROOT = str(Path(_sp.__file__).resolve().parent.parent.parent)
if _STEP_TL_ROOT not in sys.path:
    sys.path.insert(0, _STEP_TL_ROOT)

from end_to_end.model_configs import (
    Mixtral8x7B, SmallerMixtral8x7B, Qwen30B, SmallerQwen30B,
)

_EXPERT_ROUTING = {
    ("mixtral", 64): (8, 10),
    ("mixtral", 1024): (19, 9),
    ("qwen", 64): (32, 12),
    ("qwen", 1024): (22, 16),
}


def _rms_norm(x, eps=1e-6):
    """RMS normalization along last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x):
    """Rotary embedding helper: [-x2, x1] along last dim."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def compute_gold(dims):
    model_name = dims["model_name"]
    batch = dims.get("batch", 64)
    scale_seq = dims.get("scale_seq", 1)
    is_small = dims.get("is_small", False)
    stdev = dims["stdev"]
    start = dims["start"]
    end = dims["end"]

    torch.manual_seed(5)
    random.seed(42)

    # ---- Model config ----
    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        assert False, f"Unknown model_name: {model_name}"

    # ---- Input tensors (same RNG order as step_impl) ----
    input_tensor = torch.randn(batch, mc.hidden_dim)
    q_proj = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(batch, 1, mc.head_dim)
    sin = torch.randn(batch, 1, mc.head_dim)

    # ---- KV cache ----
    maxN = 4096 * scale_seq
    k_cache = torch.zeros(batch, maxN, mc.num_kv_heads, mc.head_dim)
    v_cache = torch.zeros(batch, maxN, mc.num_kv_heads, mc.head_dim)

    # ---- Expert routing (from file) ----
    routing_key = (model_name, batch)
    assert routing_key in _EXPERT_ROUTING, f"No expert routing for {routing_key}"
    iter_idx, layer_idx = _EXPERT_ROUTING[routing_key]
    routing_path = (
        Path(_STEP_TL_ROOT)
        / f"dyn_tiling/expert_routing/{model_name}_b{batch}"
        / f"iter_{iter_idx:03d}_layer_{layer_idx:03d}.npz"
    )
    assert routing_path.exists(), f"Expert routing file not found: {routing_path}"
    expert_indices = torch.from_numpy(np.load(str(routing_path))["data"])

    # ---- Expert weights and matrices (continuing RNG sequence) ----
    expert_weights = torch.softmax(
        torch.randn(batch, mc.n_activated_experts), dim=-1
    )
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

    # ---- Trace data (KV cache sequence lengths per batch element) ----
    assert batch == end - start + 1, f"batch={batch} != end-start+1={end - start + 1}"
    trace_path = (
        Path(_STEP_TL_ROOT)
        / f"dynamic_par/azure_trace/b{batch}"
        / f"conv_stdev{stdev:04d}_{start:04d}_{end:04d}.npy"
    )
    assert trace_path.exists(), f"Trace file not found: {trace_path}"
    num_token_list = np.load(str(trace_path)).astype(np.int64).tolist()
    num_token_list = [x * scale_seq for x in num_token_list]

    # Fill KV cache with random data up to each element's sequence length
    for i in range(batch):
        k_cache[i, :num_token_list[i]] = torch.randn(
            num_token_list[i], mc.num_kv_heads, mc.head_dim
        )
        v_cache[i, :num_token_list[i]] = torch.randn(
            num_token_list[i], mc.num_kv_heads, mc.head_dim
        )

    # O-projection weight (created during graph build in reshape_o_proj)
    o_proj_weight = torch.randn(mc.num_heads * mc.head_dim, mc.hidden_dim)

    # ================== Forward pass ==================
    with torch.no_grad():
        # [1] RMS Norm on input
        normed = _rms_norm(input_tensor)  # [B, HID]

        # [2] QKV projections
        Q = normed @ q_proj  # [B, num_heads * head_dim]
        K = normed @ k_proj  # [B, num_kv_heads * head_dim]
        V = normed @ v_proj  # [B, num_kv_heads * head_dim]

        Q = Q.view(batch, mc.num_heads, mc.head_dim)
        K = K.view(batch, mc.num_kv_heads, mc.head_dim)
        V = V.view(batch, mc.num_kv_heads, mc.head_dim)

        # [3] RMS Norm on Q and K (per-head normalization)
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # [4] Rotary position embeddings
        # cos/sin are [B, 1, head_dim], broadcast over heads
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        # [5] Update KV cache with new K, V
        for i in range(batch):
            k_cache[i, num_token_list[i]] = K[i]
            v_cache[i, num_token_list[i]] = V[i]

        # [6] GQA attention (exp-normalize softmax, no 1/sqrt(d) scaling)
        # Use float64 to match the STeP streaming softmax which computes raw
        # exp(scores) without max subtraction.  Float32 overflows for large
        # scores; float64 handles exp(±700) without overflow.
        attn_output = torch.zeros(batch, mc.num_heads, mc.head_dim, dtype=torch.float64)
        for i in range(batch):
            seq_len = num_token_list[i] + 1
            for h_kv in range(mc.num_kv_heads):
                q_lo = h_kv * mc.query_per_kvhead
                q_hi = q_lo + mc.query_per_kvhead
                q_group = Q[i, q_lo:q_hi, :].double()     # [qpkv, D]
                k_seq = k_cache[i, :seq_len, h_kv, :].double()  # [S, D]
                v_seq = v_cache[i, :seq_len, h_kv, :].double()  # [S, D]

                scores = q_group @ k_seq.T                 # [qpkv, S]
                exp_scores = torch.exp(scores)
                context = exp_scores @ v_seq               # [qpkv, D]
                attn_output[i, q_lo:q_hi, :] = context / exp_scores.sum(dim=-1, keepdim=True)
        attn_output = attn_output.float()

        # [7] O-projection: [B, num_heads * head_dim] @ [num_heads * head_dim, HID]
        attn_flat = attn_output.view(batch, mc.num_heads * mc.head_dim)
        o_proj_out = attn_flat @ o_proj_weight             # [B, HID]

        # [8] Residual add (attention output + original input)
        res_add_0 = o_proj_out + input_tensor              # [B, HID]

        # [9] Post-attention RMS Norm
        normed_2 = _rms_norm(res_add_0)                    # [B, HID]

        # [10] MoE: route tokens to top-k experts
        # expert_i(x) = down_i(silu(gate_i(x)) * up_i(x))
        # output[i] = sum_j(weight[i,j] * expert_j(x[i])) for j in top-k
        moe_output = torch.zeros(batch, mc.dim)
        for e in range(mc.n_routed_experts):
            idx, top_pos = torch.where(expert_indices == e)
            if len(idx) == 0:
                continue
            gate_out = normed_2[idx] @ w_gate_list[e]      # [tokens, F]
            up_out = normed_2[idx] @ w_up_list[e]          # [tokens, F]
            hidden = F.silu(gate_out) * up_out
            down_out = hidden @ w_down_list[e]             # [tokens, D]
            moe_output[idx] += down_out * expert_weights[idx, top_pos, None]

        # [11] Final residual add (MoE output + post-attention residual)
        output = moe_output + res_add_0

    return output
