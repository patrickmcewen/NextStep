"""PyTorch reference: End-to-end transformer layer (attention + MoE).

Computes a full Mixtral/Qwen transformer decode step:
  RMSNorm -> QKV + QK-RMSNorm + RoPE -> GQA Attention (KV cache)
  -> O-proj -> ResAdd -> RMSNorm -> MoE (gate/up/down + SiLU) -> ResAdd

Attention uses numerically-stable softmax (max-subtraction, no 1/sqrt(d)
scaling) in float32, matching the STeP streaming-softmax kernel. Max-sub is
mathematically equivalent to the no-max formulation since exp(-row_max)
cancels in num/denom, but it keeps exp() from overflowing in fp32.

Inputs, expert routing (loaded from .npz by precompute), trace-derived
``num_token_list``, and prefilled k_cache/v_cache come from
StepDB/precompute.py via ``get_inputs(dims)``, so reference and step_impl
see byte-identical inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(x, eps=1e-6):
    """RMS normalization along last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x):
    """Rotary embedding helper: [-x2, x1] along last dim."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, q_proj, k_proj, v_proj,
                cos, sin, k_cache, v_cache,
                expert_indices, expert_weights,
                w_gate_list, w_up_list, w_down_list,
                num_token_list, o_proj_weight):
        # Derive per-call shape constants from the tensor args. Matches the
        # ``mc.*`` attrs used by precompute.py / step_impl.py: head_dim is
        # the trailing dim of cos/sin; num_heads / num_kv_heads come from
        # the projection columns; n_routed_experts is the expert-list length;
        # dim is the trailing dim of any expert's down projection.
        batch, _ = input_tensor.shape
        head_dim = cos.shape[-1]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim
        query_per_kvhead = num_heads // num_kv_heads
        n_routed_experts = len(w_gate_list)
        dim = w_down_list[0].shape[1]

        # Don't mutate caller-owned KV cache: precompute returns a single
        # buffer reused across reference + step_impl invocations.
        k_cache = k_cache.clone()
        v_cache = v_cache.clone()

        # [1] RMS Norm on input
        normed = _rms_norm(input_tensor)

        # [2] QKV projections
        Q = (normed @ q_proj).view(batch, num_heads, head_dim)
        K = (normed @ k_proj).view(batch, num_kv_heads, head_dim)
        V = (normed @ v_proj).view(batch, num_kv_heads, head_dim)

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

        # [6] GQA attention (numerically-stable softmax, no 1/sqrt(d) scaling).
        # Vectorize across kv-heads: view Q as [Hkv, qpkv, D] and permute the
        # per-batch K/V slice to [Hkv, S, D] so the matmul broadcasts the qpkv
        # query group across each kv-head.
        attn_output = torch.zeros(batch, num_heads, head_dim)
        Q_grouped = Q.view(batch, num_kv_heads, query_per_kvhead, head_dim)
        for i in range(batch):
            seq_len = num_token_list[i] + 1
            q_i = Q_grouped[i]                                # [Hkv, qpkv, D]
            k_i = k_cache[i, :seq_len].permute(1, 0, 2)       # [Hkv, S, D]
            v_i = v_cache[i, :seq_len].permute(1, 0, 2)       # [Hkv, S, D]

            scores = q_i @ k_i.transpose(-1, -2)              # [Hkv, qpkv, S]
            row_max = scores.amax(dim=-1, keepdim=True)
            exp_scores = torch.exp(scores - row_max)
            context = exp_scores @ v_i                        # [Hkv, qpkv, D]
            attn_output[i] = (context / exp_scores.sum(dim=-1, keepdim=True)) \
                .reshape(num_heads, head_dim)

        # [7] O-projection
        attn_flat = attn_output.view(batch, num_heads * head_dim)
        o_proj_out = attn_flat @ o_proj_weight

        # [8] Residual add
        res_add_0 = o_proj_out + input_tensor

        # [9] Post-attention RMS Norm
        normed_2 = _rms_norm(res_add_0)

        # [10] MoE: y[i] = sum_j w[i,j] * down_j(silu(gate_j(x)) * up_j(x))
        moe_output = torch.zeros(batch, dim)
        for e in range(n_routed_experts):
            idx, top_pos = torch.where(expert_indices == e)
            if len(idx) == 0:
                continue
            gate_out = normed_2[idx] @ w_gate_list[e]
            up_out = normed_2[idx] @ w_up_list[e]
            hidden = F.silu(gate_out) * up_out
            down_out = hidden @ w_down_list[e]
            moe_output[idx] += down_out * expert_weights[idx, top_pos, None]

        # [11] Final residual add
        return moe_output + res_add_0


def get_init_inputs(dims):
    return []


def get_inputs(dims):
    """Return the 15 positional tensor args for ``Model.forward``.

    Delegates to ``StepDB/precompute.py:_precompute_end_to_end`` so the
    reference and step_impl see byte-identical tensors (including the
    routing .npz, the trace-derived ``num_token_list``, and the prefilled
    KV caches).
    """
    from precompute import precompute_tensors  # StepDB/precompute.py
    t = precompute_tensors("end_to_end", dims)
    return [
        t["input_tensor"], t["q_proj"], t["k_proj"], t["v_proj"],
        t["cos"], t["sin"], t["k_cache"], t["v_cache"],
        t["expert_indices"], t["expert_weights"],
        t["w_gate_list"], t["w_up_list"], t["w_down_list"],
        t["num_token_list"], t["o_proj_weight"],
    ]


def compute_gold(dims):
    with torch.no_grad():
        return Model(*get_init_inputs(dims))(*get_inputs(dims))
