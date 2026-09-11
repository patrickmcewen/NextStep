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

Inputs (synthesized RNG tensors, projection weights, expert weight stacks)
and routing tensors come from StepDB/precompute.py via ``get_inputs(dims)``.
Routing uses ``expert_onehot`` to recover (token, top-k slot) pairs per
expert without re-running the float-precision-sensitive top-k.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, q_proj, k_proj, v_proj,
                cos, sin, o_proj_weight,
                w_gate, w_up, w_down,
                expert_weights, expert_onehot):
        # Derive per-call shape constants from the tensor args. Matches the
        # ``mc.*`` attrs used by precompute.py / step_impl.py: head_dim is
        # the trailing dim of cos/sin; num_heads / num_kv_heads come from
        # the projection columns; n_routed_experts and dim come from w_gate.
        seq_len, _ = input_tensor.shape
        head_dim = cos.shape[-1]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim
        query_per_kvhead = num_heads // num_kv_heads
        n_routed_experts, dim, _ = w_gate.shape

        # [1] Pre-attention RMSNorm
        normed = _rms_norm(input_tensor)                                # [S, HID]

        # [2] QKV projections
        Q = (normed @ q_proj).view(seq_len, num_heads, head_dim)
        K = (normed @ k_proj).view(seq_len, num_kv_heads, head_dim)
        V = (normed @ v_proj).view(seq_len, num_kv_heads, head_dim)

        # [3] Per-head Q/K RMSNorm
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # [4] RoPE: cos/sin are [S, 1, D] and broadcast over the heads dim
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        # [5] GQA full-sequence attention with max-sub softmax in fp32.
        Qh = (
            Q.view(seq_len, num_kv_heads, query_per_kvhead, head_dim)
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
            seq_len, num_heads, head_dim
        )

        # [6] O-projection + first residual add
        attn_flat = attn.reshape(seq_len, num_heads * head_dim)
        o_proj_out = attn_flat @ o_proj_weight
        res_add_0 = o_proj_out + input_tensor

        # [7] Post-attention RMSNorm
        normed_2 = _rms_norm(res_add_0)

        # [8] MoE: y[t] = sum_j w[t,j] * down(silu(gate(x))*up(x)) for j in top-k.
        # expert_onehot[s, k, e] == 1 iff token s assigns its slot k to expert e,
        # so torch.where on the per-expert slice gives the (token, slot) pairs
        # that step_impl routes to expert `e_idx`. ``w_gate[e_idx]`` is the
        # contiguous [dim, moe_inter_dim] view of the e_idx-th expert weight.
        moe_output = torch.zeros(seq_len, dim, dtype=torch.float32)
        for e_idx in range(n_routed_experts):
            tok, top_pos = torch.where(expert_onehot[:, :, e_idx] == 1)
            if len(tok) == 0:
                continue
            gate_out = normed_2[tok] @ w_gate[e_idx]
            up_out = normed_2[tok] @ w_up[e_idx]
            hidden = F.silu(gate_out) * up_out
            down_out = hidden @ w_down[e_idx]
            moe_output[tok] += down_out * expert_weights[tok, top_pos, None]

        # [9] Final residual add
        return moe_output + res_add_0


def get_init_inputs(dims):
    return []


def get_inputs(dims):
    """Return the 12 positional tensor args for ``Model.forward``.

    Delegates to ``StepDB/precompute.py:_precompute_prefill_transformer_simple``
    so the reference and step_impl see byte-identical tensors. The precompute
    contains the routing computation (run in fp32 with max-subtraction softmax)
    that produces the deterministic ``expert_weights`` / ``expert_onehot``.
    """
    from precompute import precompute_tensors  # StepDB/precompute.py
    t = precompute_tensors("prefill_transformer_simple", dims)
    return [
        t["input_tensor"], t["q_proj"], t["k_proj"], t["v_proj"],
        t["cos"], t["sin"], t["o_proj_weight"],
        t["w_gate"], t["w_up"], t["w_down"],
        t["expert_weights"], t["expert_onehot"],
    ]


def compute_gold(dims):
    with torch.no_grad():
        return Model(*get_init_inputs(dims))(*get_inputs(dims))
