import torch
import torch.nn as nn
import torch.nn.functional as F

def _rms_norm(x, eps: float = 1e-6):
    """RMSNorm used inside the attention block."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def _rotate_half(x):
    """Helper for rotary‑position‑embedding."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        normed,          # [S, HID]  – output of pre‑attention RMSNorm
        q_proj,          # [HID, num_heads * head_dim]
        k_proj,          # [HID, num_kv_heads * head_dim]
        v_proj,          # [HID, num_kv_heads * head_dim]
        cos,             # [S, 1, head_dim]
        sin,             # [S, 1, head_dim]
        o_proj_weight,   # [num_heads * head_dim, HID]
    ):
        seq_len, _ = normed.shape
        head_dim = cos.shape[-1]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim
        query_per_kvhead = num_heads // num_kv_heads

        # -------- QKV projections ----------
        Q = (normed @ q_proj).view(seq_len, num_heads, head_dim)
        K = (normed @ k_proj).view(seq_len, num_kv_heads, head_dim)
        V = (normed @ v_proj).view(seq_len, num_kv_heads, head_dim)

        # -------- per‑head RMSNorm ----------
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # -------- RoPE ----------
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        # -------- GQA attention (max‑sub softmax) ----------
        Qh = (
            Q.view(seq_len, num_kv_heads, query_per_kvhead, head_dim)
            .permute(1, 2, 0, 3)
        )  # [Hkv, qpkv, S, D]
        Kh = K.permute(1, 0, 2).unsqueeze(1)  # [Hkv, 1, S, D]
        Vh = V.permute(1, 0, 2).unsqueeze(1)  # [Hkv, 1, S, D]

        scores = Qh @ Kh.transpose(-1, -2)            # [Hkv, qpkv, S, S]
        row_max = scores.amax(dim=-1, keepdim=True)  # for numerical stability
        e = torch.exp(scores - row_max)
        num = e @ Vh
        denom = e.sum(dim=-1, keepdim=True)
        attn = num / denom
        attn = attn.permute(2, 0, 1, 3).reshape(
            seq_len, num_heads, head_dim
        )  # [S, num_heads, head_dim]

        # -------- O‑projection ----------
        attn_flat = attn.reshape(seq_len, num_heads * head_dim)
        o_proj_out = attn_flat @ o_proj_weight      # [S, HID]
        return o_proj_out


def get_inputs(dims):
    """Synthetic inputs for the attention sub‑module."""
    torch.manual_seed(12345)   # unique per‑child seed

    model_name = dims["model_name"]
    is_small = dims.get("is_small", False)
    from end_to_end.model_configs import (
        Mixtral8x7B,
        SmallerMixtral8x7B,
        Qwen30B,
        SmallerQwen30B,
    )

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    seq_len = dims["seq_len"]
    hidden_dim = mc.hidden_dim

    # The RMSNorm output (`normed`) is the same shape as the model's hidden state.
    normed = torch.randn(seq_len, hidden_dim)

    q_proj = torch.randn(hidden_dim, mc.num_heads * mc.head_dim)
    k_proj = torch.randn(hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj = torch.randn(hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos = torch.randn(seq_len, 1, mc.head_dim)
    sin = torch.randn(seq_len, 1, mc.head_dim)
    o_proj_weight = torch.randn(mc.num_heads * mc.head_dim, hidden_dim)

    return [
        normed,
        q_proj,
        k_proj,
        v_proj,
        cos,
        sin,
        o_proj_weight,
    ]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    inputs = get_inputs(dims)
    return Model()(*inputs)
