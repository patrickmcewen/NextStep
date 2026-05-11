import torch
import torch.nn as nn

# Helper functions needed by this child
def _rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, q_proj, k_proj, v_proj, cos, sin, o_proj_weight):
        """
        Implements the pre‑attention RMSNorm, QKV projection, per‑head RMSNorm,
        RoPE, full‑sequence GQA attention with max‑sub softmax, and the final
        O‑projection + residual addition. Returns the tensor that will be used
        as the skip‑connection input for the MoE block (i.e. `res_add_0`).
        """
        seq_len, _ = input_tensor.shape
        head_dim = cos.shape[-1]
        num_heads = q_proj.shape[1] // head_dim
        num_kv_heads = k_proj.shape[1] // head_dim
        query_per_kvhead = num_heads // num_kv_heads

        # 1️⃣ Pre‑attention RMSNorm
        normed = _rms_norm(input_tensor)

        # 2️⃣ QKV projections
        Q = (normed @ q_proj).view(seq_len, num_heads, head_dim)
        K = (normed @ k_proj).view(seq_len, num_kv_heads, head_dim)
        V = (normed @ v_proj).view(seq_len, num_kv_heads, head_dim)

        # 3️⃣ Per‑head RMSNorm
        Q = _rms_norm(Q)
        K = _rms_norm(K)

        # 4️⃣ RoPE
        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        # 5️⃣ GQA full‑sequence attention (max‑sub softmax, fp32)
        Qh = (
            Q.view(seq_len, num_kv_heads, query_per_kvhead, head_dim)
                .permute(1, 2, 0, 3)
        )                                 # [Hkv, qpkv, S, D]
        Kh = K.permute(1, 0, 2).unsqueeze(1)   # [Hkv, 1, S, D]
        Vh = V.permute(1, 0, 2).unsqueeze(1)   # [Hkv, 1, S, D]

        scores = Qh @ Kh.transpose(-1, -2)     # [Hkv, qpkv, S, S]
        row_max = scores.amax(dim=-1, keepdim=True)
        e = torch.exp(scores - row_max)        # stability
        num = e @ Vh
        denom = e.sum(dim=-1, keepdim=True)
        attn = num / denom
        attn = attn.permute(2, 0, 1, 3).reshape(
            seq_len, num_heads, head_dim
        )                                      # [S, H, D]

        # 6️⃣ O‑projection + first residual add
        attn_flat = attn.reshape(seq_len, num_heads * head_dim)
        o_proj_out = attn_flat @ o_proj_weight
        res_add_0 = o_proj_out + input_tensor

        return res_add_0


def get_inputs(dims):
    """
    Re‑creates the subset of tensors needed for the attention block.
    The RNG order and seed are exactly the same as the full precompute,
    so these tensors match the first seven entries returned by the original
    `get_inputs`.
    """
    import torch
    from precompute import SEED
    from end_to_end.model_configs import (
        Mixtral8x7B, SmallerMixtral8x7B,
        Qwen30B, SmallerQwen30B,
    )

    torch.manual_seed(SEED)

    model_name = dims["model_name"]
    seq_len = dims["seq_len"]
    is_small = dims.get("is_small", False)

    if model_name == "mixtral":
        mc = SmallerMixtral8x7B() if is_small else Mixtral8x7B()
    elif model_name == "qwen":
        mc = SmallerQwen30B() if is_small else Qwen30B()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    input_tensor   = torch.randn(seq_len, mc.hidden_dim)
    q_proj         = torch.randn(mc.hidden_dim, mc.num_heads * mc.head_dim)
    k_proj         = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    v_proj         = torch.randn(mc.hidden_dim, mc.num_kv_heads * mc.head_dim)
    cos            = torch.randn(seq_len, 1, mc.head_dim)
    sin            = torch.randn(seq_len, 1, mc.head_dim)
    o_proj_weight  = torch.randn(mc.num_heads * mc.head_dim, mc.hidden_dim)

    return [
        input_tensor,
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
