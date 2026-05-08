"""PyTorch reference: QKV generation for a transformer attention block.

Pre-attention block of a (MHA) transformer layer:
    Q = (x @ Wq).view(B, N_HEAD, HEAD_DIM)
    K = (x @ Wk).view(B, N_HEAD, HEAD_DIM)
    V = (x @ Wv).view(B, N_HEAD, HEAD_DIM)
    Q = rms_norm(Q); K = rms_norm(K)
    Q = Q * cos + rotate_half(Q) * sin
    K = K * cos + rotate_half(K) * sin

Mirrors the step_tl/end_to_end/attention/qkv_gen.py pipeline (projection
-> view -> per-head RMSNorm -> RoPE). V skips both RMSNorm and RoPE.

compute_gold reshapes [Q, K, V] to match the step_impl output layout
[3*B*N_HEAD, HEAD_DIM]: first all Q rows, then K, then V. Within each
section the rows are batch-outer, head-inner.

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch

EPS = 1e-6


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _rms_norm(x, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def compute_gold(dims, tensors):
    B = dims["B"]
    N_HEAD = dims["N_HEAD"]
    HEAD_DIM = dims["HEAD_DIM"]

    x = tensors["x"]
    q_proj = tensors["q_proj"]
    k_proj = tensors["k_proj"]
    v_proj = tensors["v_proj"]
    cos = tensors["cos"]
    sin = tensors["sin"]

    Q = (x @ q_proj).view(B, N_HEAD, HEAD_DIM)
    K = (x @ k_proj).view(B, N_HEAD, HEAD_DIM)
    V = (x @ v_proj).view(B, N_HEAD, HEAD_DIM)

    Q = _rms_norm(Q)
    K = _rms_norm(K)

    Q = Q * cos + _rotate_half(Q) * sin
    K = K * cos + _rotate_half(K) * sin

    stacked = torch.stack([Q, K, V], dim=0)
    return stacked.reshape(-1, HEAD_DIM)
