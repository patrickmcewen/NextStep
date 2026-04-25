"""PyTorch reference: QKV generation for a transformer attention block.

Computes the pre-attention block of a (MHA) transformer layer:
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
section the rows are batch-outer, head-inner — one [N_HEAD, HEAD_DIM]
tile per batch — matching the stream of B tiles of shape
[N_HEAD, HEAD_DIM] produced by the step graph.
"""
import torch
import torch.nn as nn

SEED = 42
EPS = 1e-6


def _rotate_half(x):
    """Rotary embedding helper: concat([-x2, x1], dim=-1)."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _rms_norm(x, eps=EPS):
    """RMS normalization along last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class Model(nn.Module):
    def __init__(self, n_head, head_dim, eps=EPS):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.eps = eps

    def forward(self, x, q_proj, k_proj, v_proj, cos, sin):
        # x:         [B, D]
        # q/k/v_proj: [D, N_HEAD * HEAD_DIM]
        # cos, sin:  [B, 1, HEAD_DIM]
        B = x.shape[0]
        Q = (x @ q_proj).view(B, self.n_head, self.head_dim)
        K = (x @ k_proj).view(B, self.n_head, self.head_dim)
        V = (x @ v_proj).view(B, self.n_head, self.head_dim)

        Q = _rms_norm(Q, self.eps)
        K = _rms_norm(K, self.eps)

        Q = Q * cos + _rotate_half(Q) * sin
        K = K * cos + _rotate_half(K) * sin

        return torch.stack([Q, K, V], dim=0)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    D = dims["D"]
    N_HEAD = dims["N_HEAD"]
    HEAD_DIM = dims["HEAD_DIM"]
    assert HEAD_DIM % 2 == 0, "HEAD_DIM must be even for rotate_half"

    q_proj = torch.randn(D, N_HEAD * HEAD_DIM)
    k_proj = torch.randn(D, N_HEAD * HEAD_DIM)
    v_proj = torch.randn(D, N_HEAD * HEAD_DIM)
    x = torch.randn(B, D)
    cos = torch.randn(B, 1, HEAD_DIM)
    sin = torch.randn(B, 1, HEAD_DIM)
    return [x, q_proj, k_proj, v_proj, cos, sin]


def get_init_inputs(dims):
    return [dims["N_HEAD"], dims["HEAD_DIM"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    stacked = model(*get_inputs(dims))  # [3, B, N_HEAD, HEAD_DIM]
    # step_impl emits batch-outer, head-inner within each Q/K/V section.
    return stacked.reshape(-1, dims["HEAD_DIM"])
