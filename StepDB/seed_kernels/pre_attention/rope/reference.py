"""PyTorch reference: Rotary Position Embedding (RoPE) on Q and K.

    Q = Q * cos + rotate_half(Q) * sin
    K = K * cos + rotate_half(K) * sin

where rotate_half(x) = concat([-x[..., D/2:], x[..., :D/2]], dim=-1).

The precompute stacks Q and K along the heads axis into a single tensor
QK [batch, num_q_heads + num_kv_heads, head_dim], since rotate_half +
cos/sin multiply-add is per-head and applies identically to Q and K
rows. compute_gold runs that single pipeline and flattens to
[batch * (num_q_heads + num_kv_heads), head_dim] — rows ordered per-batch:
num_q_heads Q rows then num_kv_heads K rows.
"""
import torch


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def compute_gold(dims, tensors):
    QK = tensors["QK"]
    cos = tensors["cos"]
    sin = tensors["sin"]
    out = QK * cos + _rotate_half(QK) * sin
    return out.reshape(-1, dims["head_dim"])
