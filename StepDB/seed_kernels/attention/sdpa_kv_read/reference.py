"""PyTorch reference: single-token GQA decode with KV cache append + read.

Steps per call:
  1. Project x into per-head Q, K, V.
  2. Append the new K, V into K_cache / V_cache at slot `seq_len`.
  3. Run scaled-dot-product attention over the prefix
     (positions 0..seq_len inclusive) with GQA head mapping
       kv_head = qo_head // (num_heads // num_kv_heads).

Shapes:
  x:        [1, hidden_dim]
  W_q:      [hidden_dim, num_heads    * head_dim]
  W_k, W_v: [hidden_dim, num_kv_heads * head_dim]
  K_cache:  [batch_size, seq_len + 1, num_kv_heads, head_dim]
  V_cache:  [batch_size, seq_len + 1, num_kv_heads, head_dim]
  output:   [num_heads, head_dim]
"""
import math

import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, W_q, W_k, W_v, K_cache, V_cache, seq_len, batch_idx):
        num_kv_heads, head_dim = K_cache.shape[-2], K_cache.shape[-1]
        num_heads = W_q.shape[1] // head_dim
        assert num_heads % num_kv_heads == 0
        gqa_ratio = num_heads // num_kv_heads
        sm_scale = 1.0 / math.sqrt(head_dim)

        q = (x @ W_q).view(num_heads, head_dim)
        K_cache[batch_idx, seq_len] = (x @ W_k).view(num_kv_heads, head_dim)
        V_cache[batch_idx, seq_len] = (x @ W_v).view(num_kv_heads, head_dim)

        k = K_cache[batch_idx, : seq_len + 1].repeat_interleave(gqa_ratio, dim=1)  # [N, H, D]
        v = V_cache[batch_idx, : seq_len + 1].repeat_interleave(gqa_ratio, dim=1)  # [N, H, D]

        scores = torch.einsum("hd,nhd->hn", q, k) * sm_scale  # [H, N]
        attn = torch.softmax(scores, dim=-1)                  # [H, N]
        return torch.einsum("hn,nhd->hd", attn, v)            # [H, D]


def get_inputs(dims):
    torch.manual_seed(SEED)
    hidden_dim = dims["hidden_dim"]
    seq_len = dims["seq_len"]
    x = torch.randn(1, hidden_dim)
    num_heads = dims["num_heads"]
    head_dim = dims["head_dim"]
    num_kv_heads = dims["num_kv_heads"]
    batch_size = dims["batch_size"]
    batch_idx = dims["batch_idx"]
    W_q = torch.randn(hidden_dim, num_heads * head_dim)
    W_k = torch.randn(hidden_dim, num_kv_heads * head_dim)
    W_v = torch.randn(hidden_dim, num_kv_heads * head_dim)
    K_cache = torch.randn(batch_size, seq_len + 1, num_kv_heads, head_dim)
    V_cache = torch.randn(batch_size, seq_len + 1, num_kv_heads, head_dim)
    return x, W_q, W_k, W_v, K_cache, V_cache, seq_len, batch_idx


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model()
    x, W_q, W_k, W_v, K_cache, V_cache, seq_len, batch_idx = get_inputs(dims)
    return model(x, W_q, W_k, W_v, K_cache, V_cache, seq_len, batch_idx)
