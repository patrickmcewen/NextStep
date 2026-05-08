"""PyTorch reference: complex-reshape exercise.

A single-tensor pipeline distilled from basic_prefill_attention's lines
87-112: projection -> view -> elementwise scale -> view + permute ->
matmul-with-transpose against a separately-supplied K tensor. K is given
per-head ([S, num_kv_heads, head_dim]); only Q goes through projection.

Pipeline (with all ops named in the [bracketed] tags):

    Q  = X @ W                                             # [proj]
    Q  = Q.view(S, H, D)                                   # [view-1]
    Q  = Q * scale                                         # [scale]
    Qh = Q.view(S, Hkv, qpkv, D).permute(1, 2, 0, 3)       # [view-2 + permute]
    Kh = K.permute(1, 0, 2).unsqueeze(1)                   # [permute]
    return Qh @ Kh.transpose(-1, -2)                       # [matmul-T]

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import math

import torch


def compute_gold(dims, tensors):
    seq_len = dims["seq_len"]
    num_heads = dims["num_heads"]
    num_kv_heads = dims["num_kv_heads"]
    head_dim = dims["head_dim"]
    assert num_heads % num_kv_heads == 0, (
        f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    qpkv = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    x = tensors["x"]   # [S, HID]
    w = tensors["w"]   # [HID, num_heads * head_dim]
    k = tensors["k"]   # [S, num_kv_heads, head_dim]

    with torch.no_grad():
        Q = x @ w                                                       # [S, H*D]
        Q = Q.view(seq_len, num_heads, head_dim)                        # [S, H, D]
        Q = Q * scale
        Qh = (
            Q.view(seq_len, num_kv_heads, qpkv, head_dim)
             .permute(1, 2, 0, 3)
        )                                                               # [Hkv, qpkv, S, D]
        Kh = k.permute(1, 0, 2).unsqueeze(1)                            # [Hkv, 1, S, D]
        return Qh @ Kh.transpose(-1, -2)                                # [Hkv, qpkv, S, S]
