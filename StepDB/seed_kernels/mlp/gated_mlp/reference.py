"""PyTorch reference: Gated MLP (SwiGLU variant).

Computes: silu(x @ gate_w) * (x @ up_w) @ down_w

This is the core computation of each MoE expert in Qwen/Mixtral models.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch
import torch.nn.functional as F


def compute_gold(dims, tensors):
    x = tensors["x"]
    gate_w = tensors["gate_w"]
    up_w = tensors["up_w"]
    down_w = tensors["down_w"]
    with torch.no_grad():
        return (F.silu(x @ gate_w) * (x @ up_w)) @ down_w
