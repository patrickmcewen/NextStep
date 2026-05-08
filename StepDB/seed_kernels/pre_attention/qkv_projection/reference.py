"""PyTorch reference: QKV linear projection.

Computes the concatenated QKV as a single matmul:
  output = x @ W    [B, D] @ [D, proj_dim] -> [B, proj_dim]

Extracted from step_tl/end_to_end/attention/qkv_gen.py::projection.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    return torch.matmul(tensors["x"], tensors["W"])
