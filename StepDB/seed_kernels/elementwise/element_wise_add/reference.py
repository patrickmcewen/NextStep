"""PyTorch reference: Element-wise addition (residual connection).

Computes A + B - used for residual add in transformer decoder layers
after attention and MoE blocks.
Inputs come from StepDB/precompute.py via the `tensors` arg.
"""


def compute_gold(dims, tensors):
    return tensors["A"] + tensors["B"]
