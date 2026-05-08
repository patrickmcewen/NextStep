"""PyTorch reference: 2D identity (copy). Tests the load/store data path.

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""


def compute_gold(dims, tensors):
    return tensors["input"].clone()
