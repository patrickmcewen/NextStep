"""PyTorch reference: Identity through bufferize/streamify roundtrip.

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""


def compute_gold(dims, tensors):
    return tensors["input"].clone()
