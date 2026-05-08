"""PyTorch reference: Outer product with batch accumulation.

Computes sum over batch dimension of outer products: sum_b(a[b,:] outer b[b,:]).
Equivalent to A^T @ B where A is [B, M] and B is [B, N].

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""


def compute_gold(dims, tensors):
    return tensors["A"].T @ tensors["B_data"]
