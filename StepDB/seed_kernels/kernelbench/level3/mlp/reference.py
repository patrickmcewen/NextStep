"""PyTorch reference: KernelBench Level 3 / Problem 1 — basic MLP.

Ported from /workspace/KernelBench/KernelBench/level3/1_MLP.py. The
KernelBench source supports an arbitrary list of hidden sizes; this port
hardcodes two hidden layers (matching the level3/1_MLP.py test code) and
exposes B/D_in/D_h1/D_h2/D_out as the variable axes.

``Model.forward`` takes the data tensor AND all weights/biases as explicit
positional args — rather than wrapping ``nn.Sequential`` — so every byte
the kernel sees is produced under ``get_inputs``' control. This keeps the
RNG order fully visible in ``get_inputs`` and the matching precompute
function (``StepDB/precompute.py:_precompute_kernelbench_mlp``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, w1, b1, w2, b2, w3, b3):
        # nn.Linear computes  x @ W.T + b , so weights have shape (out, in).
        h1 = F.relu(x @ w1.T + b1)
        h2 = F.relu(h1 @ w2.T + b2)
        return h2 @ w3.T + b3


def get_init_inputs(dims):
    return []


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    D_in = dims["D_in"]
    D_h1 = dims["D_h1"]
    D_h2 = dims["D_h2"]
    D_out = dims["D_out"]

    # RNG order mirrors KernelBench's nn.Sequential(Linear, ReLU, Linear,
    # ReLU, Linear) followed by torch.rand(batch, input_size). nn.Linear
    # consumes RNG for weight (Kaiming-uniform) then bias (uniform).
    fc1 = nn.Linear(D_in, D_h1)
    fc2 = nn.Linear(D_h1, D_h2)
    fc3 = nn.Linear(D_h2, D_out)
    x = torch.rand(B, D_in)

    # Biases are emitted as (1, D) rather than (D,) so they are >=2-D and
    # loadable by StepDB's offchip_load. Broadcasting (B,D) + (1,D) in
    # Model.forward is bit-identical to (B,D) + (D,).
    return [
        x,
        fc1.weight.detach().clone().contiguous(),
        fc1.bias.detach().clone().contiguous().unsqueeze(0),
        fc2.weight.detach().clone().contiguous(),
        fc2.bias.detach().clone().contiguous().unsqueeze(0),
        fc3.weight.detach().clone().contiguous(),
        fc3.bias.detach().clone().contiguous().unsqueeze(0),
    ]


def compute_gold(dims):
    with torch.no_grad():
        return Model(*get_init_inputs(dims))(*get_inputs(dims))
