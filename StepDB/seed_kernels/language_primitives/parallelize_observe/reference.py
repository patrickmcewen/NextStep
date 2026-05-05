"""PyTorch reference: observable Parallelize semantic test.

Builds an input where every element of row b equals b. The Step graph
applies a per-consumer additive marker (consumer i adds i*MARKER) between
Parallelize and StaticReassemble, so the output reveals which consumer
saw each row.

This reference assumes round-robin per-rank-1-unit dispatch — i.e. row b
goes to consumer (b % par_factor) — which is what the Rust Parallelize
operator does for switch_cycles=[1,...]. If the Python functional emulator
slices differently (e.g. contiguous chunks where row b goes to consumer
b // (B // par_factor)), its execute(graph) output will diverge from this
reference and from the Rust simulator.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, par_factor, tile_m):
        super().__init__()
        self.par_factor = par_factor
        self.tile_m = tile_m

    def forward(self, x):
        M = x.shape[0]
        row = torch.arange(M, device=x.device) // self.tile_m
        # Consumer of row b under round-robin per rank-1 unit dispatch.
        consumer = (row % self.par_factor).to(x.dtype).unsqueeze(1)
        # Mirrors the per-consumer MulImmediate(i+1) in step_impl.py.
        return x * (consumer + 1)


def get_inputs(dims):
    torch.manual_seed(SEED)
    B = dims["B"]
    K = dims["K"]
    tile_m = dims["tile_m"]
    tile_k = dims["tile_k"]
    M = B * tile_m
    K_total = K * tile_k
    # x[b, :] = b + 1 (offset by 1 so even consumer 0 produces a non-zero
    # signal that uniquely identifies the source batch index).
    row_val = (torch.arange(B, dtype=torch.float32) + 1).repeat_interleave(tile_m).unsqueeze(1)
    x = row_val.expand(M, K_total).contiguous()
    return [x]


def get_init_inputs(dims):
    return [dims["par_factor"], dims["tile_m"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
