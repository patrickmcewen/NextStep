# Origin: Consolidation of KernelBench level1/1 (square), level1/2 (rectangular),
#         level1/6 (large_k), level1/7 (small_k). Dimensions now come from bench_config.yaml.
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.matmul(A, B)


def get_inputs(dims):
    torch.manual_seed(SEED)
    M, K, N = dims["M"], dims["K"], dims["N"]
    return [torch.randn(M, K), torch.randn(K, N)]


def get_init_inputs(dims):
    return []


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
