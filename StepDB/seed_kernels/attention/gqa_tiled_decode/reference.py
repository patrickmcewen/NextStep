"""PyTorch reference: GQA single-token decode with a tiled KV cache and
in-place append of the new key/value before attention.

Extracted from `step_tl/end_to_end/static_baseline.py:519-551` (the call to
`build_static_par`) and `step_tl/end_to_end/attention/flashattn.py`. The
flashattn graph reads `seq_len_tiled[b]` consecutive tiles of size `tile_N`
from the per-batch KV cache, replaces position `(seq_len_tiled[b] - 1) *
tile_N + offset[b]` of the loaded K and V with the incoming key/value, then
computes the softmax-weighted average

    out = sum_n exp(Q . K_n) * V_n  /  sum_n exp(Q . K_n)

which is algebraically identical to softmax(Q @ K^T) @ V. The graph skips
the max-subtraction numerical-stability trick and the 1/sqrt(head_dim)
scaling on the logits; the reference matches that math verbatim so the
inputs and outputs line up bit-for-bit with the graph. The cache write-back
is modelled as a side effect of the simulator and is intentionally not part
of the gold output.
"""
import torch
import torch.nn as nn

SEED = 42


class Model(nn.Module):
    def __init__(self, tile_N):
        super().__init__()
        self.tile_N = tile_N

    def forward(self, query, key, value, k_cache, v_cache, idx, seq_len_tiled, offset):
        batch_size, num_kv_heads, query_per_kvhead, head_dim = query.shape
        tile_N = self.tile_N
        assert key.shape == (batch_size, num_kv_heads, head_dim)
        assert value.shape == (batch_size, num_kv_heads, head_dim)
        assert k_cache.shape[0] == v_cache.shape[0]
        assert k_cache.shape[2] == num_kv_heads and k_cache.shape[3] == head_dim
        assert idx.shape == (batch_size,)
        assert seq_len_tiled.shape == (batch_size,)
        assert offset.shape == (batch_size,)

        output = torch.zeros(
            batch_size, num_kv_heads, query_per_kvhead, head_dim, dtype=query.dtype
        )

        for b in range(batch_size):
            bi = int(idx[b].item())
            num_tiles = int(seq_len_tiled[b].item())
            off = int(offset[b].item())
            assert num_tiles >= 1
            assert 0 <= off < tile_N
            L = num_tiles * tile_N
            insert_pos = (num_tiles - 1) * tile_N + off

            K = k_cache[bi, :L].clone()  # [L, num_kv_heads, head_dim]
            V = v_cache[bi, :L].clone()
            K[insert_pos] = key[b]
            V[insert_pos] = value[b]

            Kh = K.transpose(0, 1)                 # [num_kv_heads, L, head_dim]
            Vh = V.transpose(0, 1)                 # [num_kv_heads, L, head_dim]
            qh = query[b]                          # [num_kv_heads, query_per_kvhead, head_dim]
            logits = qh @ Kh.transpose(-1, -2)     # [num_kv_heads, query_per_kvhead, L]
            e = torch.exp(logits)
            num = e @ Vh                           # [num_kv_heads, query_per_kvhead, head_dim]
            denom = e.sum(dim=-1, keepdim=True)
            output[b] = num / denom

        return output


def get_inputs(dims):
    torch.manual_seed(SEED)
    batch_size = dims["batch_size"]
    num_kv_heads = dims["num_kv_heads"]
    query_per_kvhead = dims["query_per_kvhead"]
    head_dim = dims["head_dim"]
    max_seq_len_tiles = dims["max_seq_len_tiles"]
    tile_N = dims["tile_N"]
    max_seq_len = max_seq_len_tiles * tile_N

    query = torch.randn(batch_size, num_kv_heads, query_per_kvhead, head_dim)
    key = torch.randn(batch_size, num_kv_heads, head_dim)
    value = torch.randn(batch_size, num_kv_heads, head_dim)
    k_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)
    v_cache = torch.randn(batch_size, max_seq_len, num_kv_heads, head_dim)

    idx = torch.arange(batch_size, dtype=torch.int64)
    seq_len_tiled = torch.randint(
        low=1, high=max_seq_len_tiles + 1, size=(batch_size,), dtype=torch.int64
    )
    offset = torch.randint(low=0, high=tile_N, size=(batch_size,), dtype=torch.int64)
    return [query, key, value, k_cache, v_cache, idx, seq_len_tiled, offset]


def get_init_inputs(dims):
    return [dims["tile_N"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    return model(*get_inputs(dims))
