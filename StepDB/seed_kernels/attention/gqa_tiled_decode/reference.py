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

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    query = tensors["query"]
    key = tensors["key"]
    value = tensors["value"]
    k_cache = tensors["k_cache"]
    v_cache = tensors["v_cache"]
    idx = tensors["idx"]
    seq_len_tiled = tensors["seq_len_tiled"]
    offset = tensors["offset"]
    tile_N = tensors["tile_N"]

    batch_size, num_kv_heads, query_per_kvhead, head_dim = query.shape
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
        Vh = V.transpose(0, 1)
        qh = query[b]                          # [num_kv_heads, query_per_kvhead, head_dim]
        logits = qh @ Kh.transpose(-1, -2)     # [num_kv_heads, query_per_kvhead, L]
        e = torch.exp(logits)
        num = e @ Vh
        denom = e.sum(dim=-1, keepdim=True)
        output[b] = num / denom

    return output
