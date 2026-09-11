"""PyTorch reference: tiled KV cache last-tile append (no attention).

Extracted from `seed_kernels/gqa_tiled_decode/reference.py:47-58` and
mirrored in the streaming dataflow at
`step_tl/end_to_end/attention/flashattn.py:84-434` (Stages 1-6: load
tiled K/V cache, partition off the last tile, append the new key/value
at `offset` within that tile, then write the modified tile back). The
gold output is exactly the data the simulator's RandomOffChipStore
writes back per (batch, kv_head): a single `[tile_N, head_dim]` tile.

Conceptually, for each (batch `b`, kv_head `h`):

    bi = idx[b]
    last_tile = k_cache[bi, (seq_len_tiled[b]-1)*tile_N : seq_len_tiled[b]*tile_N, h]
    last_tile[offset[b]] = key[b, h]
    -> write back as the new last tile

The output is laid out to match the streaming-dataflow store: per
(b, h) emit the modified K tile then the modified V tile, walking
(b, h) row-major. Returned as a 2-D `[2 * batch * num_kv_heads *
tile_N, head_dim]` tensor so it matches `_untile_store`'s reshape.

Inputs come from StepDB/precompute.py via the `tensors` arg.
"""
import torch


def compute_gold(dims, tensors):
    key = tensors["key"]
    value = tensors["value"]
    k_cache = tensors["k_cache"]
    v_cache = tensors["v_cache"]
    idx = tensors["idx"]
    seq_len_tiled = tensors["seq_len_tiled"]
    offset = tensors["offset"]
    tile_N = tensors["tile_N"]

    batch_size, num_kv_heads, head_dim = key.shape
    new_k_tile = torch.zeros(
        batch_size, num_kv_heads, tile_N, head_dim, dtype=k_cache.dtype
    )
    new_v_tile = torch.zeros(
        batch_size, num_kv_heads, tile_N, head_dim, dtype=v_cache.dtype
    )

    for b in range(batch_size):
        bi = int(idx[b].item())
        num_tiles = int(seq_len_tiled[b].item())
        off = int(offset[b].item())
        assert num_tiles >= 1
        assert 0 <= off < tile_N
        start = (num_tiles - 1) * tile_N

        K_tile = k_cache[bi, start : start + tile_N].clone()
        V_tile = v_cache[bi, start : start + tile_N].clone()
        K_tile[off] = key[b]
        V_tile[off] = value[b]

        new_k_tile[b] = K_tile.transpose(0, 1)
        new_v_tile[b] = V_tile.transpose(0, 1)

    # Interleave K and V at the (batch, kv_head) granularity to mirror
    # StaticReassemble([append_k, append_v], merge_rank=0). Reshape to 2-D
    # to match _untile_store's output shape.
    return torch.stack([new_k_tile, new_v_tile], dim=2).reshape(-1, head_dim)
