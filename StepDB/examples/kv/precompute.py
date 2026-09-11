"""Input factory for examples/kv/kv_append_load.py.

Builds:
  cache   : [B*maxN, D]    float32   prior K rows in [b*maxN, b*maxN+num_tokens[b]),
                                     zeros for unused capacity.
  new_k   : [B, D]         float32   the new K row to splice in per request.
  Q       : [B, D]         float32   the query row per request.
  idx     : [B]            int64     cache slot per request (here: identity 0..B-1).
  seq_len : [B]            int64     tile count *after* appending the new K:
                                     ceil((num_tokens[b] + 1) / tile_N).
  offset  : [B]            int64     intra-tile row for the new K:
                                     num_tokens[b] % tile_N.
"""
import torch


def precompute(dims, seed=0):
    g = torch.Generator().manual_seed(seed)
    B       = dims["B"]
    D       = dims["D"]
    tile_N  = dims["tile_N"]
    maxN    = dims["maxN"]
    assert maxN % tile_N == 0, f"maxN={maxN} not divisible by tile_N={tile_N}"

    # Pick a varied per-request token count so seq_len and offset are
    # non-trivial. Cap at maxN - 1 so there's space to append.
    num_tokens = torch.randint(
        low=1, high=maxN, size=(B,), generator=g, dtype=torch.int64,
    )

    cache = torch.zeros(B * maxN, D, dtype=torch.float32)
    for b in range(B):
        n = int(num_tokens[b])
        cache[b * maxN : b * maxN + n] = torch.randn(n, D, generator=g)

    new_k = torch.randn(B, D, generator=g)
    Q     = torch.randn(B, D, generator=g)

    # int64 → metadata_gen wraps a raw 1-D tensor; the value-level uint64
    # interpretation is asserted at the IR level. Using int64 here keeps the
    # tensor compatible with arange/operators that lack uint64 kernels.
    seq_len = (num_tokens + 1 + tile_N - 1) // tile_N
    offset  = num_tokens % tile_N
    idx     = torch.arange(B, dtype=torch.int64)

    return {
        "cache":   cache,
        "new_k":   new_k,
        "Q":       Q,
        "idx":     idx,
        "seq_len": seq_len,
        "offset":  offset,
    }
