"""PyTorch gold for examples/kv/kv_append_load.py.

For each request ``b``:
  * read the per-batch K cache prefix (``seq_len[b]`` tiles, each tile_N rows);
  * splice ``new_k[b]`` into row ``offset[b]`` of the last tile;
  * compute the ragged Q·K^T accumulation:
        out[b] = sum_t Q[b] . K_full[b, t]    (a (1, tile_N) row of scalars)

The ragged sequence dim is *absorbed* by the accumulation, so the final output
shape is ``(B, tile_N)`` — no ``max_seq`` dimension leaks. This matches what
``binary_matmul + accum_add(rank=1)`` produces over the dynamic-length K
stream in the STeP DSL.

Layout match to ``offchip_store`` on a ``[1, B]`` (tile ``1, tile_N``) stream:
  output shape = ``(B, tile_N)``.
"""
import torch


def compute_gold(dims, tensors):
    B          = dims["B"]
    D          = dims["D"]
    tile_N     = dims["tile_N"]
    maxN       = dims["maxN"]
    ROW_OFFSET = maxN // tile_N
    assert maxN % tile_N == 0, f"maxN={maxN} not divisible by tile_N={tile_N}"

    cache   = tensors["cache"]
    new_k   = tensors["new_k"]
    Q       = tensors["Q"]
    idx     = tensors["idx"].long()
    seq_len = tensors["seq_len"].long()
    offset  = tensors["offset"].long()

    out = torch.zeros(B, 1, tile_N, dtype=cache.dtype)
    for b in range(B):
        slot     = int(idx[b])
        n_tiles  = int(seq_len[b])
        ofs      = int(offset[b])
        assert n_tiles >= 1, f"seq_len[{b}]={n_tiles} must be >= 1"
        assert ofs < tile_N, f"offset[{b}]={ofs} must be < tile_N={tile_N}"
        assert n_tiles <= ROW_OFFSET, (
            f"seq_len[{b}]={n_tiles} exceeds slot capacity ROW_OFFSET={ROW_OFFSET}"
        )

        slot_start = slot * maxN
        K_full = cache[slot_start : slot_start + n_tiles * tile_N].clone()
        K_full[(n_tiles - 1) * tile_N + ofs] = new_k[b]
        # Per-tile Q·K^T accumulation: for each tile (tile_N rows of D), compute
        # Q[b] @ K_tile^T → (1, tile_N), then sum across tiles.
        K_tiles = K_full.reshape(n_tiles, tile_N, D)
        qkt = torch.matmul(Q[b].unsqueeze(0), K_tiles.transpose(-1, -2))  # (n_tiles, 1, tile_N)
        out[b] = qkt.sum(dim=0)

    # offchip_store on a [1, B] (tile 1, tile_N) stream flattens to (B, tile_N).
    return out.reshape(B, tile_N)
