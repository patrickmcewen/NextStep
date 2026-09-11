"""KV-cache append + ragged Q·K^T accumulation (single-head, K-only).

Illustrates the dynamic-sequence-length idiom STeP is built for: the per-
request seq length is a *ragged* dim that flows through `cache_read_addr_gen`
+ `random_offchip_load`, gets updated with the new K row, and is finally
*absorbed* by a matmul+accum so the final output is per-request — no
``max_seq`` padding leaks anywhere visible.

Math (PyTorch reference):
    for b in range(B):
        K_b = cache[idx[b]*maxN : idx[b]*maxN + (seq_len[b]-1)*tile_N]  # prior tiles
        K_b_last = cache[..]                                            # last existing tile
        K_b_last[offset[b]] = new_k[b]                                  # splice
        K_b_full = concat(K_b, K_b_last)                                # (seq_len[b]*tile_N, D)
        out[b] = sum_t Q[b] . K_b_full[t]                               # scalar (dot of Q with summed K rows)

Stage map (DSL):
  1. ``cache_read_addr_gen(idx, seq_len, ROW_OFFSET)`` produces a
     ``(1, B, max_seq)`` (tile 1,1) address stream with per-request
     ``ragged_lengths = seq_len``.
  2. ``random_offchip_load(cache, raddr=...)`` reads ``(1, B, max_seq)``
     (tile ``tile_N, D``) K tiles AND, because the raddr is ragged, zeros
     padded slots — so the padded positions hold zero data.
  3. ``filter_last_tile(seq_len)`` + ``flat_partition`` route the per-batch
     last tile (col 0) and prefix tiles (col 1); padded slots have all-zero
     multihot, dropping them from both branches.
  4. ``binary_set_offset`` + ``binary_row_wise_append`` splice ``new_k[b]``
     into row ``offset[b]`` of the last tile.
  5. ``flat_reassemble`` rejoins updated_last + prefix using the same
     multihot; padded rows get zero-filler tiles (consistent with the zeros
     already produced by step 2's ragged mask).
  6. ``binary_cache_write_addr_gen`` + ``random_offchip_store`` write the
     updated last tile back to the cache (no-op at value level in the
     functional sim).
  7. ``repeat_ref(Q, K)`` lifts Q from ``(1, B)`` to ``(1, B, max_seq)``
     matching the K stream cardinality.
  8. ``binary_matmul(Q_expanded, K, weight_transposed=True)`` per-tile
     ``Q[b,1,D] @ K[b,t,D]^T → (1, tile_N)``. Padded positions yield zero
     because their K is zero.
  9. ``accum_add(..., rank=1)`` sums over the trailing (ragged) tile-row
     dim. Padded slots contribute zero, so the per-request output is
     correct without leaking the ragged dim.
 10. ``offchip_store`` emits the per-request ``(1, tile_N)`` results.

Dims:
  B          : number of concurrent requests
  D          : head dimension
  tile_N     : sequence-length tile size
  maxN       : per-slot row capacity in the cache
  ROW_OFFSET : maxN // tile_N (per-slot tile stride)
"""


def tiled_reference(dims, tensors):
    B          = dims["B"]
    D          = dims["D"]
    tile_N     = dims["tile_N"]
    maxN       = dims["maxN"]
    ROW_OFFSET = maxN // tile_N

    # ---- Source loads ----
    # Metadata sources fan out to multiple consumers (idx → read addr-gen +
    # write addr-gen; seq_len → read addr-gen + filter_last_tile + write
    # addr-gen). The IR auto-broadcast rewriter inserts Broadcast nodes for
    # these fan-outs, so no explicit `broadcast(...)` is needed at the DSL
    # surface — same as end_to_end/attention/flashattn.py.
    idx     = metadata_gen(tensors["idx"])      # [B] (tile 1,1)
    seq_len = metadata_gen(tensors["seq_len"])  # [B] (tile 1,1)
    offset  = metadata_gen(tensors["offset"])   # [B] (tile 1,1)

    new_k = offchip_load(                       # [1, B] (tile 1, D)
        tensors["new_k"],
        stride=(1,), out_shape_tiled=(B,),
        tile_row=1, tile_col=D,
    )
    q = offchip_load(                           # [1, B] (tile 1, D)
        tensors["Q"],
        stride=(1,), out_shape_tiled=(B,),
        tile_row=1, tile_col=D,
    )

    # ---- Stage 1+2: ragged-K cache read (padded zeros via ragged_lengths) ----
    raddr = cache_read_addr_gen(idx, seq_len, row_offset=ROW_OFFSET)
    k_tiles = random_offchip_load(
        tensors["cache"], raddr=raddr,
        tile_row=tile_N, tile_col=D,
    )

    # ---- Stage 3+4: split off last tile + splice new K row ----
    last_sel = filter_last_tile(seq_len)
    last_tile, prefix_tiles = flat_partition(
        k_tiles, last_sel, n=2, partition_rank=0,
    )
    offset_flat = flatten(offset, min_rank=0, max_rank=1)
    new_k_flat  = flatten(new_k,  min_rank=0, max_rank=1)
    last_with_off = binary_set_offset(last_tile, offset_flat)
    last_updated  = binary_row_wise_append(last_with_off, new_k_flat)

    # ---- Stage 5: reassemble updated_last + prefix into the K stream ----
    # flat_reassemble emits stream `control.stream + (n_active=1,)` =
    # (1, B, max_seq, 1). Collapse the trailing 1 with the max_seq dim so the
    # downstream stream rank matches repeat_ref's contract.
    appended_k_raw = flat_reassemble(
        [last_updated, prefix_tiles],
        control=last_sel, reassemble_rank=0,
    )
    appended_k = flatten(appended_k_raw, min_rank=0, max_rank=1)

    # ---- Stage 6: write the updated last tile back to the cache ----
    # CacheWriteAddrGen yields ``idx*ROW_OFFSET + seq_len`` per request.
    waddr = binary_cache_write_addr_gen(idx, seq_len, row_offset=ROW_OFFSET)
    waddr_flat = flatten(waddr, min_rank=0, max_rank=1)
    random_offchip_store(
        tensors["cache"],
        wdata=last_updated, waddr=waddr_flat,
        tile_row=tile_N, tile_col=D,
    )

    # ---- Stage 7-9: ragged-Q·K^T → accum over seq dim → per-request output ----
    # repeat_ref adds a new innermost stream dim matching `appended_k`'s last
    # stream dim (max_seq), so Q's stream aligns with K's at one entry per
    # K tile of each request.
    q_expanded = repeat_ref(q, appended_k)
    qkt = binary_matmul(q_expanded, appended_k, weight_transposed=True)
    # Sum over the ragged (max_seq) stream dim. Padded slots are zero because
    # random_offchip_load already masked them out; their contribution is 0.
    pooled = accum_add(qkt, rank=1)

    # ---- Stage 10: store per-request output ----
    return offchip_store(pooled)
