def tiled_reference(dims: dict, tensors: dict):
    """
    Tile-grid transpose: (M, K) -> (K, M).

    A full tensor transpose decomposes into two independent pieces:
      (a) per-tile transpose -- each (tile_m, tile_k) tile becomes
          (tile_k, tile_m). Done at load time via transposed=True.
      (b) tile-grid axis swap -- the (M_g, K_g) grid of tiles is re-emitted
          as a (K_g, M_g) grid. Done via bufferize + streamify with
          swapped strides.

    The streamify formula linear_idx(*idx) = sum(idx[k] * stride[k]) with
    stride=(1, K_g) and out_shape_tiled=(K_g, M_g) reads the row-major
    flat buffer in column-major order over the tile grid:
        linear_idx(k, m) = k*1 + m*K_g  ->  picks buffer flat at m*K_g + k,
    which is buffer position (m, k). That is exactly what a tile-grid
    transpose needs.

    NOT EXPRESSIBLE with bufferize + streamify alone: tile-grid swaps or
    cyclic shifts (e.g. rotate_half's [1, 0] reorder), because the
    streamify stride formula always satisfies linear_idx(0,...,0) = 0.
    Sub-tile element rearrangement is also out of scope -- bufferize and
    streamify operate at tile granularity only.
    """
    M = dims["M"]
    K = dims["K"]
    tile_m = dims["tile_m"]
    tile_k = dims["tile_k"]

    M_g = M // tile_m
    K_g = K // tile_k

    # (a) Per-tile transpose at load: each tile comes back as (tile_k, tile_m).
    loaded = offchip_load(
        tensors["input"],
        stride=(K_g, 1),
        out_shape_tiled=(M_g, K_g),
        tile_row=tile_m,
        tile_col=tile_k,
        transposed=True,
    )  # stream (1, M_g, K_g), tile (tile_k, tile_m)

    # (b) Tile-grid axis swap via bufferize + streamify.
    buf = bufferize(loaded, rank=2)
    out = streamify(buf, stride=(1, K_g), out_shape_tiled=(K_g, M_g))
    # stream (1, K_g, M_g), tile (tile_k, tile_m)

    return offchip_store(out)  # (K, M)
