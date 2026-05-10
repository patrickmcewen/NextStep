def tiled_reference(dims: dict, tensors: dict):
    """
    4D axis permute: (B, H, S, D) -> (B*S, H*D).

    PyTorch equivalent:
        x.permute(0, 2, 1, 3).reshape(B*S, H*D)

    n-D buffer-grid permutation recipe:
      1) Load with tile (1, D) so the inner axis lives within one tile.
      2) Choose out_shape_tiled to enumerate buffer positions in the
         row-major order of the desired OUTPUT layout. Equivalently:
         the i-th output index walks buffer dim pi, where pi is
         the i-th axis of the output permutation.
      3) Set stride[i] = flat_stride_of_buffer_dim_pi, where
         flat_stride[j] = prod(buffer_shape[j+1:]).

    For buffer grid (B, H, S):  flat_stride = (H*S, S, 1).
    Output order (B, S, H) -> permutation pi = (0, 2, 1) ->
        stride = (flat_stride[0], flat_stride[2], flat_stride[1])
               = (H*S,           1,               S).
    Then linear_idx(b, s, h) = b*H*S + s*1 + h*S = b*H*S + h*S + s,
    which is the row-major flat index of buffer position (b, h, s).

    NOT EXPRESSIBLE with bufferize + streamify alone: tile-grid swaps or
    cyclic shifts (e.g. rotate_half's [1, 0]). The stride formula
    linear_idx = sum(idx[k] * stride[k]) always yields 0 at (0,...,0),
    so the first emitted tile must be buffer tile 0.
    """
    B = dims["B"]
    H = dims["H"]
    S = dims["S"]
    D = dims["D"]

    # Identity load: tile-grid (B, H, S), each tile is one D-vector.
    loaded = offchip_load(
        tensors["input"],
        stride=(H * S, S, 1),
        out_shape_tiled=(B, H, S),
        tile_row=1,
        tile_col=D,
    )  # stream (1, B, H, S), tile (1, D)

    buf = bufferize(loaded, rank=3)
    out = streamify(buf, stride=(H * S, 1, S), out_shape_tiled=(B, S, H))
    # stream (1, B, S, H), tile (1, D)

    return offchip_store(out)  # (B*S, H*D)
