def tiled_reference(dims: dict, tensors: dict):
    """
    Head split + permute: (S, H*D) -> (H, S*D).

    PyTorch equivalent:
        x.view(S, H, D).permute(1, 0, 2).reshape(H, S*D)

    Key insight: when each tile already isolates the rearranged axis
    (here D lives entirely within one tile of shape (1, D)), the
    view+permute collapses into a buffer-grid axis swap. No
    retile_streamify or reshape_stream is needed -- only bufferize +
    streamify with swapped strides.

    Stride choice rationale. The buffer grid is (S, H) with row-major
    flat index s*H + h. Output position (h_out, s_out) should map to
    buffer position (s=s_out, h=h_out), i.e. flat = s_out*H + h_out.
    With stride=(1, H) and out_shape_tiled=(H, S):
        linear_idx(h_out, s_out) = h_out*1 + s_out*H = s_out*H + h_out.

    NOT EXPRESSIBLE with bufferize + streamify alone: tile-grid swaps or
    cyclic shifts (e.g. rotate_half's [1, 0] reorder). The stride
    formula linear_idx = sum(idx[k] * stride[k]) always yields 0 at
    (0,...,0), so the first emitted output tile must be buffer tile 0.
    """
    S = dims["S"]
    H = dims["H"]
    D = dims["D"]

    # Each tile is one head's D-vector; the tile-grid is (S, H).
    loaded = offchip_load(
        tensors["input"],
        stride=(H, 1),
        out_shape_tiled=(S, H),
        tile_row=1,
        tile_col=D,
    )  # stream (1, S, H), tile (1, D)

    buf = bufferize(loaded, rank=2)
    out = streamify(buf, stride=(1, H), out_shape_tiled=(H, S))
    # stream (1, H, S), tile (1, D)

    return offchip_store(out)  # (H, S*D)
