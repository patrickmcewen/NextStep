def tiled_reference(dims, tensors):
    """
    Small‑tile GEMM with explicit K‑dim reduction.
    Tile sizes are taken from ``dims`` (tile_m, tile_k, tile_n) and are
    small enough to satisfy the 5 MiB on‑chip budget.
    The algorithm streams over the K dimension, broadcasting each A‑tile
    across the N axis and each B‑tile across the M axis, then accumulates
    the partial results with a sequential element‑wise add.
    """
    # Matrix dimensions
    M = dims["M"]
    K = dims["K"]
    N = dims["N"]

    # Tile sizes (provided by the problem statement)
    tile_m = dims["tile_m"]   # 128
    tile_k = dims["tile_k"]   # 128
    tile_n = dims["tile_n"]   # 256

    # Number of tiles along each axis
    M_tiles = M // tile_m      # 4096 / 128 = 32
    K_tiles = K // tile_k      # 4096 / 128 = 32
    N_tiles = N // tile_n      # 4096 / 256 = 16

    # Accumulator for the output (initially None)
    C_acc = None

    # Loop over the K‑tiles, broadcasting each tile across the other axis.
    for k_idx in range(K_tiles):
        # --------------------------------------------------------------
        # Load a slice of A: (tile_m, tile_k) tiled over (M_tiles, N_tiles)
        #   * stride[0] = K_tiles   – move to the next K‑tile when the M‑index changes
        #   * stride[1] = 0         – broadcast the same A‑tile across all N‑tiles
        #   * start_tile_idx = k_idx – selects the k‑th K‑tile for every M‑tile
        # --------------------------------------------------------------
        A_k = offchip_load(
            tensors["A"],
            (K_tiles, 0),                 # stride
            (M_tiles, N_tiles),           # out_shape_tiled (stream shape)
            tile_m,                       # tile_row
            tile_k,                       # tile_col
            start_tile_idx=k_idx,
        )

        # --------------------------------------------------------------
        # Load a slice of B: (tile_k, tile_n) tiled over (M_tiles, N_tiles)
        #   * stride[0] = 0         – broadcast the same B‑tile across all M‑tiles
        #   * stride[1] = 1         – walk across N‑tiles within this K‑tile row
        #   * start_tile_idx = k_idx * N_tiles – base index of the K‑tile row
        # --------------------------------------------------------------
        B_k = offchip_load(
            tensors["B"],
            (0, 1),                       # stride
            (M_tiles, N_tiles),           # out_shape_tiled (stream shape)
            tile_k,                       # tile_row
            tile_n,                       # tile_col
            start_tile_idx=k_idx * N_tiles,
        )

        # Multiply the two broadcasted tiles (produces stream shape (M_tiles, N_tiles),
        # tile shape (tile_m, tile_n))
        C_part = binary_matmul(A_k, B_k)

        # Accumulate the partial result into the running sum.
        if C_acc is None:
            C_acc = C_part
        else:
            C_acc = binary_add(C_acc, C_part)

    # Write the accumulated result back to off‑chip memory.
    return offchip_store(C_acc)