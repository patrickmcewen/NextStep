def compute_qkv(Q, K, V, *, out_shapes, out_perms=None):
    """
    Transform the on‑chip Q, K, V streams into the shapes required by the
    attention kernel.

    Input tensors have shape (S, H, D) = (seq_len, num_heads, head_dim) with
    tiling (tile_rows = H, tile_cols = D).  The desired outputs are:
        Qh : (num_kv_heads, q_per_kv, seq_len, head_dim)
             → (4, 4, 64, 32)
        Kh, Vh : (num_kv_heads, 1, seq_len, head_dim)
                 → (4, 1, 64, 32)

    The transformation proceeds by:
      * retile_streamify(..., chunk=1) – moves the head dimension from the
        tile into the stream (tile rows become 1).
      * reshape_stream – first splits the combined stream dimension (S * Hkv)
        into (Hkv, S), then (Hkv) into (num_kv_heads, q_per_kv).
      * accum_retile_row – absorbs the innermost stream dimension (the
        sequence length) back into the tile rows, giving tile_rows = S.
      * promote – for K and V, inserts a singleton stream dimension so that
        the final shape is (num_kv_heads, 1, S, D).

    All operations are pure DSL calls; no raw tensor arithmetic is used.
    """
    # ----- Q ---------------------------------------------------------------
    # Move head dimension (tile rows = 16) into the stream.
    q = retile_streamify(Q, chunk=1, split_row=True)          # (1024, 1, 32)
    # Split stream (1024 = 16 * 64) → (16, 64)
    q = reshape_stream(q, chunk_size=64, rank=0)              # (16, 64, 1, 32)
    # Split the outer stream (16) → (4, 4)
    q = reshape_stream(q, chunk_size=4, rank=1)               # (4, 4, 64, 1, 32)
    # Absorb innermost stream (64) into tile rows.
    Qh = accum_retile_row(q, rank=1)                          # (4, 4, 64, 32)

    # ----- helper for K / V -------------------------------------------------
    def _kv_transform(x):
        # Move head dimension (tile rows = 4) into the stream.
        x = retile_streamify(x, chunk=1, split_row=True)      # (256, 1, 32)
        # Split stream (256 = 4 * 64) → (4, 64)
        x = reshape_stream(x, chunk_size=64, rank=0)          # (4, 64, 1, 32)
        # Absorb innermost stream (64) into tile rows → (4, 64, 32)
        x = accum_retile_row(x, rank=1)
        # Insert a singleton stream dimension: (4, 1, 64, 32)
        x = promote(x, rank=0)
        return x

    # ----- K ---------------------------------------------------------------
    Kh = _kv_transform(K)

    # ----- V ---------------------------------------------------------------
    Vh = _kv_transform(V)

    return Qh, Kh, Vh