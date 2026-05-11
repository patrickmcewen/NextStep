def qkv_projection(normed, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    """
    Compute Q, K, V streams.

    * `normed` – on‑chip token stream: shape (seq_len, 1, hidden).
    * `q_proj`, `k_proj`, `v_proj` – RAW weight matrices.
    * `cos` – provides the head dimension (last dimension).

    Steps for each projection:
    1. Load the RAW weight with `offchip_load`, broadcasting the single
       weight tile across the token dimension (`out_shape_tiled=[seq_len]`).
    2. Collapse the leading singleton and the token dimension with `flatten`
       → a weight stream whose stream shape matches `normed`.
    3. Perform batched matrix multiplication (`binary_matmul`).
    4. Split the hidden dimension into heads using `retile_streamify`
       (column split, chunk = head_dim).

    The resulting streams have shapes:
        Q : (seq_len, num_q_heads, 1, head_dim)   → (64,16,1,32)
        K : (seq_len, num_kv_heads, 1, head_dim) → (64,4,1,32)
        V : (seq_len, num_kv_heads, 1, head_dim) → (64,4,1,32)

    These stream shapes correspond to the required vanilla shapes
    (64,16,32), (64,4,32), (64,4,32) after flattening the tile row.
    """
    # ------------------------------------------------------------------
    # Common dimensions
    # ------------------------------------------------------------------
    head_dim = cos.shape[-1]                # e.g. 32
    seq_len = normed.shape[0]               # e.g. 64
    hidden = normed.shape[-1]               # e.g. 512

    # ------------------------------------------------------------------
    # Q projection
    # ------------------------------------------------------------------
    q_weight_raw = offchip_load(
        q_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=hidden,
        tile_col=hidden,
    )
    q_weight = flatten(q_weight_raw, min_rank=0, max_rank=1)  # (seq_len, hidden, hidden)

    Q_raw = binary_matmul(normed, q_weight)                  # (seq_len, 1, hidden)
    Q = retile_streamify(Q_raw, chunk=head_dim, split_row=False)  # split hidden → heads

    # ------------------------------------------------------------------
    # K projection
    # ------------------------------------------------------------------
    k_weight_raw = offchip_load(
        k_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=hidden,
        tile_col=k_proj.shape[1],
    )
    k_weight = flatten(k_weight_raw, min_rank=0, max_rank=1)  # (seq_len, hidden, hidden_kv)

    K_raw = binary_matmul(normed, k_weight)                  # (seq_len, 1, hidden_kv)
    K = retile_streamify(K_raw, chunk=head_dim, split_row=False)

    # ------------------------------------------------------------------
    # V projection
    # ------------------------------------------------------------------
    v_weight_raw = offchip_load(
        v_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=hidden,
        tile_col=v_proj.shape[1],
    )
    v_weight = flatten(v_weight_raw, min_rank=0, max_rank=1)  # (seq_len, hidden, hidden_kv)

    V_raw = binary_matmul(normed, v_weight)                  # (seq_len, 1, hidden_kv)
    V = retile_streamify(V_raw, chunk=head_dim, split_row=False)

    return Q, K, V