def qkv_projection(normed, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    """
    Compute Q, K, V streams from the normalized input and the three projection
    matrices.

    * `normed` – on‑chip token stream of shape (seq_len, 1, hidden).
    * `q_proj`, `k_proj`, `v_proj` – RAW weight matrices (off‑chip).
    * `cos` – provides the head dimension (`head_dim = cos.shape[-1]`).

    The DSL steps are:

    1. Load each RAW weight with `offchip_load`.  The weight is tiled once
       (tile_row = hidden, tile_col = weight_out_dim) and streamed over the
       token dimension (`out_shape_tiled=[seq_len]`).  This yields a stream
       shape (1, seq_len) which we collapse to (seq_len,) with `flatten`,
       giving a weight stream whose stream dimension matches that of `normed`.

    2. Perform a batched matrix multiplication (`binary_matmul`) between the
       token stream and the weight stream, producing a stream of shape
       (seq_len, 1, hidden_out).

    3. Split the hidden output dimension into heads with `retile_streamify`
       (column split, `chunk=head_dim`).  This produces a combined stream
       dimension of size `seq_len * num_heads` and tile shape (1, head_dim).

    4. Restore the separate token and head stream dimensions using
       `reshape_stream` with `chunk_size=num_heads` (or `num_kv_heads`).

    The final streams have shapes:
        Q : (seq_len, num_q_heads, 1, head_dim)
        K : (seq_len, num_kv_heads, 1, head_dim)
        V : (seq_len, num_kv_heads, 1, head_dim)

    which correspond to the required vanilla shapes
    (seq_len, num_heads, head_dim).
    """
    # ------------------------------------------------------------------
    # Common dimensions
    # ------------------------------------------------------------------
    head_dim = cos.shape[-1]          # e.g. 32
    seq_len = normed.shape[0]         # e.g. 64
    hidden = normed.shape[-1]         # e.g. 512

    # Number of heads for each projection
    num_q_heads = q_proj.shape[1] // head_dim      # 512 // 32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim     # 128 // 32 = 4

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
    q_weight = flatten(q_weight_raw, min_rank=0, max_rank=1)      # (seq_len, hidden, hidden)

    Q_raw = binary_matmul(normed, q_weight)                      # (seq_len, 1, hidden)
    Q_retiled = retile_streamify(Q_raw, chunk=head_dim, split_row=False)
    Q = reshape_stream(Q_retiled, chunk_size=num_q_heads, rank=0)  # (seq_len, num_q_heads, 1, head_dim)

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
    k_weight = flatten(k_weight_raw, min_rank=0, max_rank=1)      # (seq_len, hidden, hidden_kv)

    K_raw = binary_matmul(normed, k_weight)                      # (seq_len, 1, hidden_kv)
    K_retiled = retile_streamify(K_raw, chunk=head_dim, split_row=False)
    K = reshape_stream(K_retiled, chunk_size=num_kv_heads, rank=0)  # (seq_len, num_kv_heads, 1, head_dim)

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
    v_weight = flatten(v_weight_raw, min_rank=0, max_rank=1)      # (seq_len, hidden, hidden_kv)

    V_raw = binary_matmul(normed, v_weight)                      # (seq_len, 1, hidden_kv)
    V_retiled = retile_streamify(V_raw, chunk=head_dim, split_row=False)
    V = reshape_stream(V_retiled, chunk_size=num_kv_heads, rank=0)  # (seq_len, num_kv_heads, 1, head_dim)

    return Q, K, V