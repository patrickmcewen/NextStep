def qkv_projection(normed, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Pure‑Python dimension extraction (scalar math – allowed)
    # ----------------------------------------------------------------------
    seq_len    = normed.shape[0]          # 64
    hidden_dim = normed.shape[-1]         # 512
    head_dim   = cos.shape[-1]            # 32

    # Number of heads for Q and for KV (derived from weight shapes)
    num_q_heads  = q_proj.shape[1] // head_dim   # 512//32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim   # 128//32 = 4

    # ----------------------------------------------------------------------
    # Put the on‑chip 'normed' tensor into a 2‑D stream (adds a trailing
    # stream dimension of size 1).  Result: stream shape (seq_len, 1),
    # tile shape (1, hidden_dim).
    # ----------------------------------------------------------------------
    normed_stream = reshape_stream(normed, chunk_size=1, rank=0)   # (S,1,1,H)

    # ----------------------------------------------------------------------
    # Load the three projection matrices from off‑chip and reshape them
    # into streams of shape (seq_len, num_heads) with tile shape
    # (hidden_dim, head_dim).  The stride (0, 1) broadcasts each head’s
    # tile across the entire sequence dimension.
    # ----------------------------------------------------------------------
    # Q‑projection
    q_weight_raw = offchip_load(
        q_proj,
        stride=(0, 1),
        out_shape_tiled=(seq_len, num_q_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    q_weight = flatten(q_weight_raw, min_rank=1, max_rank=2)      # (S, Hq, hidden, head_dim)

    # K‑projection
    k_weight_raw = offchip_load(
        k_proj,
        stride=(0, 1),
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    k_weight = flatten(k_weight_raw, min_rank=1, max_rank=2)      # (S, Hkv, hidden, head_dim)

    # V‑projection
    v_weight_raw = offchip_load(
        v_proj,
        stride=(0, 1),
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    v_weight = flatten(v_weight_raw, min_rank=1, max_rank=2)      # (S, Hkv, hidden, head_dim)

    # ----------------------------------------------------------------------
    # Broadcast the normed tensor across the head dimension so that its
    # stream shape matches the weight streams.
    # expand_ref adds the extra head‑wise stream dimension (size 1 → num_*)
    # while preserving the tile layout.
    # ----------------------------------------------------------------------
    q_norm = expand_ref(normed_stream, q_weight, expand_rank=1)   # (S, Hq, 1, hidden)
    k_norm = expand_ref(normed_stream, k_weight, expand_rank=1)  # (S, Hkv, 1, hidden)
    v_norm = expand_ref(normed_stream, v_weight, expand_rank=1)  # (S, Hkv, 1, hidden)

    # ----------------------------------------------------------------------
    # Matrix multiplication (per‑head): (1 × hidden) @ (hidden × head_dim)
    # ----------------------------------------------------------------------
    Q_mat = binary_matmul(q_norm, q_weight)   # (S, Hq, 1, head_dim)
    K_mat = binary_matmul(k_norm, k_weight)   # (S, Hkv, 1, head_dim)
    V_mat = binary_matmul(v_norm, v_weight)   # (S, Hkv, 1, head_dim)

    # ----------------------------------------------------------------------
    # Merge the per‑head stream dimension into the tile‑row dimension,
    # yielding the requested vanilla shapes:
    #   Q: (seq_len, num_q_heads, head_dim)
    #   K, V: (seq_len, num_kv_heads, head_dim)
    # ----------------------------------------------------------------------
    Q = accum_retile_row(Q_mat, rank=1)   # (S,)×tile(num_q_heads, head_dim)
    K = accum_retile_row(K_mat, rank=1)   # (S,)×tile(num_kv_heads, head_dim)
    V = accum_retile_row(V_mat, rank=1)   # (S,)×tile(num_kv_heads, head_dim)

    return Q, K, V