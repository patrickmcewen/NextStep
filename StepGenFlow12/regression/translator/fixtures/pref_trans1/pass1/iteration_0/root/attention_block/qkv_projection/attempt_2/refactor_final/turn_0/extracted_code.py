def qkv_projection(normed, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Compute static dimensions (pure Python – no tensor math)
    # ----------------------------------------------------------------------
    seq_len      = normed.shape[0]           # sequence length (e.g. 64)
    hidden_dim   = normed.shape[-1]          # hidden dimension (e.g. 512)
    head_dim     = cos.shape[-1]             # head dimension (e.g. 32)

    # Number of heads for Q and for KV (derived from weight shapes)
    num_q_heads  = q_proj.shape[1] // head_dim   # 512/32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim   # 128/32 = 4

    # ----------------------------------------------------------------------
    # Prepare the on‑chip 'normed' tensor:
    #   reshape_stream adds a trailing stream dimension of size 1,
    #   turning (S, 1, H) -> (S, 1, 1, H)  (stream shape: (S, 1))
    # ----------------------------------------------------------------------
    normed_1 = reshape_stream(normed, chunk_size=1, rank=0)  # (S,1,1,H)

    # ----------------------------------------------------------------------
    # Helper to load a projection matrix and flatten away the leading 1
    # that offchip_load always inserts.
    #   – stride (0,1) broadcasts across the sequence dimension and indexes
    #     the head dimension.
    #   – out_shape_tiled = (seq_len, num_heads) creates the desired stream.
    #   – tile_row = hidden_dim, tile_col = head_dim.
    #   – flatten merges the leading singleton stream dim with seq_len.
    # ----------------------------------------------------------------------
    def load_and_flatten(proj, num_heads):
        w = offchip_load(
            proj,
            stride=(0, 1),
            out_shape_tiled=(seq_len, num_heads),
            tile_row=hidden_dim,
            tile_col=head_dim,
        )
        # offchip_load returns shape (1, seq_len, num_heads, hidden_dim, head_dim)
        # flatten merges the leading 1 with seq_len → (seq_len, num_heads, hidden_dim, head_dim)
        return flatten(w, min_rank=1, max_rank=2)

    # ----------------------------------------------------------------------
    # Load Q, K, V projection weights
    # ----------------------------------------------------------------------
    q_weight = load_and_flatten(q_proj, num_q_heads)   # (S, Hq, hidden, head_dim)
    k_weight = load_and_flatten(k_proj, num_kv_heads)  # (S, Hkv, hidden, head_dim)
    v_weight = load_and_flatten(v_proj, num_kv_heads)  # (S, Hkv, hidden, head_dim)

    # ----------------------------------------------------------------------
    # Broadcast the normed tensor to match the weight stream shape.
    #   expand_ref expands the trailing stream dimension (size 1) to the
    #   required number of heads.
    # ----------------------------------------------------------------------
    q_norm = expand_ref(normed_1, q_weight, expand_rank=1)   # (S, Hq, 1, hidden_dim)
    k_norm = expand_ref(normed_1, k_weight, expand_rank=1)  # (S, Hkv, 1, hidden_dim)
    v_norm = expand_ref(normed_1, v_weight, expand_rank=1)  # (S, Hkv, 1, hidden_dim)

    # ----------------------------------------------------------------------
    # Matrix multiply: (1 x hidden) @ (hidden x head_dim) -> (1 x head_dim)
    # ----------------------------------------------------------------------
    Q_mat = binary_matmul(q_norm, q_weight)  # (S, Hq, 1, head_dim)
    K_mat = binary_matmul(k_norm, k_weight)  # (S, Hkv, 1, head_dim)
    V_mat = binary_matmul(v_norm, v_weight)  # (S, Hkv, 1, head_dim)

    # ----------------------------------------------------------------------
    # Merge the per‑head stream dimension into the tile rows.
    #   accum_retile_row(rank=1) converts (S, H, 1, D) → (S, H, D)
    # ----------------------------------------------------------------------
    Q = accum_retile_row(Q_mat, rank=1)  # (seq_len, num_q_heads, head_dim)
    K = accum_retile_row(K_mat, rank=1)  # (seq_len, num_kv_heads, head_dim)
    V = accum_retile_row(V_mat, rank=1)  # (seq_len, num_kv_heads, head_dim)

    return Q, K, V