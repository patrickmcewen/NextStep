# The Q/K/V projection performs three independent linear maps on the
# on‑chip tensor ``normed`` (shape (seq_len, 1, hidden_dim)).  The weight
# matrices are off‑chip RAW tensors, so they must be streamed onto‑chip
# first.  We broadcast each weight across the sequence dimension using
# ``offchip_load_ref`` (the reference tensor supplies the desired stream
# shape).  After the dense matmul we have a tile of shape (1, out_dim).
# The output dimension is ``num_heads * head_dim``; we split the column
# dimension into ``head_dim``‑sized chunks, turning the head‑index into a
# stream dimension (``retile_streamify``).  ``reshape_stream`` then
# separates this stream into the original sequence and head dimensions.
# Finally ``accum_retile_row`` merges the head stream back into the tile
# rows, yielding a tile of shape (num_heads, head_dim) with a single
# outer stream dimension (seq_len).  This produces the required vanilla
# shapes:
#   Q : (seq_len, num_heads, head_dim)
#   K : (seq_len, num_kv_heads, head_dim)
#   V : (seq_len, num_kv_heads, head_dim)
def qkv_projection(normed, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # Basic dimensions (all scalars, allowed in Python code)
    # -----------------------------------------------------------------
    head_dim = cos.shape[-1]                     # e.g. 32
    hidden_dim = normed.shape[-1]                # e.g. 512

    # -----------------------------------------------------------------
    # Load projection weights and broadcast over the sequence dimension
    # -----------------------------------------------------------------
    # The reference stream is ``normed`` (shape (seq_len, 1, hidden_dim)).
    # ``offchip_load_ref`` expands the weight to shape (seq_len, hidden, out_dim).
    q_weight = offchip_load_ref(
        normed, q_proj, [], [], hidden_dim, q_proj.shape[1]
    )
    k_weight = offchip_load_ref(
        normed, k_proj, [], [], hidden_dim, k_proj.shape[1]
    )
    v_weight = offchip_load_ref(
        normed, v_proj, [], [], hidden_dim, v_proj.shape[1]
    )

    # -----------------------------------------------------------------
    # Linear projections (dense matmuls)
    # -----------------------------------------------------------------
    Q_dense = binary_matmul(normed, q_weight)   # (seq_len, 1, hidden * num_heads)
    K_dense = binary_matmul(normed, k_weight)   # (seq_len, 1, hidden * num_kv_heads)
    V_dense = binary_matmul(normed, v_weight)   # (seq_len, 1, hidden * num_kv_heads)

    # -----------------------------------------------------------------
    # Convert to multi‑head layout:
    #   1. Split the column dimension into ``head_dim``‑sized chunks
    #      (heads become a new stream dimension).
    #   2. Reshape that stream dimension back into (seq_len, num_heads).
    #   3. Merge the head stream into the tile rows.
    # -----------------------------------------------------------------
    # Q: num_heads = q_proj.shape[1] // head_dim
    num_heads = q_proj.shape[1] // head_dim
    Q = retile_streamify(Q_dense, chunk=head_dim, split_row=False)
    Q = reshape_stream(Q, chunk_size=num_heads, rank=0)
    Q = accum_retile_row(Q, rank=1)

    # K/V: num_kv_heads = k_proj.shape[1] // head_dim
    num_kv_heads = k_proj.shape[1] // head_dim
    K = retile_streamify(K_dense, chunk=head_dim, split_row=False)
    K = reshape_stream(K, chunk_size=num_kv_heads, rank=0)
    K = accum_retile_row(K, rank=1)

    V = retile_streamify(V_dense, chunk=head_dim, split_row=False)
    V = reshape_stream(V, chunk_size=num_kv_heads, rank=0)
    V = accum_retile_row(V, rank=1)

    # -----------------------------------------------------------------
    # Return the three projected tensors.
    # -----------------------------------------------------------------
    return Q, K, V