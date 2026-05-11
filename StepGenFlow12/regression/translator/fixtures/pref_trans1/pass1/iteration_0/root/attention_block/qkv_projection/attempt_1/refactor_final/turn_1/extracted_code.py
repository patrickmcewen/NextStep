# Q/K/V projection for a multi‑head transformer.
#   * normed   – on‑chip stream: (seq_len, 1, hidden_dim)  (tile rows = 1, tile cols = hidden_dim)
#   * q_proj/k_proj/v_proj – off‑chip weight matrices; we stream them across `seq_len`
#                            with a trivial 1‑tile broadcast using `offchip_load_ref`.
#   * cos is not used here (it will be consumed downstream).
#
# The projection proceeds as:
#   1. Load each weight and broadcast it over the sequence dimension.
#   2. Dense matmul (`binary_matmul`) produces a tile of shape (1, out_dim).
#   3. Split the column tile (out_dim = num_heads * head_dim) into a new stream
#      dimension for the heads (`retile_streamify` on the tile‑column).
#   4. Merge the newly created head‑stream dimension with the (still‑unit) tile‑row
#      using `accum_retile_row`; the result has tile shape (num_heads, head_dim)
#      and a single stream dimension (seq_len), i.e. the vanilla shape
#      (seq_len, num_heads, head_dim) required by the parent.
def qkv_projection(normed, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # --------------------------------------------------------------
    # Scalar dimensions (pure Python integers)
    # --------------------------------------------------------------
    head_dim = cos.shape[-1]                     # e.g. 32
    hidden_dim = normed.shape[-1]                # e.g. 512

    # --------------------------------------------------------------
    # Broadcast the off‑chip weights across the sequence stream.
    # `stride=[0]` means the same tile is reused for every token.
    # `out_shape_tiled=[1]` creates a single tile that will later be
    # expanded to match `normed`'s stream shape (seq_len, 1).
    # --------------------------------------------------------------
    q_weight = offchip_load_ref(
        normed,                         # reference stream (seq_len, 1, hidden_dim)
        q_proj,                         # underlying off‑chip matrix (hidden_dim, out_dim)
        [0],                            # stride – broadcast
        [1],                            # out_shape_tiled – one tile
        hidden_dim,                     # tile rows of the weight
        q_proj.shape[1]                 # tile cols = output dimension (num_heads * head_dim)
    )
    k_weight = offchip_load_ref(
        normed,
        k_proj,
        [0],
        [1],
        hidden_dim,
        k_proj.shape[1]
    )
    v_weight = offchip_load_ref(
        normed,
        v_proj,
        [0],
        [1],
        hidden_dim,
        v_proj.shape[1]
    )

    # --------------------------------------------------------------
    # Dense linear projections.
    # Each result has tile shape (1, out_dim).
    # --------------------------------------------------------------
    Q_dense = binary_matmul(normed, q_weight)   # (seq_len, 1, 1, out_dim)
    K_dense = binary_matmul(normed, k_weight)   # (seq_len, 1, 1, out_dim)
    V_dense = binary_matmul(normed, v_weight)   # (seq_len, 1, 1, out_dim)

    # --------------------------------------------------------------
    # Convert the column tile (out_dim) into a head‑stream dimension.
    # After `retile_streamify` we have shape (seq_len, num_heads, 1, head_dim).
    # `accum_retile_row` merges the unit tile‑row into that head stream,
    # yielding a tile of shape (num_heads, head_dim) with a single stream
    # dimension (seq_len) → vanilla shape (seq_len, num_heads, head_dim).
    # --------------------------------------------------------------
    # Q
    num_heads = q_proj.shape[1] // head_dim
    Q = retile_streamify(Q_dense, chunk=head_dim, split_row=False)   # (S, H, 1, head_dim)
    Q = accum_retile_row(Q, rank=1)                                 # (S, H, head_dim)

    # K
    num_kv_heads = k_proj.shape[1] // head_dim
    K = retile_streamify(K_dense, chunk=head_dim, split_row=False)   # (S, KV_H, 1, head_dim)
    K = accum_retile_row(K, rank=1)                                 # (S, KV_H, head_dim)

    # V
    V = retile_streamify(V_dense, chunk=head_dim, split_row=False)   # (S, KV_H, 1, head_dim)
    V = accum_retile_row(V, rank=1)                                 # (S, KV_H, head_dim)

    return Q, K, V