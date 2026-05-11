def qkv_projection(normed, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    """
    Compute Q, K, V from the normalized input and projection matrices.

    * `normed`  : (seq_len, 1, hidden)   – on‑chip stream (seq_len = 64, hidden = 512)
    * `q_proj`  : RAW weight (hidden, hidden)          – 512 × 512
    * `k_proj` / `v_proj` : RAW weights (hidden, hidden/kv) – 512 × 128
    * `cos`     : provides `head_dim = cos.shape[-1]` (32)

    The algorithm follows the STeP DSL constraints:

    1. Load each RAW weight with a single tile using ``offchip_load``.
       The tile size equals the whole matrix (e.g. 512×512).  The stream
       dimensions of the loaded weight are (1, 1); they are expanded to
       match the token stream of ``normed`` with ``expand_ref`` (expand_rank=2).

    2. Perform a batched matrix multiplication with ``binary_matmul``.
       This yields Q/K/V in tile shape (1, hidden) for each token.

    3. Split the hidden dimension (the tile‑column) into heads using
       ``retile_streamify`` (``split_col=True``) with ``chunk=head_dim``.
       The operation merges the token and head axes into a single stream
       dimension of size ``seq_len * num_heads``.

    4. Restore the two separate stream dimensions (token, head) via
       ``reshape_stream`` where ``chunk_size`` = number of heads.
       The final streamed shape is (seq_len, num_heads, 1, head_dim), which
       corresponds to the required vanilla shape (seq_len, num_heads, head_dim).

    The same steps are applied to Q, K, and V, with the appropriate
    number of heads derived from the weight shapes.
    """
    # ------------------------------------------------------------------ #
    # 1. Common dimensions
    # ------------------------------------------------------------------ #
    head_dim = cos.shape[-1]                 # 32
    hidden_dim = normed.shape[-1]            # 512
    seq_len = normed.shape[0]                # 64

    # Number of heads for Q and for K/V (KV heads)
    num_q_heads = q_proj.shape[1] // head_dim          # 512 // 32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim         # 128 // 32 = 4

    # ------------------------------------------------------------------ #
    # 2. Load and broadcast Q projection weight
    # ------------------------------------------------------------------ #
    q_weight_raw = offchip_load(
        q_proj,
        stride=[0],
        out_shape_tiled=[1],
        tile_row=hidden_dim,
        tile_col=hidden_dim,
    )
    q_weight = expand_ref(q_weight_raw, normed, expand_rank=2)

    # Q = normed @ q_weight   -> (seq_len, 1, hidden)
    Q_raw = binary_matmul(normed, q_weight)

    # Split hidden (tile‑col) into heads and reshape back to (seq_len, heads)
    Q_retiled = retile_streamify(Q_raw, chunk=head_dim, split_col=True)
    Q = reshape_stream(Q_retiled, chunk_size=num_q_heads, rank=0)

    # ------------------------------------------------------------------ #
    # 3. Load and broadcast K projection weight
    # ------------------------------------------------------------------ #
    k_weight_raw = offchip_load(
        k_proj,
        stride=[0],
        out_shape_tiled=[1],
        tile_row=hidden_dim,
        tile_col=k_proj.shape[1],
    )
    k_weight = expand_ref(k_weight_raw, normed, expand_rank=2)

    # K = normed @ k_weight   -> (seq_len, 1, hidden_kv)
    K_raw = binary_matmul(normed, k_weight)

    K_retiled = retile_streamify(K_raw, chunk=head_dim, split_col=True)
    K = reshape_stream(K_retiled, chunk_size=num_kv_heads, rank=0)

    # ------------------------------------------------------------------ #
    # 4. Load and broadcast V projection weight
    # ------------------------------------------------------------------ #
    v_weight_raw = offchip_load(
        v_proj,
        stride=[0],
        out_shape_tiled=[1],
        tile_row=hidden_dim,
        tile_col=v_proj.shape[1],
    )
    v_weight = expand_ref(v_weight_raw, normed, expand_rank=2)

    V_raw = binary_matmul(normed, v_weight)

    V_retiled = retile_streamify(V_raw, chunk=head_dim, split_col=True)
    V = reshape_stream(V_retiled, chunk_size=num_kv_heads, rank=0)

    # ------------------------------------------------------------------ #
    # 5. Return the three streams
    # ------------------------------------------------------------------ #
    return Q, K, V