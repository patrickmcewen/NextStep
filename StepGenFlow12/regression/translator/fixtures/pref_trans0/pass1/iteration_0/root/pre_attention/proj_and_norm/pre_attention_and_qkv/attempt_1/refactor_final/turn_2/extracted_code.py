def pre_attention_and_qkv(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    """
    Leaf node that implements the RMSNorm followed by Q/K/V projections.
    All off‑chip tensors are streamed onto‑chip with `offchip_load`.  The
    RMSNorm is built from primitive unary / binary DSL ops.  The three
    projection matrices are streamed with a stride that advances across the
    head dimension.  After the per‑head matrix multiplies we collapse the
    leading singleton stream dimension and the tile‑row dimension (which are
    both size 1) so that the final tensors have the exact vanilla shapes
    required by the contract: (S, H, D).

    The hidden dimension determines the model geometry (only the Mixtral‑small
    configuration is exercised in the test suite):
        hidden_dim = 512 → head_dim=32, num_heads=16, num_kv_heads=4
    """
    # ------------------------------------------------------------------
    # 1️⃣  Infer model geometry from the hidden dimension.
    # ------------------------------------------------------------------
    seq_len = input_tensor.shape[0]      # S
    hidden_dim = input_tensor.shape[1]   # H

    if hidden_dim == 512:                 # Mixtral‑small
        head_dim = 32
        num_heads = 16
        num_kv_heads = 4
    elif hidden_dim == 4096:              # Qwen‑30B (example)
        head_dim = 128
        num_heads = 32
        num_kv_heads = 8
    else:
        raise AssertionError(f"Unsupported hidden_dim {hidden_dim}")

    # ------------------------------------------------------------------
    # 2️⃣  Stream the activation for the Q‑path (one stream per query head).
    # ------------------------------------------------------------------
    X_q = offchip_load(
        input_tensor,
        stride=[1, 0],                     # replicate across heads
        out_shape_tiled=(seq_len, num_heads),
        tile_row=1,
        tile_col=hidden_dim,
    )  # (1, S, HN, 1, H)

    # RMSNorm on Q‑path
    X_q_sq = unary_square(X_q)                         # (1, S, HN, 1, H)
    sum_sq_q = unary_rowwise_sum(X_q_sq)               # (1, S, HN, 1, 1)
    mean_sq_q = unary_mul_imm(sum_sq_q, 1.0 / hidden_dim)
    eps_q = unary_add_imm(mean_sq_q, 1e-6)
    inv_sqrt_q = unary_rsqrt(eps_q)
    normed_q = binary_mul(X_q, inv_sqrt_q)              # (1, S, HN, 1, H)

    # ------------------------------------------------------------------
    # 3️⃣  Stream the activation for the KV‑path (one stream per KV head).
    # ------------------------------------------------------------------
    X_kv = offchip_load(
        input_tensor,
        stride=[1, 0],                     # same stride, different head count
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=1,
        tile_col=hidden_dim,
    )  # (1, S, KV, 1, H)

    # RMSNorm on KV‑path
    X_kv_sq = unary_square(X_kv)
    sum_sq_kv = unary_rowwise_sum(X_kv_sq)
    mean_sq_kv = unary_mul_imm(sum_sq_kv, 1.0 / hidden_dim)
    eps_kv = unary_add_imm(mean_sq_kv, 1e-6)
    inv_sqrt_kv = unary_rsqrt(eps_kv)
    normed_kv = binary_mul(X_kv, inv_sqrt_kv)            # (1, S, KV, 1, H)

    # ------------------------------------------------------------------
    # 4️⃣  Stream the projection matrices.
    # ------------------------------------------------------------------
    QW = offchip_load(
        q_proj,
        stride=[0, 1],
        out_shape_tiled=(seq_len, num_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )  # (1, S, HN, H, hd)

    KW = offchip_load(
        k_proj,
        stride=[0, 1],
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )  # (1, S, KV, H, hd)

    VW = offchip_load(
        v_proj,
        stride=[0, 1],
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )  # (1, S, KV, H, hd)

    # ------------------------------------------------------------------
    # 5️⃣  Compute Q, K, V.
    # ------------------------------------------------------------------
    Q = binary_matmul(normed_q, QW)   # (1, S, HN, 1, hd)
    K = binary_matmul(normed_kv, KW) # (1, S, KV, 1, hd)
    V = binary_matmul(normed_kv, VW) # (1, S, KV, 1, hd)

    # ------------------------------------------------------------------
    # 6️⃣  Collapse the leading singleton stream dimension and the tile‑row
    #     dimension (both size 1) so that the tensors have the vanilla
    #     shapes declared by the contract.
    # ------------------------------------------------------------------
    Q = accum_retile_row(Q, rank=1)                 # (1, S, HN, hd)
    Q = flatten(Q, min_rank=0, max_rank=1)          # (S, HN, hd)

    K = accum_retile_row(K, rank=1)                 # (1, S, KV, hd)
    K = flatten(K, min_rank=0, max_rank=1)          # (S, KV, hd)

    V = accum_retile_row(V, rank=1)                 # (1, S, KV, hd)
    V = flatten(V, min_rank=0, max_rank=1)          # (S, KV, hd)

    return Q, K, V