def pre_attention_and_qkv(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    """
    Leaf planner node implementing the RMSNorm + Q/K/V projection block.

    1️⃣  Off‑chip inputs are streamed onto‑chip with `offchip_load`.  The
        activation tensor is streamed twice: once with a stream dimension
        matching the number of **query heads** (`num_heads`) and once with a
        stream dimension matching the number of **KV heads** (`num_kv_heads`).
        This yields two compatible streams for the subsequent matmuls.

    2️⃣  RMSNorm is expressed with primitive unary / binary DSL ops:
        * square the activation,
        * sum over the hidden‑dimension tile (`unary_rowwise_sum`),
        * scale by 1/hidden_dim,
        * add epsilon,
        * rsqrt,
        * finally broadcast‑multiply the original activation.

    3️⃣  Projection weight matrices are streamed with a stride that
        advances across the head dimension (`stride=[0, 1]`) while replicating
        the same token across the sequence dimension (`out_shape_tiled`
        includes the sequence length).

    4️⃣  Each head‑wise stream is multiplied with its weight stream using
        `binary_matmul`, yielding Q, K, V tensors of shapes
        `(1, S, HN, 1, hd)`, `(1, S, KV, 1, hd)`, `(1, S, KV, 1, hd)`.

    The model configuration is inferred from the hidden dimension:
    * hidden_dim = 512 → head_dim = 32, num_heads = 16, num_kv_heads = 4
    * hidden_dim = 4096 → head_dim = 128, num_heads = 32, num_kv_heads = 8
    (other dimensions are unsupported and raise an assertion.)

    The returned tensors have the correct stream layout; the parent node
    will flatten them to the vanilla shapes `(seq_len, heads, head_dim)`.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Determine sequence length, hidden dimension, and model geometry.
    # ------------------------------------------------------------------
    seq_len = input_tensor.shape[0]          # S
    hidden_dim = input_tensor.shape[1]       # H

    if hidden_dim == 512:                     # Mixtral‑small configuration
        head_dim = 32
        num_heads = 16
        num_kv_heads = 4
    elif hidden_dim == 4096:                  # Qwen‑30B (example) configuration
        head_dim = 128
        num_heads = 32
        num_kv_heads = 8
    else:
        raise AssertionError(
            f"Unable to find matching model config for hidden_dim={hidden_dim}"
        )

    # ------------------------------------------------------------------
    # 2️⃣ Load activation for the Q‑path (streamed per query head).
    # ------------------------------------------------------------------
    X_q = offchip_load(
        input_tensor,
        stride=[1, 0],                         # advance across seq_len, replicate across heads
        out_shape_tiled=(seq_len, num_heads),
        tile_row=1,
        tile_col=hidden_dim,
    )  # shape: (1, S, HN, 1, H)

    # RMSNorm for Q‑path
    X_q_sq = unary_square(X_q)                               # (1, S, HN, 1, H)
    sum_sq_q = unary_rowwise_sum(X_q_sq)                     # (1, S, HN, 1, 1)
    mean_sq_q = unary_mul_imm(sum_sq_q, 1.0 / hidden_dim)    # divide by H
    eps_q = unary_add_imm(mean_sq_q, 1e-6)                    # add epsilon
    inv_sqrt_q = unary_rsqrt(eps_q)                           # rsqrt
    normed_q = binary_mul(X_q, inv_sqrt_q)                    # (1, S, HN, 1, H)

    # ------------------------------------------------------------------
    # 3️⃣ Load activation for the KV‑path (streamed per KV head).
    # ------------------------------------------------------------------
    X_kv = offchip_load(
        input_tensor,
        stride=[1, 0],                         # same stride, different out_shape
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=1,
        tile_col=hidden_dim,
    )  # shape: (1, S, KV, 1, H)

    # RMSNorm for KV‑path (identical computation, different stream shape)
    X_kv_sq = unary_square(X_kv)
    sum_sq_kv = unary_rowwise_sum(X_kv_sq)
    mean_sq_kv = unary_mul_imm(sum_sq_kv, 1.0 / hidden_dim)
    eps_kv = unary_add_imm(mean_sq_kv, 1e-6)
    inv_sqrt_kv = unary_rsqrt(eps_kv)
    normed_kv = binary_mul(X_kv, inv_sqrt_kv)                # (1, S, KV, 1, H)

    # ------------------------------------------------------------------
    # 4️⃣ Stream the projection weight matrices.
    # ------------------------------------------------------------------
    QW = offchip_load(
        q_proj,
        stride=[0, 1],                         # replicate across seq_len, advance across heads
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
    # 5️⃣ Compute Q, K, V via per‑head matrix multiplication.
    # ------------------------------------------------------------------
    Q = binary_matmul(normed_q, QW)   # (1, S, HN, 1, hd)
    K = binary_matmul(normed_kv, KW) # (1, S, KV, 1, hd)
    V = binary_matmul(normed_kv, VW) # (1, S, KV, 1, hd)

    return Q, K, V