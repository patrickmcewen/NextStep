# The implementation loads off‑chip tensors, projects `normed` into Q/K/V,
# splits the hidden dimension into per‑head tiles, applies a per‑head RMSNorm,
# and finally adds Rotary‑Position‑Embedding (RoPE).  All heavy work is done with
# DSL primitives; only `torch.cat` (a pure shape operation) is used to implement
# the `rotate_half` helper.
def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------------
    # Extract scalar dimensions needed for tiling.
    # ------------------------------------------------------------------------
    seq_len   = normed.shape[1]                # 64
    head_dim  = cos.shape[-1]                 # 32
    num_heads = q_proj.shape[1] // head_dim   # 512 // 32 = 16
    num_kv_heads = k_proj.shape[1] // head_dim   # 128 // 32 = 4
    eps = 1e-6

    # ------------------------------------------------------------------------
    # 1️⃣ Load RAW off‑chip tensors.
    #    - Weight matrices are broadcast across the sequence dimension.
    #    - Cosine / sine embeddings are streamed over (seq_len, num_heads) or
    #      (seq_len, num_kv_heads).  The stride `[1, 0]` makes the tensor vary
    #      with the sequence index but stay constant across the head index.
    # ------------------------------------------------------------------------
    q_proj_s = offchip_load(
        q_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=512,
        tile_col=512,
    )
    k_proj_s = offchip_load(
        k_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=512,
        tile_col=128,
    )
    v_proj_s = offchip_load(
        v_proj,
        stride=[0],
        out_shape_tiled=[seq_len],
        tile_row=512,
        tile_col=128,
    )
    # Cos / sin for the query heads
    cos_q = offchip_load(
        cos,
        stride=[1, 0],
        out_shape_tiled=[seq_len, num_heads],
        tile_row=1,
        tile_col=head_dim,
    )
    sin_q = offchip_load(
        sin,
        stride=[1, 0],
        out_shape_tiled=[seq_len, num_heads],
        tile_row=1,
        tile_col=head_dim,
    )
    # Cos / sin for the KV heads
    cos_k = offchip_load(
        cos,
        stride=[1, 0],
        out_shape_tiled=[seq_len, num_kv_heads],
        tile_row=1,
        tile_col=head_dim,
    )
    sin_k = offchip_load(
        sin,
        stride=[1, 0],
        out_shape_tiled=[seq_len, num_kv_heads],
        tile_row=1,
        tile_col=head_dim,
    )

    # ------------------------------------------------------------------------
    # 2️⃣ Linear projections (batched matmul).
    # ------------------------------------------------------------------------
    Q_raw = binary_matmul(normed, q_proj_s)   # (1, 64, 1, 512)
    K_raw = binary_matmul(normed, k_proj_s)   # (1, 64, 1, 128)
    V_raw = binary_matmul(normed, v_proj_s)   # (1, 64, 1, 128)

    # ------------------------------------------------------------------------
    # 3️⃣ Split the hidden dimension into per‑head tiles:
    #     - retile_streamify breaks the tile‑column into chunks of size `head_dim`,
    #       producing a combined stream dimension of size seq_len * num_heads.
    #     - reshape_stream separates that combined dimension back into
    #       (seq_len, num_heads).
    #     - accum_retile_row merges the (head) stream dimension into the tile‑row
    #       dimension, yielding the final (seq_len, num_heads, head_dim) layout.
    # ------------------------------------------------------------------------
    # Q
    Q = retile_streamify(Q_raw, chunk=head_dim, split_row=False)          # (1, 1024, 1, 32)
    Q = reshape_stream(Q, chunk_size=num_heads, rank=0)                   # (1, 64, 16, 1, 32)
    Q = accum_retile_row(Q, rank=1)                                      # (1, 64, 16, 32)

    # K
    K = retile_streamify(K_raw, chunk=head_dim, split_row=False)          # (1, 256, 1, 32)
    K = reshape_stream(K, chunk_size=num_kv_heads, rank=0)                # (1, 64, 4, 1, 32)
    K = accum_retile_row(K, rank=1)                                      # (1, 64, 4, 32)

    # V (same head layout as K)
    V = retile_streamify(V_raw, chunk=head_dim, split_row=False)          # (1, 256, 1, 32)
    V = reshape_stream(V, chunk_size=num_kv_heads, rank=0)                # (1, 64, 4, 1, 32)
    V = accum_retile_row(V, rank=1)                                      # (1, 64, 4, 32)

    # ------------------------------------------------------------------------
    # 4️⃣ Per‑head RMSNorm (applied to Q and K only).
    # ------------------------------------------------------------------------
    def rms_norm(t):
        sq      = unary_square(t)                                   # (…, H, D)
        sum_sq  = unary_rowwise_sum(sq)                             # (…, H, 1)
        mean    = unary_mul_imm(sum_sq, constant=1.0 / head_dim)    # (…, H, 1)
        add_eps = unary_add_imm(mean, constant=eps)                # (…, H, 1)
        rsqrt   = unary_rsqrt(add_eps)                              # (…, H, 1)
        return binary_mul(t, rsqrt)                                 # broadcast over D

    Q = rms_norm(Q)
    K = rms_norm(K)
    # V is not RMS‑normalized.

    # ------------------------------------------------------------------------
    # 5️⃣ Prepare RoPE embeddings: merge the head‑stream dimension into tile rows.
    # ------------------------------------------------------------------------
    cos_q = accum_retile_row(cos_q, rank=1)   # (1, 64, 16, 32)
    sin_q = accum_retile_row(sin_q, rank=1)   # (1, 64, 16, 32)
    cos_k = accum_retile_row(cos_k, rank=1)   # (1, 64,  4, 32)
    sin_k = accum_retile_row(sin_k, rank=1)   # (1, 64,  4, 32)

    # ------------------------------------------------------------------------
    # 6️⃣ Helper for RoPE: rotate the last dimension and negate the second half.
    # ------------------------------------------------------------------------
    def rotate_half(x):
        half = x.shape[-1] // 2
        first = x[..., :half]
        second = x[..., half:]
        neg_second = unary_mul_imm(second, constant=-1.0)
        return torch.cat([neg_second, first], dim=-1)

    # Apply RoPE to Q and K.
    Q = binary_add(
        binary_mul(Q, cos_q),
        binary_mul(rotate_half(Q), sin_q),
    )
    K = binary_add(
        binary_mul(K, cos_k),
        binary_mul(rotate_half(K), sin_k),
    )
    # V remains unchanged.

    # ------------------------------------------------------------------------
    # 7️⃣ Return the three tensors in the order required by the contract.
    # ------------------------------------------------------------------------
    return Q, K, V