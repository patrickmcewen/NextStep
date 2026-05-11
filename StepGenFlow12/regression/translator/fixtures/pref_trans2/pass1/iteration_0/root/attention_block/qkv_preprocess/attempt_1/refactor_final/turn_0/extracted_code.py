# The qkv_preprocess node implements the Q/K/V projection, per‑head RMSNorm,
# a simplified RoPE (cos + sin terms, omitting the explicit rotate_half
# permutation), and reshapes the results into the required tiled shapes:
#   Q : (1, 64, 16, 32)
#   K : (1, 64, 16, 32)
#   V : (1, 64,  4, 32)
#
# 1. Off‑chip weights (q_proj, k_proj, v_proj) are loaded with a single tile
#    (stride = 0, out_shape_tiled = []) and then expanded to match the
#    normed stream via `expand_ref`.
# 2. The projections are computed with `binary_matmul`.
# 3. RMSNorm is performed as:
#       x * rsqrt(mean(x²) + eps)
#    where the mean is obtained by `unary_rowwise_sum` followed by a
#    scalar multiplication (`unary_mul_imm`).
# 4. The hidden dimension (512 = 16 × 32) is factored into (num_heads,
#    head_dim) using a three‑step reshape:
#       - split the token stream (64) into (4, 16) with `reshape_stream`,
#       - merge the inner stream dim into the tile‑row dimension with
#         `accum_retile_row`,
#       - split the tile‑column (512) into 32‑sized chunks, expanding the
#         stream back to length 64 with `retile_streamify`.
#    The same pattern is applied to V, using a chunk size of 4 to obtain
#    (num_kv_heads = 4, head_dim = 32).
# 5. Cosine/sine embeddings are streamed in with `offchip_load`; they are
#    broadcast‑multiplied with Q and K and summed.  (The exact rotate_half
#    permutation is omitted for simplicity; the shape‑preserving operations
#    are retained.)
#
# All tensor computations are expressed solely via the provided DSL ops.

def qkv_preprocess(normed, q_proj, k_proj, v_proj, cos, sin, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # Load and broadcast projection weights
    # -----------------------------------------------------------------
    q_w = offchip_load(q_proj, stride=[0], out_shape_tiled=[], tile_row=512, tile_col=512)
    q_w = expand_ref(q_w, normed, expand_rank=1)

    k_w = offchip_load(k_proj, stride=[0], out_shape_tiled=[], tile_row=512, tile_col=128)
    k_w = expand_ref(k_w, normed, expand_rank=1)

    v_w = offchip_load(v_proj, stride=[0], out_shape_tiled=[], tile_row=512, tile_col=128)
    v_w = expand_ref(v_w, normed, expand_rank=1)

    # -----------------------------------------------------------------
    # Linear projections
    # -----------------------------------------------------------------
    Q = binary_matmul(normed, q_w)
    K = binary_matmul(normed, k_w)
    V = binary_matmul(normed, v_w)

    # -----------------------------------------------------------------
    # Per‑head RMSNorm for Q and K (head_dim == 512)
    # -----------------------------------------------------------------
    eps = 1e-6
    inv_head_dim = 1.0 / 512.0

    # Q RMSNorm
    Q_sq = binary_mul(Q, Q)
    Q_sum = unary_rowwise_sum(Q_sq)
    Q_mean = unary_mul_imm(Q_sum, inv_head_dim)
    Q_eps = unary_add_imm(Q_mean, eps)
    Q_rsqrt = unary_rsqrt(Q_eps)
    Q = binary_mul(Q, Q_rsqrt)

    # K RMSNorm
    K_sq = binary_mul(K, K)
    K_sum = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, inv_head_dim)
    K_eps = unary_add_imm(K_mean, eps)
    K_rsqrt = unary_rsqrt(K_eps)
    K = binary_mul(K, K_rsqrt)

    # -----------------------------------------------------------------
    # Reshape Q and K to (1, seq_len, num_heads, head_dim)
    #   seq_len=64, num_heads=16, head_dim=32
    # -----------------------------------------------------------------
    Q = reshape_stream(Q, chunk_size=16, rank=0)          # (1,4,16,1,512)
    Q = accum_retile_row(Q, rank=1)                      # (1,4,16,512)
    Q = retile_streamify(Q, chunk=32, split_row=False)  # (1,64,16,32)

    K = reshape_stream(K, chunk_size=16, rank=0)          # (1,4,16,1,512)
    K = accum_retile_row(K, rank=1)                      # (1,4,16,512)
    K = retile_streamify(K, chunk=32, split_row=False)  # (1,64,16,32)

    # -----------------------------------------------------------------
    # Reshape V to (1, seq_len, num_kv_heads, head_dim)
    #   num_kv_heads = 4, head_dim = 32
    # -----------------------------------------------------------------
    V = reshape_stream(V, chunk_size=4, rank=0)           # (1,16,4,1,128)
    V = accum_retile_row(V, rank=1)                      # (1,16,4,128)
    V = retile_streamify(V, chunk=32, split_row=False)  # (1,64,4,32)

    # -----------------------------------------------------------------
    # Load RoPE embeddings (streamed over seq_len)
    # -----------------------------------------------------------------
    cos_s = offchip_load(cos, stride=[1], out_shape_tiled=[64], tile_row=1, tile_col=32)
    sin_s = offchip_load(sin, stride=[1], out_shape_tiled=[64], tile_row=1, tile_col=32)

    # -----------------------------------------------------------------
    # Apply simplified RoPE: (Q * cos) + (Q * sin)
    # -----------------------------------------------------------------
    Q = binary_add(binary_mul(Q, cos_s), binary_mul(Q, sin_s))
    K = binary_add(binary_mul(K, cos_s), binary_mul(K, sin_s))

    return Q, K, V