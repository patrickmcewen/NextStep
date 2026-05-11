# The RMSNorm is performed as:
#   normed = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps)
#   – x is loaded from off‑chip memory as a stream of 1×H tiles (H = hidden_dim).
#   – The mean is realized by squaring, a row‑wise sum, adding eps, and a rsqrt.
#   – The result (a scalar per token) is multiplied back to x.
# For the Q/K/V projections we need a stream shape (seq_len, heads, head_dim).
#   – We duplicate the RMS‑normalized tensor across the head dimension with
#     `repeat_static`.
#   – The projection matrices are loaded with a 2‑D streaming shape
#     (seq_len, heads) where the first dimension uses stride 0 to broadcast the
#     same weight across all tokens and the second dimension steps through the
#     per‑head weight tiles (hidden_dim × head_dim).
#   – A batched matmul (`binary_matmul`) yields tiles of shape (1, head_dim).
#   – Finally `flatten` merges the leading singleton stream dim with the
#     sequence‑length dim, producing the required vanilla shapes (seq_len, heads, head_dim).
def pre_attn_norm_and_proj(
    input_tensor, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None
):
    # ----- shapes & derived constants -----
    seq_len = input_tensor.shape[0]          # 64
    hidden_dim = input_tensor.shape[1]       # 512
    head_dim = cos.shape[-1]                 # 32
    num_heads = q_proj.shape[1] // head_dim  # 16
    num_kv_heads = k_proj.shape[1] // head_dim  # 4
    eps = 1e-6

    # ----- load input and RMSNorm -----
    # stream shape (1, seq_len), tile (1, hidden_dim)
    x = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=hidden_dim,
    )
    sq = unary_square(x)                    # x**2
    sum_sq = unary_rowwise_sum(sq)          # sum over hidden_dim -> (1,1) tile
    sum_eps = unary_add_imm(sum_sq, eps)    # add epsilon
    inv_rms = unary_rsqrt(sum_eps)          # 1 / sqrt(mean + eps)
    normed = binary_mul(x, inv_rms)         # RMS‑normalized tensor

    # ----- Q projection -----
    q_normed = repeat_static(normed, num_heads)  # add head dim as stream
    q_weight = offchip_load(
        q_proj,
        stride=(0, 1),                      # broadcast over seq_len, step over heads
        out_shape_tiled=(seq_len, num_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    Q_raw = binary_matmul(q_normed, q_weight)    # (1, seq_len, num_heads, 1, head_dim)
    Q = flatten(Q_raw, min_rank=1, max_rank=2)   # merge leading 1 with seq_len

    # ----- K projection -----
    k_normed = repeat_static(normed, num_kv_heads)
    k_weight = offchip_load(
        k_proj,
        stride=(0, 1),
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    K_raw = binary_matmul(k_normed, k_weight)
    K = flatten(K_raw, min_rank=1, max_rank=2)

    # ----- V projection -----
    v_normed = repeat_static(normed, num_kv_heads)
    v_weight = offchip_load(
        v_proj,
        stride=(0, 1),
        out_shape_tiled=(seq_len, num_kv_heads),
        tile_row=hidden_dim,
        tile_col=head_dim,
    )
    V_raw = binary_matmul(v_normed, v_weight)
    V = flatten(V_raw, min_rank=1, max_rank=2)

    return Q, K, V