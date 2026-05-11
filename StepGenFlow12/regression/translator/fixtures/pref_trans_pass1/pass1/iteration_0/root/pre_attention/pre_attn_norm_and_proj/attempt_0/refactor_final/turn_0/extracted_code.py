# Implementation reasoning:
# 1. Load all RAW tensors as single‑tile streams using `offchip_load`.
# 2. Apply RMSNorm: square, row‑wise sum, divide by hidden_dim, add epsilon,
#    compute rsqrt, and broadcast‑multiply with the original input.
# 3. Project the normalized activations to Q, K, V with `binary_matmul`
#    against the three projection matrices.
# 4. Convert the resulting (seq_len × out_dim) tiles into the required
#    (seq_len, num_heads, head_dim) / (seq_len, num_kv_heads, head_dim)
#    layouts by splitting the tile rows (into heads) and tile columns
#    (into head_dim) using `retile_streamify`.
# 5. After the two split steps each output has stream shape (1, seq_len);
#    flatten the two stream dimensions to obtain a single stream dimension
#    of size `seq_len`, yielding shapes (64, 16, 32) for Q and
#    (64, 4, 32) for K and V as required.
# 6. Return the three StepTensors; the unused `cos` argument is ignored.

def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # Load raw tensors as single‑tile streams.
    input_s = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=64,
        tile_col=512,
    )
    q_proj_s = offchip_load(
        q_proj,
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=512,
        tile_col=512,
    )
    k_proj_s = offchip_load(
        k_proj,
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=512,
        tile_col=128,
    )
    v_proj_s = offchip_load(
        v_proj,
        stride=(1,),
        out_shape_tiled=(1,),
        tile_row=512,
        tile_col=128,
    )

    # ---- RMSNorm ---------------------------------------------------------
    eps = 1e-6
    hidden_dim = input_tensor.shape[-1]  # 512

    x_sq = unary_square(input_s)                     # x²
    sum_sq = unary_rowwise_sum(x_sq)                 # Σ x² over hidden dim
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)  # (Σ x²) / hidden_dim
    add_eps = unary_add_imm(mean_sq, eps)            # + ε
    inv_sqrt = unary_rsqrt(add_eps)                  # 1 / sqrt(...)
    normed = binary_mul(input_s, inv_sqrt)           # x * rsqrt(...)

    # ---- Q projection ----------------------------------------------------
    Q_full = binary_matmul(normed, q_proj_s)               # tile (64, 512)
    Q_split_rows = retile_streamify(Q_full, chunk=16, split_row=True)   # split rows → heads
    Q_split = retile_streamify(Q_split_rows, chunk=32, split_row=False)  # split cols → head_dim
    Q = flatten(Q_split, min_rank=0, max_rank=1)          # merge outer dim & seq_len

    # ---- K projection ----------------------------------------------------
    K_full = binary_matmul(normed, k_proj_s)               # tile (64, 128)
    K_split_rows = retile_streamify(K_full, chunk=4, split_row=True)    # split rows → kv heads
    K_split = retile_streamify(K_split_rows, chunk=32, split_row=False)  # split cols → head_dim
    K = flatten(K_split, min_rank=0, max_rank=1)          # merge outer dim & seq_len

    # ---- V projection ----------------------------------------------------
    V_full = binary_matmul(normed, v_proj_s)               # tile (64, 128)
    V_split_rows = retile_streamify(V_full, chunk=4, split_row=True)    # split rows → kv heads
    V_split = retile_streamify(V_split_rows, chunk=32, split_row=False)  # split cols → head_dim
    V = flatten(V_split, min_rank=0, max_rank=1)          # merge outer dim & seq_len

    return Q, K, V