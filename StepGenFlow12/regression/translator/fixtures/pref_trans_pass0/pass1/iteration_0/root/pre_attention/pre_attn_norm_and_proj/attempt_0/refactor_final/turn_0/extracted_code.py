# Implementation reasoning:
# 1. Load all raw off‑chip tensors with `offchip_load`.  We stream them as a
#    single tile (out_shape_tiled = (1,)), using the full dimensions as the
#    tile size so that the resulting stream shape is (1, 1, ..., tile_row, tile_col).
# 2. RMSNorm = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps).
#    - Square the input with `binary_mul`.
#    - Sum over the hidden dimension using `unary_rowwise_sum`.
#    - Convert the sum to a mean by multiplying with 1/hidden_dim (`unary_mul_imm`).
#    - Add ε (`unary_add_imm`) and apply rsqrt (`unary_rsqrt`).
#    - Multiply the original input by the rsqrt factor (`binary_mul`), letting the
#      broadcasted scalar broadcast over the hidden tile dimension.
# 3. Project to Q, K, V with `binary_matmul`.
# 4. Reshape each projection from (seq_len, heads*head_dim) to
#    (seq_len, heads, head_dim):
#       a) Split the column tile (heads*head_dim) into a stream of `heads`
#          and a column tile of size `head_dim` using `retile_streamify(...,
#          split_row=False)`.
#       b) Split the row tile (seq_len) into a stream of `seq_len` and a row tile
#          of size `heads` (for Q) or `kv_heads` (for K,V) using
#          `retile_streamify(..., split_row=True)`.
#       c) The result has a leading dummy stream dim of size 1; merge it with the
#          seq_len stream dim using `flatten(..., min_rank=0, max_rank=1)`.
# 5. Return the three reshaped streams.  `out_shapes` and `out_perms` are
#    accepted but not used (the caller supplies them for contract checking).

def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos, *, out_shapes, out_perms=None):
    # --- dimension constants -------------------------------------------------
    seq_len = input_tensor.shape[0]          # 64
    hidden_dim = input_tensor.shape[1]       # 512
    head_dim = cos.shape[-1]                 # 32

    # --- load raw tensors as single‑tile streams ----------------------------
    inp = offchip_load(
        input_tensor, stride=(1,), out_shape_tiled=(1,),
        tile_row=seq_len, tile_col=hidden_dim
    )
    w_q = offchip_load(
        q_proj, stride=(1,), out_shape_tiled=(1,),
        tile_row=hidden_dim, tile_col=q_proj.shape[1]
    )
    w_k = offchip_load(
        k_proj, stride=(1,), out_shape_tiled=(1,),
        tile_row=hidden_dim, tile_col=k_proj.shape[1]
    )
    w_v = offchip_load(
        v_proj, stride=(1,), out_shape_tiled=(1,),
        tile_row=hidden_dim, tile_col=v_proj.shape[1]
    )

    # --- RMSNorm ------------------------------------------------------------
    eps = 1e-6
    sq = binary_mul(inp, inp)                          # x^2
    sum_sq = unary_rowwise_sum(sq)                     # sum over hidden dim
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)  # divide by hidden_dim
    mean_eps = unary_add_imm(mean_sq, eps)             # + eps
    rsqrt = unary_rsqrt(mean_eps)                      # rsqrt(...)
    normed = binary_mul(inp, rsqrt)                    # x * rsqrt(...)

    # --- Q, K, V matmuls ----------------------------------------------------
    q_mat = binary_matmul(normed, w_q)   # (1,1,seq_len, heads*head_dim)
    k_mat = binary_matmul(normed, w_k)   # (1,1,seq_len, kv_heads*head_dim)
    v_mat = binary_matmul(normed, w_v)   # (1,1,seq_len, kv_heads*head_dim)

    # --- helper to reshape projection results -------------------------------
    def _reshape_proj(mat, num_heads):
        # split hidden dimension into heads (stream) and head_dim (tile)
        col_split = retile_streamify(mat, chunk=head_dim, split_row=False)
        # split the seq_len tile into (seq_len) stream and heads tile rows
        row_split = retile_streamify(col_split, chunk=num_heads, split_row=True)
        # merge the leading dummy stream dim with the seq_len stream dim
        return flatten(row_split, min_rank=0, max_rank=1)

    # number of heads for each projection
    num_q_heads = q_proj.shape[1] // head_dim          # 16
    num_kv_heads = k_proj.shape[1] // head_dim         # 4

    Q = _reshape_proj(q_mat, num_q_heads)
    K = _reshape_proj(k_mat, num_kv_heads)
    V = _reshape_proj(v_mat, num_kv_heads)

    return Q, K, V