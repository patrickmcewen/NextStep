# Implementation reasoning:
# 1️⃣ Load the raw `input_tensor` as a stream of tiles where the hidden dimension
#    lives in the **tile column** (size = hidden_dim).  This lets us compute
#    RMSNorm using `unary_rowwise_sum` which reduces over the tile‑column
#    dimension.
# 2️⃣ RMSNorm:   x² → sum → mean (+eps) → rsqrt → multiply back.
#    The mean is obtained by scaling the summed squares with 1/hidden_dim.
#    The scalar rsqrt tile is broadcast over the hidden‑dim tile column by
#    ordinary PyTorch broadcasting inside `binary_mul`.
# 3️⃣ For each linear projection (Q, K, V) we:
#    • Load the weight matrix with tile shape (hidden_dim, 1) and a stream shape
#      (seq_len, output_dim).  The stride `(0, 1)` broadcasts the same weight
#      across all sequence positions.
#    • Broadcast the normalized activation across the `output_dim` stream
#      dimension using `expand_ref` (the activation’s trailing stream dim is 1).
#    • Perform the matrix‑multiply with `binary_matmul`.  Because the activation
#      tiles are (1 × hidden_dim) and the weight tiles are (hidden_dim × 1),
#      the matmul reduces over the hidden dimension and yields tiles of shape
#      (1 × 1).  The resulting stream shape is (seq_len, output_dim).
#    • Finally split `output_dim` into `(num_heads, head_dim)` (or
#      `(num_kv_heads, head_dim)`) with `reshape_stream`, using `head_dim` as
#      the chunk size.
# 4️⃣ Return the three projected tensors.  Each output has stream shape
#    `(seq_len, num_heads, head_dim)` (or `(seq_len, num_kv_heads, head_dim)`)
#    and tile shape (1, 1), satisfying the required output signatures.
#
# All tensor operations are expressed using the provided DSL functions; no raw
# PyTorch arithmetic is used.

def pre_attention_and_qkv(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    # -----------------------------------------------------------------
    # Extract model‑level dimensions from inputs and the contract.
    # -----------------------------------------------------------------
    seq_len = input_tensor.shape[0]                # S
    hidden_dim = input_tensor.shape[1]             # H
    # Q output shape: (seq_len, num_heads, head_dim)
    _, num_heads, head_dim = out_shapes[0]
    # K/V output shape: (seq_len, num_kv_heads, head_dim)
    _, num_kv_heads, _ = out_shapes[1]

    # -----------------------------------------------------------------
    # 1️⃣ Load the raw activation tensor.
    #    Tile = (1, hidden_dim)  → hidden_dim lives in the tile‑column.
    # -----------------------------------------------------------------
    act = offchip_load(
        input_tensor,
        stride=(1, 0),                     # (row_stride, col_stride) for (seq_len, 1) streaming
        out_shape_tiled=(seq_len, 1),      # one tile per token, broadcast over hidden dim
        tile_row=1,
        tile_col=hidden_dim,
        transposed=False,
        par_dispatch=1,
    )

    # -----------------------------------------------------------------
    # 2️⃣ RMSNorm:  x * rsqrt( mean(x²) + eps )
    #    - square the activation
    #    - sum over the hidden‑dim (tile column) -> mean by scaling
    #    - add epsilon, rsqrt, then multiply back.
    # -----------------------------------------------------------------
    eps = 1e-6

    sq = unary_square(act)
    sum_sq = unary_rowwise_sum(sq)                     # sum over hidden_dim (tile col)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / hidden_dim)  # divide by hidden_dim
    mean_eps = unary_add_imm(mean_sq, eps)
    rsqrt = unary_rsqrt(mean_eps)                      # scalar per token (tile 1×1)

    normed = binary_mul(act, rsqrt)                    # broadcast rsqrt over hidden_dim

    # -----------------------------------------------------------------
    # Helper to compute a projection (Q / K / V)
    # -----------------------------------------------------------------
    def _project(weight, out_dim):
        # Load weight matrix.
        #   Tile = (hidden_dim, 1)      → each column of the weight is a tile.
        #   Stream shape = (seq_len, out_dim)  (broadcast across seq_len).
        w = offchip_load(
            weight,
            stride=(0, 1),                      # broadcast over seq_len, step over out_dim
            out_shape_tiled=(seq_len, out_dim),
            tile_row=hidden_dim,
            tile_col=1,
            transposed=False,
            par_dispatch=1,
        )
        # Broadcast the normalized activation across the out_dim stream dim.
        norm_b = expand_ref(normed, w, expand_rank=1)  # expands the size‑1 dim to out_dim

        # Matrix multiply: (1 × hidden_dim) @ (hidden_dim × 1) → (1 × 1) tile.
        proj_raw = binary_matmul(norm_b, w, weight_transposed=False)

        # Split the output_dim stream dimension into (heads, head_dim).
        proj = reshape_stream(proj_raw, chunk_size=head_dim, rank=0)
        return proj

    # -----------------------------------------------------------------
    # Compute Q, K, V using the helper.
    # -----------------------------------------------------------------
    out_dim_q = q_proj.shape[1]               # num_heads * head_dim
    out_dim_k = k_proj.shape[1]               # num_kv_heads * head_dim
    out_dim_v = v_proj.shape[1]               # same as out_dim_k

    Q = _project(q_proj, out_dim_q)
    K = _project(k_proj, out_dim_k)
    V = _project(v_proj, out_dim_v)

    return Q, K, V