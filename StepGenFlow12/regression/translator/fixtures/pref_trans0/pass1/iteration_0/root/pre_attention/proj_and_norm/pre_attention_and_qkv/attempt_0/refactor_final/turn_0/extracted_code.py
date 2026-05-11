# Implementation notes:
# 1. Load the raw off‑chip tensors as tile streams.  The input sequence is streamed
#    over the first dimension (seq_len) with a tile of shape (1, hidden_dim).
#    The projection matrices are loaded as whole‑tensor tiles (no streaming) and
#    then expanded to match the input’s stream shape using `expand_ref`.
# 2. RMSNorm is expressed with DSL ops:
#        x² → row‑wise sum → divide by hidden_dim → add ε → rsqrt → multiply.
#    All arithmetic uses binary/unary DSL calls; the only scalar constants are
#    injected via `unary_to_const_int` or `unary_add_imm`.
# 3. Q, K, V are obtained with `binary_matmul`.  The result has tile shape
#    (1, hidden_dim).  To achieve the required output tiling (num_heads, head_dim)
#    we:
#        a) split the column tile into chunks of size `head_dim` using
#           `retile_streamify(..., split_row=False)`.  This introduces a new
#           stream dimension equal to the number of heads.
#        b) merge that stream dimension into the tile‑row dimension with
#           `accum_retile_row`, turning the tile into (num_heads, head_dim).
#        c) collapse the leading singleton stream dimension together with the
#           sequence‑length stream dimension via `flatten`, yielding a stream
#           shape (seq_len, num_heads, head_dim) which matches the contract
#           shape (64, 16, 32) for Q and (64, 4, 32) for K and V.
# 4. The head dimension is taken from the first output shape; the number of KV
#    heads is inferred from the shapes of the K/V projection matrices.
# 5. No raw torch arithmetic is used – every operation is a DSL call.

def pre_attention_and_qkv(input_tensor, q_proj, k_proj, v_proj, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # 0. Extract static dimensions from the vanilla (off‑chip) tensors.
    # ------------------------------------------------------------------
    seq_len = input_tensor.shape[0]          # 64 for the given config
    hidden_dim = input_tensor.shape[1]       # 512
    # Head dimension is the trailing size of the first requested output.
    head_dim = out_shapes[0][2]              # 32

    # ------------------------------------------------------------------
    # 1. Load the input token tensor and apply RMSNorm.
    # ------------------------------------------------------------------
    # Load as a stream over the sequence dimension (tile: 1 × hidden_dim).
    inp = offchip_load(
        input_tensor,
        stride=(1, 1),                # (grid_c, 1) for row‑major tile indexing
        out_shape_tiled=(seq_len, 1),
        tile_row=1,
        tile_col=hidden_dim,
    )
    # RMSNorm: x * rsqrt(mean(x²) + eps)
    sq = unary_square(inp)                               # x²
    sum_sq = unary_rowwise_sum(sq)                       # sum over hidden dim → (…,1,1)
    denom = unary_to_const_int(sum_sq, constant=hidden_dim)  # hidden_dim as float tensor
    mean_sq = binary_div(sum_sq, denom)                  # mean of squares
    mean_eps = unary_add_imm(mean_sq, constant=1e-6)     # add ε
    inv_rms = unary_rsqrt(mean_eps)                      # rsqrt(mean + ε)
    normed = binary_mul(inp, inv_rms)                    # scale original tensor

    # ------------------------------------------------------------------
    # 2. Load and broadcast the projection matrices.
    # ------------------------------------------------------------------
    # Helper to load a weight matrix and broadcast it to the normed stream.
    def load_and_broadcast(weight, out_dim):
        w_loaded = offchip_load(
            weight,
            stride=(1, 1),               # no streaming for the weight matrix
            out_shape_tiled=(1, 1),
            tile_row=hidden_dim,
            tile_col=out_dim,
        )
        # Expand over the seq_len dimension (expand the trailing two stream dims).
        return expand_ref(w_loaded, normed, expand_rank=2)

    # Q projection matrix (hidden_dim → num_heads * head_dim)
    q_stream = load_and_broadcast(q_proj, hidden_dim)   # out_dim = hidden_dim (512)

    # K and V projection matrices (hidden_dim → num_kv_heads * head_dim)
    kv_dim = k_proj.shape[1]      # 128 (num_kv_heads * head_dim)
    k_stream = load_and_broadcast(k_proj, kv_dim)
    v_stream = load_and_broadcast(v_proj, kv_dim)

    # ------------------------------------------------------------------
    # 3. Compute Q, K, V with matmul and reshape to (seq_len, heads, head_dim).
    # ------------------------------------------------------------------
    # --- Q ---
    q_raw = binary_matmul(normed, q_stream)               # (…,1, hidden_dim)
    q_split = retile_streamify(q_raw, chunk=head_dim, split_row=False)  # (…, num_heads,1,head_dim)
    q_merged = accum_retile_row(q_split, rank=1)          # (…, num_heads, head_dim)
    Q = flatten(q_merged, min_rank=0, max_rank=1)         # (seq_len, num_heads, head_dim)

    # --- K ---
    k_raw = binary_matmul(normed, k_stream)
    k_split = retile_streamify(k_raw, chunk=head_dim, split_row=False)
    k_merged = accum_retile_row(k_split, rank=1)
    K = flatten(k_merged, min_rank=0, max_rank=1)

    # --- V ---
    v_raw = binary_matmul(normed, v_stream)
    v_split = retile_streamify(v_raw, chunk=head_dim, split_row=False)
    v_merged = accum_retile_row(v_split, rank=1)
    V = flatten(v_merged, min_rank=0, max_rank=1)

    return Q, K, V