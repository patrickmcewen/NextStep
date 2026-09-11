# pre_attn_norm_and_proj
# ----------------------
# Implements the pre‑attention RMSNorm followed by Q, K, V linear projections.
# All RAW tensors are first streamed onto‑chip with `offchip_load`.  RMSNorm is
# built from unary/binary ops (square → row‑wise sum → mean → ε‑add → rsqrt →
# scale).  Each weight matrix is broadcast across the sequence dimension,
# multiplied with the normalized activations, and then reshaped from
# (seq_len, heads*head_dim) → (seq_len, heads, head_dim).  The extra leading
# singleton stream dimension introduced by `offchip_load` is removed with
# `flatten`, merging the (1, seq_len) stream into a single `seq_len` dim.
# The resulting tensors have the exact shapes declared by the parent:
#   Q → (64, 16, 32)   K → (64,  4, 32)   V → (64,  4, 32)

def pre_attn_norm_and_proj(input_tensor, q_proj, k_proj, v_proj, cos,
                          *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Scalar shape constants (pure Python arithmetic is allowed)
    # ------------------------------------------------------------------
    seq_len    = input_tensor.shape[0]          # 64
    hidden_dim = input_tensor.shape[1]          # 512
    head_dim   = cos.shape[-1]                  # 32
    num_heads  = q_proj.shape[1] // head_dim    # 16
    num_kv_heads = k_proj.shape[1] // head_dim  # 4
    eps = 1e-6

    # ------------------------------------------------------------------
    # Load the raw tensors into the on‑chip stream format.
    # offchip_load always adds a leading singleton stream dimension.
    # ------------------------------------------------------------------
    # input_tensor: stream over the sequence, tile = (1, hidden_dim)
    inp = offchip_load(
        input_tensor,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=hidden_dim,
    )  # shape: stream(1, seq_len) × tile(1, hidden_dim)

    # ------------------------------------------------------------------
    # RMSNorm = x * rsqrt(mean(x²) + eps)
    # ------------------------------------------------------------------
    inp_sq   = unary_square(inp)                         # tile(1, hidden_dim)
    sum_sq   = unary_rowwise_sum(inp_sq)                 # tile(1,1)
    mean_sq  = unary_mul_imm(sum_sq, 1.0 / hidden_dim)   # tile(1,1)
    mean_eps = unary_add_imm(mean_sq, eps)               # tile(1,1)
    scale    = unary_rsqrt(mean_eps)                     # tile(1,1)
    normed   = binary_mul(inp, scale)                    # tile(1, hidden_dim)

    # ------------------------------------------------------------------
    # Helper: project `normed` with a weight matrix and reshape to
    # (seq_len, heads, head_dim).  Returns a tensor of shape
    # stream(1, seq_len) × tile(heads, head_dim).
    # ------------------------------------------------------------------
    def proj_and_reshape(weight, out_dim, heads):
        # Broadcast weight across the sequence dimension.
        w = offchip_load(
            weight,
            stride=(0,),                # same tile for every seq position
            out_shape_tiled=(seq_len,),
            tile_row=hidden_dim,
            tile_col=out_dim,
        )  # shape: stream(1, seq_len) × tile(hidden_dim, out_dim)

        # MatMul: (seq_len, hidden_dim) @ (hidden_dim, out_dim) → (seq_len, out_dim)
        raw = binary_matmul(normed, w)               # stream(1, seq_len) × tile(1, out_dim)

        # Promote to insert a dummy stream dim before the tile‑col.
        raw_promoted = promote(raw, rank=0)          # stream(1, seq_len, 1) × tile(1, out_dim)

        # Split the tile‑col (out_dim = heads * head_dim) into a new stream dim.
        raw_split = retile_streamify(
            raw_promoted,
            chunk=head_dim,
            split_row=False,                         # split column dimension
        )  # shape: stream(1, seq_len, heads) × tile(1, head_dim)

        # Merge the new stream dim (heads) into the tile‑row dimension.
        final = accum_retile_row(raw_split, rank=1)  # stream(1, seq_len) × tile(heads, head_dim)
        return final

    # ------------------------------------------------------------------
    # Compute Q, K, V and remove the leading singleton stream dim.
    # ------------------------------------------------------------------
    Q_raw = proj_and_reshape(q_proj, q_proj.shape[1], num_heads)      # (1, seq_len) × (16,32)
    K_raw = proj_and_reshape(k_proj, k_proj.shape[1], num_kv_heads)   # (1, seq_len) × (4,32)
    V_raw = proj_and_reshape(v_proj, v_proj.shape[1], num_kv_heads)   # (1, seq_len) × (4,32)

    # Flatten the leading singleton with the sequence dimension.
    Q = flatten(Q_raw, min_rank=0, max_rank=1)   # stream(seq_len) × tile(16,32) → (64,16,32)
    K = flatten(K_raw, min_rank=0, max_rank=1)   # (64,4,32)
    V = flatten(V_raw, min_rank=0, max_rank=1)   # (64,4,32)

    return Q, K, V