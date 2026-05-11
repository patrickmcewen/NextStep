# The rope operation can be expressed entirely with STeP DSL primitives.
#   1. Load the RAW cosine/sine tables into on‑chip streams (flatten the leading 1).
#   2. For each input tensor (Q or K):
#        a) Move the head dimension (tile rows) into the stream using retile_streamify(chunk=1, split_row=True).
#        b) Split the feature dimension (tile cols) into two halves with retile_streamify(chunk=half, split_row=False).
#        c) Flatten the two stream axes so that the two halves become a single stream dimension.
#        d) Parallelize the flattened stream into two tensors, one per half.
#        e) Replicate the cosine/sine streams for each head, split them the same way and parallelize.
#        f) Compute the four products:
#              – Q_half0 * cos_half0
#              – Q_half1 * cos_half1
#              – Q_half1 * sin_half0   (negated)
#              – Q_half0 * sin_half1
#           and combine them with binary_add / unary_mul_imm.
#        g) Interleave the two resulting halves (static_reassemble) and merge the half‑stream back
#           into the column dimension (reshape_stream + accum_retile_col).
#        h) Restore the original head‑as‑row layout (reshape_stream + accum_retile_row).
#   3. Return the transformed Q and K tensors.
#
# All tensor manipulations use only DSL calls; raw PyTorch ops are avoided.

def apply_rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # -------------------- Load COS / SIN ------------------------------------
    seq_len = cos.shape[0]                # vanilla sequence length
    head_dim = cos.shape[-1]               # should be 32
    half_dim = head_dim // 2               # 16

    # Load the raw tables and flatten away the leading singleton dimension
    cos_stream = flatten(
        offchip_load(
            cos,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=1,
            tile_col=head_dim,
        ),
        min_rank=0,
        max_rank=1,
    )
    sin_stream = flatten(
        offchip_load(
            sin,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=1,
            tile_col=head_dim,
        ),
        min_rank=0,
        max_rank=1,
    )

    # ----------------------------------------------------------------------
    def _rope_one(x):
        # Number of heads for this tensor (tile rows)
        n_heads = x.shape[-2]               # int

        # 1) Move heads into the stream dimension (each head becomes a separate token)
        x_head_stream = retile_streamify(x, chunk=1, split_row=True)   # (S*H, 1, 1, C)

        # 2) Split the feature dimension into two halves (creates a half‑stream)
        x_split = retile_streamify(x_head_stream, chunk=half_dim, split_row=False)  # (S*H, 2, 1, half)

        # 3) Flatten the two stream axes so that the two halves become a single stream dimension
        x_flat = flatten(x_split, min_rank=0, max_rank=1)            # (S*H*2, 1, half)

        # 4) Separate the two halves
        x_parts = parallelize(x_flat, 2)                             # [x0, x1], each (S*H, 1, half)

        # ------------------------------------------------------------------
        # Prepare matching cosine and sine streams for this tensor
        # Replicate the (seq_len, 1, C) tables for each head
        cos_rep = repeat_static(cos_stream, factor=n_heads)         # (S, H, 1, C)
        sin_rep = repeat_static(sin_stream, factor=n_heads)         # (S, H, 1, C)

        # Merge the (S, H) stream dims to align with x_parts
        cos_flat = flatten(cos_rep, min_rank=0, max_rank=1)          # (S*H, 1, C)
        sin_flat = flatten(sin_rep, min_rank=0, max_rank=1)          # (S*H, 1, C)

        # Split cosine / sine the same way as the data
        cos_split = retile_streamify(cos_flat, chunk=half_dim, split_row=False)  # (S*H, 2, 1, half)
        sin_split = retile_streamify(sin_flat, chunk=half_dim, split_row=False)  # (S*H, 2, 1, half)

        # Flatten the half‑stream dimension
        cos_flat2 = flatten(cos_split, min_rank=0, max_rank=1)      # (S*H*2, 1, half)
        sin_flat2 = flatten(sin_split, min_rank=0, max_rank=1)      # (S*H*2, 1, half)

        # Separate into two halves
        cos_parts = parallelize(cos_flat2, 2)                       # [c0, c1], each (S*H, 1, half)
        sin_parts = parallelize(sin_flat2, 2)                       # [s0, s1], each (S*H, 1, half)

        # ------------------------------------------------------------------
        # Compute the four products required by RoPE
        term0 = binary_mul(x_parts[0], cos_parts[0])                # Q_half0 * cos0
        term1 = binary_mul(x_parts[1], cos_parts[1])                # Q_half1 * cos1

        cross0 = binary_mul(x_parts[1], sin_parts[0])               # Q_half1 * sin0
        cross1 = binary_mul(x_parts[0], sin_parts[1])               # Q_half0 * sin1

        # Negate the first cross term (the rotated second half)
        cross0_neg = unary_mul_imm(cross0, -1.0)

        # Combine the terms for each half
        out0 = binary_add(term0, cross0_neg)                        # first output half
        out1 = binary_add(term1, cross1)                            # second output half

        # ------------------------------------------------------------------
        # Interleave the two halves back into a single stream
        merged_half = static_reassemble([out0, out1])               # (2*S*H, 1, half)

        # Merge the half‑stream into the column dimension
        merged_half = reshape_stream(merged_half, chunk_size=2, rank=0)   # (S*H, 2, 1, half)
        merged = accum_retile_col(merged_half, rank=1)                   # (S*H, 1, C)

        # Restore the original head‑as‑row layout
        merged = reshape_stream(merged, chunk_size=n_heads, rank=0)      # (S, H, 1, C)
        merged = accum_retile_row(merged, rank=1)                        # (S, H, C)

        return merged

    # Apply the rope transformation to Q and K separately
    Q_out = _rope_one(Q)
    K_out = _rope_one(K)

    return Q_out, K_out