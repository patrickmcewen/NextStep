# The rope operation is implemented using only STeP DSL primitives.
#   * Raw cosine/sine tensors are loaded with `offchip_load` and flattened to a
#     regular stream (shape (seq_len, 1, dim)).
#   * For each input tensor (Q or K) we:
#       1. Move the per‑head dimension (tile rows) into the stream with
#          `retile_streamify(..., split_row=True)`.
#       2. Split the feature dimension into two halves using
#          `retile_streamify(..., split_row=False)`.
#       3. Introduce an explicit “half‑index” stream dimension with
#          `reshape_stream(..., chunk_size=2)`.
#       4. Flatten the two stream axes and use `parallelize` to obtain the
#          two halves as separate streams.
#   * Cosine and sine are broadcast to the same head‑as‑stream shape using
#     `repeat_ref` with a reference built from the current input tensor.
#   * After broadcasting, cosine and sine are split the same way as the
#     input tensor, flattened, and parallelized.
#   * The four products required by RoPE are computed with `binary_mul`,
#     `binary_add` and a negation via `unary_mul_imm`.
#   * The two output halves are interleaved with `static_reassemble`,
#     the half‑index stream is merged back into the column dimension with
#     `reshape_stream` + `accum_retile_col`, and finally the head dimension
#     is restored with another `reshape_stream` + `accum_retile_row`.
#   * The transformed Q and K tensors are returned.

def apply_rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # ----------------------------------------------------------------------
    # Load and flatten the raw cosine / sine tables.
    # After `flatten` we have shape (seq_len, 1, dim).
    seq_len = cos.shape[0]                 # Python int
    dim = cos.shape[-1]                    # Python int
    half_dim = dim // 2                    # Python int, dim is even

    cos_stream = flatten(
        offchip_load(
            cos,
            stride=(1,),
            out_shape_tiled=(seq_len,),
            tile_row=1,
            tile_col=dim,
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
            tile_col=dim,
        ),
        min_rank=0,
        max_rank=1,
    )

    # ----------------------------------------------------------------------
    def _rope_one(x):
        # x: (seq_len, heads, dim)   where tile rows = heads, tile cols = dim
        heads = x.shape[-2]                     # Python int
        # 1) Move heads into the stream dimension.
        x_stream = retile_streamify(x, chunk=1, split_row=True)   # (seq_len*heads, 1, dim)

        # 2) Split the feature dimension into two halves.
        x_split = retile_streamify(x_stream, chunk=half_dim, split_row=False)  # (seq_len*heads*2, 1, half)

        # 3) Introduce an explicit half‑index stream dimension.
        x_split2 = reshape_stream(x_split, chunk_size=2, rank=0)   # (seq_len*heads, 2, 1, half)

        # 4) Flatten the two stream axes and separate the halves.
        x_flat = flatten(x_split2, min_rank=0, max_rank=1)        # (seq_len*heads*2, 1, half)
        x_parts = parallelize(x_flat, 2)                          # [x0, x1], each (seq_len*heads, 1, half)

        # ------------------------------------------------------------------
        # Broadcast cosine and sine to match the head‑as‑stream layout.
        # Build a reference with stream shape (seq_len, heads).
        ref = reshape_stream(
            retile_streamify(x, chunk=1, split_row=True),
            chunk_size=heads,
            rank=0,
        )  # (seq_len, heads, 1, dim)

        # Repeat cos / sin across the head dimension.
        cos_ref = repeat_ref(cos_stream, ref)   # (seq_len, heads, 1, dim)
        sin_ref = repeat_ref(sin_stream, ref)   # (seq_len, heads, 1, dim)

        # Collapse the two stream axes so we can reuse the same split logic as for x.
        cos_flat = flatten(cos_ref, min_rank=0, max_rank=1)   # (seq_len*heads, 1, dim)
        sin_flat = flatten(sin_ref, min_rank=0, max_rank=1)   # (seq_len*heads, 1, dim)

        # Split cosine / sine the same way as the data tensor.
        cos_split = retile_streamify(cos_flat, chunk=half_dim, split_row=False)   # (seq_len*heads*2, 1, half)
        sin_split = retile_streamify(sin_flat, chunk=half_dim, split_row=False)   # (seq_len*heads*2, 1, half)

        # Introduce the half‑index stream dimension.
        cos_split2 = reshape_stream(cos_split, chunk_size=2, rank=0)   # (seq_len*heads, 2, 1, half)
        sin_split2 = reshape_stream(sin_split, chunk_size=2, rank=0)   # (seq_len*heads, 2, 1, half)

        # Flatten and separate the halves.
        cos_flat2 = flatten(cos_split2, min_rank=0, max_rank=1)       # (seq_len*heads*2, 1, half)
        sin_flat2 = flatten(sin_split2, min_rank=0, max_rank=1)       # (seq_len*heads*2, 1, half)
        cos_parts = parallelize(cos_flat2, 2)                         # [c0, c1]
        sin_parts = parallelize(sin_flat2, 2)                         # [s0, s1]

        # ------------------------------------------------------------------
        # Compute RoPE: out0 = Q0 * cos0 - Q1 * sin0
        #               out1 = Q1 * cos1 + Q0 * sin1
        term0 = binary_mul(x_parts[0], cos_parts[0])                # Q0 * cos0
        term1 = binary_mul(x_parts[1], cos_parts[1])                # Q1 * cos1

        cross0 = binary_mul(x_parts[1], sin_parts[0])               # Q1 * sin0
        cross1 = binary_mul(x_parts[0], sin_parts[1])               # Q0 * sin1

        cross0_neg = unary_mul_imm(cross0, -1.0)                     # -Q1 * sin0

        out0 = binary_add(term0, cross0_neg)                        # first half
        out1 = binary_add(term1, cross1)                            # second half

        # ------------------------------------------------------------------
        # Interleave the two halves back together.
        merged_half = static_reassemble([out0, out1])               # (2*seq_len*heads, 1, half)

        # Merge half‑index stream into the column dimension.
        merged_half = reshape_stream(merged_half, chunk_size=2, rank=0)  # (seq_len*heads, 2, 1, half)
        merged = accum_retile_col(merged_half, rank=1)                 # (seq_len*heads, 1, dim)

        # Restore the original head‑as‑stream layout.
        merged = reshape_stream(merged, chunk_size=heads, rank=0)     # (seq_len, heads, 1, dim)
        merged = accum_retile_row(merged, rank=1)                     # (seq_len, heads, dim)

        return merged

    # Apply the rope transformation to both Q and K.
    Q_out = _rope_one(Q)
    K_out = _rope_one(K)

    return Q_out, K_out