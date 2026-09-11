# per_head_norm_and_rope
# ------------------------------------------------------------
# 1. Load the RAW cosine / sine tensors from off‑chip, flatten the leading
#    singleton dimension that `offchip_load` adds, and keep them as streams.
# 2. Perform per‑head RMSNorm using only DSL primitives.
# 3. Apply RoPE.  The rotation‐half operation is implemented by splitting the
#    head dimension into two halves (using `retile_streamify`), separating the
#    halves into independent streams (`flatten` + `parallelize`), applying the
#    RoPE formulas
#        first_half  = a * cos_a  –  b * sin_a
#        second_half = b * cos_b  +  a * sin_b
#    where *a* and *b* are the two halves of the input, and then merging the
#    halves back (using `static_reassemble`, `reshape_stream`,
#    `accum_retile_col`).  This reproduces the exact behaviour of the
#    reference `_rotate_half` helper without using any Python indexing.
# 4. V is passed through unchanged.
#
# All arithmetic is expressed via DSL calls; only scalar Python operations
# (e.g. computing the half‑size) are used.
def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    # -------------------------------------------------
    # Load RAW cosine / sine tensors
    # -------------------------------------------------
    cos_loaded = offchip_load(
        cos, stride=(1,), out_shape_tiled=(64,), tile_row=1, tile_col=32
    )
    sin_loaded = offchip_load(
        sin, stride=(1,), out_shape_tiled=(64,), tile_row=1, tile_col=32
    )
    # Remove the leading singleton added by offchip_load
    cos_stream = flatten(cos_loaded, min_rank=0, max_rank=1)  # stream(64,)×tile(1,32)
    sin_stream = flatten(sin_loaded, min_rank=0, max_rank=1)  # stream(64,)×tile(1,32)

    # -------------------------------------------------
    # RMS‑Norm for Q
    # -------------------------------------------------
    Q_sq   = unary_square(Q)
    Q_sum  = unary_rowwise_sum(Q_sq)                 # (..., tile_r, 1)
    Q_mean = unary_mul_imm(Q_sum, 1.0 / 32.0)         # divide by head_dim
    Q_eps  = unary_add_imm(Q_mean, 1e-6)
    Q_rsqrt = unary_rsqrt(Q_eps)
    Q_norm = binary_mul(Q, Q_rsqrt)

    # -------------------------------------------------
    # RMS‑Norm for K
    # -------------------------------------------------
    K_sq   = unary_square(K)
    K_sum  = unary_rowwise_sum(K_sq)
    K_mean = unary_mul_imm(K_sum, 1.0 / 32.0)
    K_eps  = unary_add_imm(K_mean, 1e-6)
    K_rsqrt = unary_rsqrt(K_eps)
    K_norm = binary_mul(K, K_rsqrt)

    # -------------------------------------------------
    # Helper: apply RoPE to a (stream, tile) tensor
    # -------------------------------------------------
    def apply_rope(x, cos_t, sin_t):
        # x, cos_t, sin_t all have shape stream(64,)×tile(R, head_dim)
        head_dim = x.shape[-1]          # Python int
        half = head_dim // 2            # split point

        # ---- Split the head dimension into two halves ----
        # retile column => new stream dim (seq_len*2) and half‑sized tile_c
        x_split = retile_streamify(x, chunk=half, split_row=False)
        cos_split = retile_streamify(cos_t, chunk=half, split_row=False)
        sin_split = retile_streamify(sin_t, chunk=half, split_row=False)

        # reshape stream (seq_len*2) → (seq_len, 2)
        x_resh = reshape_stream(x_split, chunk_size=2, rank=0)
        cos_resh = reshape_stream(cos_split, chunk_size=2, rank=0)
        sin_resh = reshape_stream(sin_split, chunk_size=2, rank=0)

        # flatten the two stream dimensions and parallelize to obtain the two halves
        x_flat = flatten(x_resh, min_rank=0, max_rank=1)      # stream(64*2,)×tile(R,half)
        cos_flat = flatten(cos_resh, min_rank=0, max_rank=1)
        sin_flat = flatten(sin_resh, min_rank=0, max_rank=1)

        x_parts = parallelize(x_flat, n=2)    # [a, b]
        cos_parts = parallelize(cos_flat, n=2)  # [cos_a, cos_b]
        sin_parts = parallelize(sin_flat, n=2)  # [sin_a, sin_b]

        a, b = x_parts
        cos_a, cos_b = cos_parts
        sin_a, sin_b = sin_parts

        # ---- RoPE formulas for the two halves ----
        a_cos = binary_mul(a, cos_a)                         # a * cos_a
        b_cos = binary_mul(b, cos_b)                         # b * cos_b
        b_sin = binary_mul(b, sin_a)                         # b * sin_a
        a_sin = binary_mul(a, sin_b)                         # a * sin_b

        first_half  = binary_add(a_cos, unary_mul_imm(b_sin, -1.0))  # a*cos_a - b*sin_a
        second_half = binary_add(b_cos, a_sin)                       # b*cos_b + a*sin_b

        # ---- Re‑assemble the halves back into the original head dimension ----
        # interleave the two streams (a token’s first half, then its second half)
        interleaved = static_reassemble([first_half, second_half])

        # reshape stream (seq_len*2) → (seq_len, 2) again
        inter_resh = reshape_stream(interleaved, chunk_size=2, rank=0)

        # merge the half‑index stream dimension into the tile column dimension
        out = accum_retile_col(inter_resh, rank=1)

        return out

    # -------------------------------------------------
    # Apply RoPE to Q and K
    # -------------------------------------------------
    Q_out = apply_rope(Q_norm, cos_stream, sin_stream)
    K_out = apply_rope(K_norm, cos_stream, sin_stream)

    # V is unchanged
    V_out = V

    return Q_out, K_out, V_out