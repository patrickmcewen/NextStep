# Implementation reasoning:
# 1. Apply per‑head RMS‑Norm to Q and K:
#    rms_norm(x) = x * rsqrt( (x*x).rowwise_sum() / head_dim + eps )
#    – square with binary_mul,
#    – row‑wise sum with unary_rowwise_sum,
#    – divide by head_dim with unary_mul_imm,
#    – add epsilon with unary_add_imm,
#    – rsqrt with unary_rsqrt,
#    – final scaling with binary_mul.
#
# 2. Load the RAW cos/sin tensors from off‑chip.
#    offchip_load creates a stream (1, seq_len, 1, head_dim);
#    flatten merges the leading singleton stream dim, yielding (seq_len, 1, head_dim),
#    which can broadcast against Q/K tile rows.
#
# 3. RoPE rotation of half the vector:
#    rotate_half(x):
#        – split the column tile into two halves via retile_streamify (chunk = head_dim/2).
#        – parallelize the resulting stream into two streams (first half, second half).
#        – negate the second half with unary_mul_imm(-1).
#        – interleave them in swapped order using static_reassemble([second, first]).
#        – reshape the combined stream back to (seq_len, 2, …) with reshape_stream.
#        – absorb the inner stream dim (size 2) into the column tile via accum_retile_col.
#    This reproduces torch.cat([-x[...,h:], x[...,:h]], dim=-1).
#
# 4. Apply RoPE:
#        Q_out = Q_norm * cos + rotate_half(Q_norm) * sin
#        K_out = K_norm * cos + rotate_half(K_norm) * sin
#    V passes through unchanged.
#
# 5. Return the three tensors; the shapes match the required
#    (64, 16, 32), (64, 4, 32), (64, 4, 32).


def per_head_norm_and_rope(Q, K, V, cos, sin, *, out_shapes, out_perms=None):
    # ----- helper: per‑head RMS‑Norm -----
    def rms_norm(x):
        # static head dimension (last tile dimension)
        head_dim = int(x.tensor.shape[-1])
        # x * x
        sq = binary_mul(x, x)
        # sum over columns, keep dim → shape (..., 1)
        sum_sq = unary_rowwise_sum(sq)
        # mean = sum / head_dim
        mean_sq = unary_mul_imm(sum_sq, 1.0 / head_dim)
        # add epsilon
        mean_eps = unary_add_imm(mean_sq, 1e-6)
        # rsqrt
        inv_rms = unary_rsqrt(mean_eps)
        # final scaling
        return binary_mul(x, inv_rms)

    # ----- helper: rotate‑half operation for RoPE -----
    def rotate_half(x):
        # split column tile into two halves (chunk = head_dim/2)
        half = int(x.tensor.shape[-1] // 2)
        split = retile_streamify(x, chunk=half, split_row=False)

        # split the stream into two halves (first, second)
        halves = parallelize(split, n=2)          # [first_half, second_half]

        # negate the second half
        halves[1] = unary_mul_imm(halves[1], -1.0)

        # interleave with swapped order (second first)
        swapped = static_reassemble([halves[1], halves[0]])

        # reshape stream dim back to (seq_len, 2)
        reshaped = reshape_stream(swapped, chunk_size=2, rank=0)

        # absorb the inner stream dim into the column tile
        return accum_retile_col(reshaped, rank=1)

    # ----- RMS‑Norm Q and K -----
    Q_norm = rms_norm(Q)
    K_norm = rms_norm(K)

    # ----- load and prepare cos / sin -----
    seq_len = int(cos.shape[0])
    head_dim = int(cos.shape[-1])          # should equal Q/K head dimension

    # off‑chip load (creates a leading singleton stream dimension)
    cos_loaded = offchip_load(
        cos,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=head_dim,
    )
    sin_loaded = offchip_load(
        sin,
        stride=(1,),
        out_shape_tiled=(seq_len,),
        tile_row=1,
        tile_col=head_dim,
    )

    # remove the leading singleton stream dim so the stream shape is (seq_len,)
    cos_stream = flatten(cos_loaded, min_rank=0, max_rank=1)
    sin_stream = flatten(sin_loaded, min_rank=0, max_rank=1)

    # ----- apply RoPE to Q -----
    Q_rot = rotate_half(Q_norm)
    Q_part_cos = binary_mul(Q_norm, cos_stream)
    Q_part_sin = binary_mul(Q_rot, sin_stream)
    Q_out = binary_add(Q_part_cos, Q_part_sin)

    # ----- apply RoPE to K -----
    K_rot = rotate_half(K_norm)
    K_part_cos = binary_mul(K_norm, cos_stream)
    K_part_sin = binary_mul(K_rot, sin_stream)
    K_out = binary_add(K_part_cos, K_part_sin)

    # V is unchanged
    return Q_out, K_out, V