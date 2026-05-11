# Implementation reasoning:
# Load the RAW input via `offchip_load` with tile size 1×hidden_dim (tile_row=1,
# tile_col=512) and stream over the sequence length dimension. The load emits a
# stream with an extra leading singleton dimension (shape (1, 64, 1, 512)).
# We flatten the two stream dimensions into a single one to obtain the required
# shape (64, 1, 512). The RMS‑norm is expressed entirely with DSL ops:
#   * square the input (`binary_mul`),
#   * sum across the hidden dimension (`unary_rowwise_sum`),
#   * compute the mean by scaling with 1/hidden_dim (`unary_mul_imm`),
#   * add epsilon (`unary_add_imm`),
#   * take the reciprocal square‑root (`unary_rsqrt`),
#   * finally multiply the original tensor by this factor (`binary_mul`).
# All tensor manipulations use DSL primitives, preserving the streaming shape.

def pre_attention_norm(input_tensor, *, out_shapes, out_perms=None):
    # Load the off‑chip tensor as a stream of 1×hidden_dim tiles.
    # stride=1 advances across the sequence dimension.
    x = offchip_load(
        input_tensor,
        stride=[1],
        out_shape_tiled=[64],
        tile_row=1,
        tile_col=512,
    )
    # Merge the leading singleton stream dimension with the sequence stream.
    x = flatten(x, min_rank=0, max_rank=1)   # shape: (64, 1, 512)

    # x²
    x_sq = binary_mul(x, x)

    # Sum over hidden dimension (tile column) → shape (64, 1, 1)
    sum_sq = unary_rowwise_sum(x_sq)

    # Mean = sum / hidden_dim (512)
    mean_sq = unary_mul_imm(sum_sq, 1.0 / 512.0)

    # Add epsilon for numerical stability.
    eps = 1e-6
    mean_eps = unary_add_imm(mean_sq, eps)

    # rsqrt(mean + eps)
    rsqrt_mean = unary_rsqrt(mean_eps)

    # RMS‑norm: x * rsqrt(mean + eps)
    out = binary_mul(x, rsqrt_mean)

    return out