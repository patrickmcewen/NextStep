# The rope operation requires element‑wise multiplication of Q/K with the positional
# embeddings (cos, sin) plus the same multiplication after rotating the last
# dimension by half.  The raw cos/sin tensors must be streamed onto‑chip first.
# We load them with `offchip_load`, which yields a stream shape (1, seq_len, …).
# A `flatten` merges the leading singleton stream dimension with the sequence
# dimension, giving the desired shape (seq_len, 1, head_dim).  The rotation is
# performed by a helper that swaps the two halves of the last dimension and
# negates the moved half (using `unary_mul_imm`).  All arithmetic is expressed
# with the DSL compute ops `binary_mul` and `binary_add`.  The function returns
# the rope‑applied Q and K tensors with the required output shapes.
def apply_rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # Helper that rotates the last dimension by half and negates the wrapped half.
    def _rotate_half(x):
        half = x.shape[-1] // 2
        # Negate second half using a DSL unary multiplication.
        second_half = x[..., half:]
        neg_second = unary_mul_imm(second_half, -1.0, compute_bw=1)
        first_half = x[..., :half]
        # Concatenate the halves (this cat is a pure layout change, not arithmetic).
        return torch.cat([neg_second, first_half], dim=-1)

    # Load RAW `cos` and `sin` tensors onto‑chip.
    # We stream across the sequence dimension (out_shape_tiled = [seq_len])
    # with stride 1, then flatten away the leading singleton stream dim.
    seq_len = cos.shape[0]  # same as sin.shape[0] and Q.shape[0]

    cos_loaded = offchip_load(
        cos,
        stride=[1],
        out_shape_tiled=[seq_len],
        tile_row=cos.shape[-2],   # 1
        tile_col=cos.shape[-1],   # head_dim (32)
    )
    cos_stream = flatten(cos_loaded, min_rank=0, max_rank=1)

    sin_loaded = offchip_load(
        sin,
        stride=[1],
        out_shape_tiled=[seq_len],
        tile_row=sin.shape[-2],
        tile_col=sin.shape[-1],
    )
    sin_stream = flatten(sin_loaded, min_rank=0, max_rank=1)

    # Apply rope to Q.
    Q_cos = binary_mul(Q, cos_stream, compute_bw=1)
    Q_sin = binary_mul(_rotate_half(Q), sin_stream, compute_bw=1)
    Q_out = binary_add(Q_cos, Q_sin, compute_bw=1)

    # Apply rope to K.
    K_cos = binary_mul(K, cos_stream, compute_bw=1)
    K_sin = binary_mul(_rotate_half(K), sin_stream, compute_bw=1)
    K_out = binary_add(K_cos, K_sin, compute_bw=1)

    return Q_out, K_out