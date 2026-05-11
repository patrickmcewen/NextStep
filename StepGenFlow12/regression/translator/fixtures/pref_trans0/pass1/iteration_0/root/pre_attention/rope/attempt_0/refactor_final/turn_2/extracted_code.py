# Implementation reasoning:
# • `cos` and `sin` are RAW tensors, so we stream them on‑chip with `offchip_load`.
#   The underlying layout is (seq_len, 1, head_dim); we stream over the
#   sequence dimension with tiles (1, head_dim) and then collapse the leading
#   singleton stream dim using `flatten`, giving shape (seq_len,)×tile(1,head_dim).
# • RoPE’s rotate‑half operation can be expressed without any Python slicing or
#   `torch.cat` by:
#     1. Splitting the column dimension into two halves with `retile_streamify`,
#        which turns the two halves into a doubled stream dimension.
#     2. Using `parallelize` to separate the two halves into distinct streams.
#     3. Negating the original second half (`unary_mul_imm`) and then swapping
#        the order by interleaving the streams with `static_reassemble`.
#     4. Converting the interleaved stream back to a (seq_len, 2, …) shape with
#        `reshape_stream`.
#     5. Merging the “2” stream dimension into the column tile dimension via
#        `accum_retile_col`, which yields the rotated tensor.
# • The RoPE formula `out = x * cos + rotate_half(x) * sin` is then built with
#   the binary compute DSL ops.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # ------------------------------------------------------------------
    # Load RAW rotary embeddings onto‑chip and flatten to a single stream.
    # ------------------------------------------------------------------
    tile_row = 1
    tile_col = Q.shape[-1]                     # head_dim = 32
    cos_loaded = flatten(
        offchip_load(cos, stride=[1], out_shape_tiled=[64],
                     tile_row=tile_row, tile_col=tile_col),
        min_rank=0,
        max_rank=1,
    )
    sin_loaded = flatten(
        offchip_load(sin, stride=[1], out_shape_tiled=[64],
                     tile_row=tile_row, tile_col=tile_col),
        min_rank=0,
        max_rank=1,
    )

    # ------------------------------------------------------------------
    # Helper implementing the `_rotate_half` logic using only DSL ops.
    # ------------------------------------------------------------------
    def _rotate_half(x):
        half = x.shape[-1] // 2                     # Python scalar (16)

        # 1️⃣ Split the column dimension into two halves -> doubled stream.
        split = retile_streamify(x, chunk=half, split_row=False)   # (2*S, R, half)

        # 2️⃣ Separate the two halves into distinct streams.
        halves = parallelize(split, 2)                # each (S, R, half)
        first_half, second_half = halves[0], halves[1]

        # 3️⃣ Negate the original second half (will become the first half).
        second_half_neg = unary_mul_imm(second_half, -1.0)

        # 4️⃣ Interleave streams in swapped order: negated second half then first half.
        interleaved = static_reassemble([second_half_neg, first_half])  # (2*S, R, half)

        # 5️⃣ Reshape stream dim back to (S, 2, R, half).
        interleaved = reshape_stream(interleaved, chunk_size=2, rank=0)

        # 6️⃣ Merge the “2‑half” stream into the column tile dimension.
        out = accum_retile_col(interleaved)          # (S, R, 2*half) = (S, R, C)
        return out

    # ------------------------------------------------------------------
    # Apply RoPE to Q.
    # ------------------------------------------------------------------
    Q_cos = binary_mul(Q, cos_loaded)                # Q * cos
    Q_rot = _rotate_half(Q)                          # rotate_half(Q)
    Q_rot_sin = binary_mul(Q_rot, sin_loaded)        # rotate_half(Q) * sin
    Q_out = binary_add(Q_cos, Q_rot_sin)             # final Q_out

    # ------------------------------------------------------------------
    # Apply RoPE to K.
    # ------------------------------------------------------------------
    K_cos = binary_mul(K, cos_loaded)                # K * cos
    K_rot = _rotate_half(K)                          # rotate_half(K)
    K_rot_sin = binary_mul(K_rot, sin_loaded)        # rotate_half(K) * sin
    K_out = binary_add(K_cos, K_rot_sin)             # final K_out

    return Q_out, K_out