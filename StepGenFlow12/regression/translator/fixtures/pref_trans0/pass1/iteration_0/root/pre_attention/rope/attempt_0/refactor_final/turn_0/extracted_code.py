# Implementation reasoning:
# - `cos` and `sin` are RAW (off‑chip) tensors, so they must be streamed in with
#   `offchip_load`.  The underlying layout is (seq_len, 1, head_dim) and we
#   stream over the sequence dimension with tiles (1, head_dim).  After the load
#   the tensor has a leading singleton stream dimension (shape (1, 64, 1, 32));
#   we collapse that dimension with `flatten(..., min_rank=0, max_rank=1)` so
#   the resulting stream shape matches the on‑chip tensors Q/K ((64, *, *)).
# - `_rotate_half` reproduces the PyTorch helper used in the reference model:
#   it splits the last dimension in half, negates the second half, and swaps
#   the halves.  This logic is kept as a small local helper; the surrounding
#   computation is expressed entirely with DSL calls.
# - The RoPE formula is then applied with the DSL binary ops:
#       Q_out = Q * cos + rotate_half(Q) * sin
#       K_out = K * cos + rotate_half(K) * sin
# - The function returns the two on‑chip streams; the caller will handle any
#   downstream consumers or off‑chip store.
def rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # Load RAW tensors into on‑chip streams (seq_len = 64, tile_row = 1, tile_col = 32)
    cos_loaded = flatten(
        offchip_load(cos, stride=[1], out_shape_tiled=[64], tile_row=1, tile_col=32),
        min_rank=0,
        max_rank=1,
    )
    sin_loaded = flatten(
        offchip_load(sin, stride=[1], out_shape_tiled=[64], tile_row=1, tile_col=32),
        min_rank=0,
        max_rank=1,
    )

    # Helper that matches the reference `_rotate_half` implementation
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    # Apply the rotation to Q and K
    Q_rot = _rotate_half(Q)
    K_rot = _rotate_half(K)

    # RoPE: Q_out = Q * cos + rotate_half(Q) * sin
    Q_out = binary_add(binary_mul(Q, cos_loaded), binary_mul(Q_rot, sin_loaded))
    # RoPE: K_out = K * cos + rotate_half(K) * sin
    K_out = binary_add(binary_mul(K, cos_loaded), binary_mul(K_rot, sin_loaded))

    return Q_out, K_out