# Implementation reasoning:
# - `cos` and `sin` are RAW tensors; they must be loaded into the on‑chip stream domain before any DSL consumer.
#   We use `offchip_load_ref` with `Q` (or `K`, they have the same stream shape) as the reference
#   to broadcast the loaded tensors to shape (seq_len, 1, head_dim).
# - Rotation of the last dimension by half is performed by the helper `_rotate_half`,
#   which uses standard PyTorch `torch.cat` and negation. This is a shape‑changing operation
#   that is not covered by the DSL primitives, but it mirrors the reference implementation.
# - Element‑wise multiplication and addition are expressed with the DSL compute ops
#   `binary_mul` and `binary_add`.  Broadcast semantics (`tile_r=1` for `cos`/`sin`) are
#   handled automatically by the DSL.
# - The final rotated Q and K tensors are returned as a tuple, matching the required
#   output shapes `(seq_len, num_heads, head_dim)` and `(seq_len, num_kv_heads, head_dim)`.

def apply_rope(Q, K, cos, sin, *, out_shapes, out_perms=None):
    # Helper to rotate the last dimension by half (as in the reference implementation)
    def _rotate_half(x):
        half = x.shape[-1] // 2
        # Negate the second half and concatenate with the first half
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    # Load RAW positional embeddings into streams matching Q/K stream shape
    # `stride` and `out_shape_tiled` are empty because the tiling grid is 1×1 for these tensors.
    cos_stream = offchip_load_ref(
        Q,          # reference tensor to infer stream shape (seq_len,)
        cos,
        stride=[],                # no streaming stride needed
        out_shape_tiled=[],       # no additional streaming dimensions
        tile_row=cos.shape[-2],   # =1
        tile_col=cos.shape[-1],   # =head_dim (32)
    )
    sin_stream = offchip_load_ref(
        Q,
        sin,
        stride=[],
        out_shape_tiled=[],
        tile_row=sin.shape[-2],   # =1
        tile_col=sin.shape[-1],   # =head_dim (32)
    )

    # Rotate Q and K halves
    Q_rot = _rotate_half(Q)
    K_rot = _rotate_half(K)

    # Compute rope‑applied Q
    Q_cos = binary_mul(Q, cos_stream, compute_bw=1)
    Q_sin = binary_mul(Q_rot, sin_stream, compute_bw=1)
    Q_out = binary_add(Q_cos, Q_sin, compute_bw=1)

    # Compute rope‑applied K
    K_cos = binary_mul(K, cos_stream, compute_bw=1)
    K_sin = binary_mul(K_rot, sin_stream, compute_bw=1)
    K_out = binary_add(K_cos, K_sin, compute_bw=1)

    return Q_out, K_out