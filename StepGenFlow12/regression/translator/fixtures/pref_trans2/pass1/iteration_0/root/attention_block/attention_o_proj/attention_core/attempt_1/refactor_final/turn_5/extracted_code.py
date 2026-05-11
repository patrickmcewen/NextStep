# ----------------------------------------------------------------------
# Implementation notes
# ----------------------------------------------------------------------
# Q, K, V are on‑chip streams:
#   Q : (1, 64) × tile (16, 32)   # seq_len=64, num_heads=16, head_dim=32
#   K : (1, 64) × tile ( 4, 32)   # seq_len=64, num_kv_heads=4, head_dim=32
#   V : (1, 64) × tile ( 4, 32)
#
# The child `attention_compute` expects its arguments in *vanilla* shape:
#   Qh : (4, 4, 64, 32)           # (kv_heads, heads_per_kv, seq_len, head_dim)
#   Kh : (4, 1, 64, 32)
#   Vh : (4, 1, 64, 32)
#
# We obtain those layouts by loading the tensors with `offchip_load`,
# specifying a stride that maps the desired (kv, heads_per_kv, seq) indices
# onto the underlying (seq, head_row) layout, and then folding the sequence
# dimension into the tile‑row side with `accum_retile_row`.
#
#   Q:  stride = (4, 1, 16)   → out_shape_tiled = (kv=4, heads_per_kv=4, seq=64)
#   K/V: stride = (1, 4)      → out_shape_tiled = (kv=4, seq=64)
#
# After the loads we have tile‑row = 1; `accum_retile_row(rank=1)` merges the
# last stream dimension (seq) into the tile‑row, yielding tile‑row = 64.
# The resulting streams match the vanilla shapes expected by the blackbox.
# ----------------------------------------------------------------------
def attention_core(Q, K, V, *, out_shapes, out_perms=None):
    # Q → (kv=4, heads_per_kv=4, seq=64)  × tile (1, 32) → tile rows = 64
    Qh = offchip_load(
        Q,
        stride=(4, 1, 16),
        out_shape_tiled=(4, 4, 64),
        tile_row=1,
        tile_col=32,
    )
    Qh = accum_retile_row(Qh, rank=1)  # (1,4,4) × tile (64,32)

    # K → (kv=4, seq=64) × tile (1, 32) → tile rows = 64
    Kh = offchip_load(
        K,
        stride=(1, 4),
        out_shape_tiled=(4, 64),
        tile_row=1,
        tile_col=32,
    )
    Kh = accum_retile_row(Kh, rank=1)  # (1,4) × tile (64,32)

    # V → (kv=4, seq=64) × tile (1, 32) → tile rows = 64
    Vh = offchip_load(
        V,
        stride=(1, 4),
        out_shape_tiled=(4, 64),
        tile_row=1,
        tile_col=32,
    )
    Vh = accum_retile_row(Vh, rank=1)  # (1,4) × tile (64,32)

    # Core attention computation (blackbox)
    attn = attention_compute(
        Qh, Kh, Vh,
        out_shapes=out_shapes,
        out_perms=out_perms,
    )
    return attn